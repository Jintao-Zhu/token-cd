"""Map LIBERO instance segmentation to spatially ordered connector token footprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

from research.coreact_exploration.instrumentation import PrefixToken


GRID_SIZE = 8
SOURCE_IMAGE_SIZE = 256
CELL_SIZE = SOURCE_IMAGE_SIZE // GRID_SIZE


@dataclass(frozen=True)
class TokenRegion:
    camera_id: str
    visual_token_index: int
    row: int
    column: int
    region: str
    relevant_fraction: float
    protected_fraction: float
    background_fraction: float


def instance_label_map(instances_to_ids: dict) -> dict[str, int]:
    """Robosuite instance segmentation uses insertion-order index plus one; zero is background."""
    return {name: index + 1 for index, name in enumerate(instances_to_ids)}


def protected_instance_names(instances_to_ids: dict) -> set[str]:
    markers = ("panda", "gripper", "mount")
    return {name for name in instances_to_ids if any(marker in name.casefold() for marker in markers)}


def token_regions(
    segmentation: np.ndarray,
    *,
    camera_id: str,
    label_by_name: dict[str, int],
    relevant_names: set[str],
    protected_names: set[str],
    relevant_threshold: float = 0.05,
) -> list[TokenRegion]:
    if segmentation.shape == (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE, 1):
        segmentation = segmentation[..., 0]
    if segmentation.shape != (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE):
        raise ValueError(f"expected 256x256 segmentation, got {segmentation.shape}")
    relevant_labels = {label_by_name[name] for name in relevant_names if name in label_by_name}
    protected_labels = {label_by_name[name] for name in protected_names if name in label_by_name}
    if not relevant_labels:
        raise ValueError("none of the task-relevant instances appear in segmentation metadata")
    output = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            cell = segmentation[
                row * CELL_SIZE : (row + 1) * CELL_SIZE,
                column * CELL_SIZE : (column + 1) * CELL_SIZE,
            ]
            relevant = float(np.isin(cell, list(relevant_labels)).mean())
            protected = float(np.isin(cell, list(protected_labels)).mean())
            background = float((cell == 0).mean())
            if protected > 0:
                region = "protected"
            elif relevant >= relevant_threshold:
                region = "relevant"
            elif relevant == 0:
                region = "background"
            else:
                region = "ambiguous"
            output.append(
                TokenRegion(
                    camera_id=camera_id,
                    visual_token_index=row * GRID_SIZE + column,
                    row=row,
                    column=column,
                    region=region,
                    relevant_fraction=relevant,
                    protected_fraction=protected,
                    background_fraction=background,
                )
            )
    return output


def attach_prefix_indices(span_map: list[PrefixToken], regions: list[TokenRegion]) -> list[dict]:
    lookup = {(region.camera_id, region.visual_token_index): region for region in regions}
    output = []
    for token in span_map:
        if token.modality != "visual" or token.visual_token_index is None:
            continue
        key = (token.camera_id, token.visual_token_index)
        if key not in lookup:
            raise ValueError(f"missing region mapping for {key}")
        output.append({"prefix_index": token.index, **asdict(lookup[key])})
    if len(output) != len(regions):
        raise ValueError(f"mapped {len(output)} prefix tokens for {len(regions)} regions")
    return output


def select_region_groups(
    mapped: list[dict],
    scores: torch.Tensor,
    *,
    cap: int,
    minimum: int,
    random_seed: int,
) -> tuple[dict[str, list[int]], int]:
    relevant = [row["prefix_index"] for row in mapped if row["region"] == "relevant"]
    background = [row["prefix_index"] for row in mapped if row["region"] == "background"]
    count = min(cap, len(relevant), len(background))
    if count < minimum:
        raise ValueError(
            f"insufficient region tokens for matched groups: relevant={len(relevant)} "
            f"background={len(background)} minimum={minimum}"
        )
    cpu_scores = scores.detach().float().cpu()
    relevant_sorted = sorted(relevant, key=lambda index: (-float(cpu_scores[index]), index))
    relevant_low = sorted(relevant, key=lambda index: (float(cpu_scores[index]), index))
    background_high = sorted(background, key=lambda index: (-float(cpu_scores[index]), index))
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    random_order = torch.randperm(len(background), generator=generator)[:count].tolist()
    groups = {
        "relevant_high": relevant_sorted[:count],
        "background_high": background_high[:count],
        "relevant_low": relevant_low[:count],
        "background_random": [background[index] for index in random_order],
    }
    return groups, count


def mapping_sha256(mapped: list[dict]) -> str:
    payload = json.dumps(mapped, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
