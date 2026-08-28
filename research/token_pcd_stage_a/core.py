from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention


TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_open_drawer",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "google_robot_place_apple_in_closed_top_drawer",
)
ACTION_DIM = 7
VISUAL_TOKENS = 256
PATCH_GRID = 16
ALPHA = 0.8


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().float().numpy().tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mask_patch_overlaps(mask: np.ndarray, size: int = 224) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    if size % PATCH_GRID:
        raise ValueError(f"Image size {size} is not divisible by {PATCH_GRID}")
    patch = size // PATCH_GRID
    return resized.reshape(PATCH_GRID, patch, PATCH_GRID, patch).mean(axis=(1, 3)).reshape(-1)


def token_ids_from_mask(mask: np.ndarray, threshold: float) -> tuple[list[int], np.ndarray]:
    overlaps = mask_patch_overlaps(mask)
    return np.flatnonzero(overlaps >= threshold).astype(int).tolist(), overlaps


def matched_random_ids(object_ids: Sequence[int], seed: int, token_count: int = VISUAL_TOKENS) -> list[int]:
    object_set = set(int(index) for index in object_ids)
    candidates = np.asarray([index for index in range(token_count) if index not in object_set])
    if len(object_ids) > len(candidates):
        raise ValueError("Not enough non-object tokens for matched random control")
    return sorted(np.random.default_rng(seed).choice(candidates, size=len(object_ids), replace=False).astype(int).tolist())


def ensure_empty_action_token(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    if torch.all(input_ids[:, -1] == 29871):
        return input_ids, attention_mask
    empty = torch.full((1, 1), 29871, dtype=input_ids.dtype, device=input_ids.device)
    visible = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat((input_ids, empty), dim=1), torch.cat((attention_mask, visible), dim=1)


@torch.inference_mode()
def clean_action_ids(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids, attention_mask = ensure_empty_action_token(inputs)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs["pixel_values"],
        max_new_tokens=ACTION_DIM,
        min_new_tokens=None,
        eos_token_id=None,
        do_sample=False,
    )
    ids = output[:, -ACTION_DIM:]
    if ids.shape != (1, ACTION_DIM):
        raise RuntimeError(f"Expected seven clean action tokens, got {tuple(ids.shape)}")
    return ids


@torch.inference_mode()
def teacher_forced_logits(
    model: Any,
    inputs: dict[str, torch.Tensor],
    clean_ids: torch.Tensor,
    selected_indices: Sequence[int] = (),
    replacement_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Any]:
    base_ids, base_mask = ensure_empty_action_token(inputs)
    teacher_ids = torch.cat((base_ids, clean_ids[:, :-1]), dim=1)
    teacher_mask = torch.cat((base_mask, torch.ones_like(clean_ids[:, :-1], dtype=base_mask.dtype)), dim=1)
    with projector_intervention(model, selected_indices, replacement_mean) as trace:
        output = model(
            input_ids=teacher_ids,
            attention_mask=teacher_mask,
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            return_dict=True,
        )
    if trace.before is None or trace.before.shape[1] != VISUAL_TOKENS:
        raise RuntimeError(f"Expected 256 projector tokens, got {None if trace.before is None else trace.before.shape}")
    query_indices = [VISUAL_TOKENS + base_ids.shape[1] - 1 + offset for offset in range(ACTION_DIM)]
    logits = output.logits[0, query_indices].detach().float().cpu()
    if logits.shape[0] != ACTION_DIM or not torch.isfinite(logits).all():
        raise RuntimeError("Invalid teacher-forced logits")
    return logits, trace


def action_token_slice(model: Any) -> slice:
    # OpenVLA maps 256 action bins to vocabulary IDs vocab_size-256 .. vocab_size-1.
    return slice(int(model.vocab_size) - 256, int(model.vocab_size))


def direction_metrics(model: Any, clean: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    token_slice = action_token_slice(model)
    return torch.softmax(clean[:, token_slice], dim=-1) - torch.softmax(branch[:, token_slice], dim=-1)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left.flatten().double(), right.flatten().double()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator) if denominator > 0 else float("nan")


def pcd_ids(clean: torch.Tensor, negative: torch.Tensor, alpha: float = ALPHA) -> torch.Tensor:
    final = clean.clone()
    final[:-1] = (1.0 + alpha) * clean[:-1] - alpha * negative[:-1]
    return final.argmax(dim=-1)


def decode_action_ids(model: Any, action_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    token_ids = action_ids.detach().cpu().numpy()
    discretized = np.clip(model.vocab_size - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
