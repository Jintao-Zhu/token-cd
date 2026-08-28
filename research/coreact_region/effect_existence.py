"""Locked helpers for the segmentation-grounded effect-existence gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Sequence

import numpy as np
import torch

from research.coreact_region.region_mapping import (
    attach_prefix_indices,
    instance_label_map,
    protected_instance_names,
    token_regions,
)


HORIZONS = (10, 30, 60)
K_VALUES = (4, 8, 16)
TARGET_COVERAGE_THRESHOLD = 0.0


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_effect_groups(
    env,
    raw: dict,
    ranked: dict,
    *,
    random_seed: int,
) -> tuple[list[dict], dict[str, list[int]], dict]:
    """Build genuine target/relevant groups and count-matched background controls."""
    object_name, goal_name, _ = _task_entities(env)
    instances = env._env.env.model.instances_to_ids
    labels = instance_label_map(instances)
    protected = protected_instance_names(instances)
    target_regions = []
    task_relevant_regions = []
    for camera_id, segmentation_key in (
        ("camera1", "agentview_segmentation_instance"),
        ("camera2", "robot0_eye_in_hand_segmentation_instance"),
    ):
        target = token_regions(
            raw[segmentation_key],
            camera_id=camera_id,
            label_by_name=labels,
            relevant_names={object_name},
            protected_names=protected,
            relevant_threshold=TARGET_COVERAGE_THRESHOLD,
        )
        task_relevant = token_regions(
            raw[segmentation_key],
            camera_id=camera_id,
            label_by_name=labels,
            relevant_names={object_name, goal_name},
            protected_names=protected,
            relevant_threshold=0.05,
        )
        target_regions.extend(target)
        task_relevant_regions.extend(task_relevant)
    attached_target = attach_prefix_indices(ranked["span_map"], target_regions)
    attached_relevant = attach_prefix_indices(ranked["span_map"], task_relevant_regions)
    relevant_by_index = {row["prefix_index"]: row for row in attached_relevant}
    camera_rows = []
    for row in attached_target:
        task_row = relevant_by_index[row["prefix_index"]]
        row["target_fraction"] = row.pop("relevant_fraction")
        row["task_relevant_fraction"] = task_row["relevant_fraction"]
        row["target_region"] = row.pop("region")
        row["task_region"] = task_row["region"]
        camera_rows.append(row)

    target = [
        row["prefix_index"]
        for row in camera_rows
        if row["target_fraction"] > TARGET_COVERAGE_THRESHOLD
        and row["protected_fraction"] == 0
    ]
    relevant = [row["prefix_index"] for row in camera_rows if row["task_region"] == "relevant"]
    background = [
        row["prefix_index"]
        for row in camera_rows
        if row["task_relevant_fraction"] == 0 and row["protected_fraction"] == 0
    ]
    scores = ranked["scores"].detach().float().cpu()
    relevant = sorted(set(relevant), key=lambda index: (-float(scores[index]), index))
    target = sorted(set(target))
    background = sorted(set(background))
    groups: dict[str, list[int]] = {"full_target": target}
    for k in K_VALUES:
        if len(relevant) >= k:
            groups[f"relevant_high_k{k}"] = relevant[:k]
    rng = np.random.default_rng(random_seed)
    for name, group in list(groups.items()):
        if len(background) < len(group):
            continue
        groups[f"background_random_match_{name}"] = sorted(
            rng.choice(background, size=len(group), replace=False).tolist()
        )
    audit = {
        "object_name": object_name,
        "goal_name": goal_name,
        "target_token_count": len(target),
        "task_relevant_token_count": len(relevant),
        "eligible_background_token_count": len(background),
        "supported_k": [k for k in K_VALUES if len(relevant) >= k],
        "groups_sha256": canonical_json_sha256(groups),
    }
    return camera_rows, groups, audit


def summarize_progress(initial, snapshots: Sequence, horizon: int) -> dict:
    if horizon < 1 or len(snapshots) < horizon:
        raise ValueError(f"need {horizon} snapshots, got {len(snapshots)}")
    selected = snapshots[:horizon]
    min_eef = min(row.eef_object_distance for row in selected)
    min_goal = min(row.object_goal_distance for row in selected)
    reach = float(
        np.clip(
            (initial.eef_object_distance - min_eef) / max(initial.eef_object_distance, 1e-9),
            -1,
            1,
        )
    )
    transport = float(
        np.clip(
            (initial.object_goal_distance - min_goal) / max(initial.object_goal_distance, 1e-9),
            -1,
            1,
        )
    )
    grasp = float(any(row.grasped for row in selected))
    predicate = float(any(row.predicate for row in selected))
    return {
        "horizon": horizon,
        "initial": asdict(initial),
        "final": asdict(selected[-1]),
        "minimum_eef_object_distance": min_eef,
        "minimum_object_goal_distance": min_goal,
        "reach_progress": reach,
        "transport_progress": transport,
        "grasp_ever": bool(grasp),
        "predicate_ever": bool(predicate),
        "composite_progress": (reach + transport + grasp + predicate) / 4.0,
    }


def select_critical_indices(
    before: Sequence,
    after: Sequence,
    *,
    near_reach_threshold_m: float,
    near_place_threshold_m: float,
) -> list[dict]:
    if len(before) != len(after):
        raise ValueError("before/after progress lengths differ")
    selected = []
    first_grasp = next((i for i, row in enumerate(after) if row.grasped), None)
    if first_grasp is not None:
        selected.append({"phase": "pre_grasp", "state_index": first_grasp, "fallback": False})
    else:
        index = int(np.argmin([row.eef_object_distance for row in before]))
        if before[index].eef_object_distance <= near_reach_threshold_m:
            selected.append({"phase": "near_reach", "state_index": index, "fallback": True})

    first_predicate = next((i for i, row in enumerate(after) if row.predicate), None)
    place_candidates = [
        i for i, row in enumerate(before)
        if row.grasped and not row.predicate and (first_predicate is None or i <= first_predicate)
    ]
    if place_candidates:
        index = min(place_candidates, key=lambda i: before[i].object_goal_distance)
        phase = "pre_place" if any(row.predicate for row in after[index:]) else "near_place"
        if phase == "pre_place" or before[index].object_goal_distance <= near_place_threshold_m:
            selected.append({"phase": phase, "state_index": index, "fallback": phase != "pre_place"})
    return selected


def _task_entities(env):
    from research.coreact_region.segmented_runtime import task_entities

    return task_entities(env)
