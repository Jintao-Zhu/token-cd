"""Pure helpers for the closed-loop causal token dataset pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np


PHASES = ("approach", "pre_grasp", "grasp_contact", "transport", "pre_place")
PRE_PLACE_RESTORE_MARGIN_M = 0.03


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_seed(base: int, task_id: int, init_id: int, replicate: int, purpose: int) -> int:
    """Derive stable, non-overlapping seeds without relying on Python hash()."""
    return base + task_id * 1_000_000 + init_id * 10_000 + replicate * 100 + purpose


def select_phase_index(progress: Sequence, target_phase: str) -> tuple[int, str, bool]:
    """Select a replan-boundary state using clean trajectory metadata only."""
    if not progress:
        raise ValueError("empty trajectory")
    if target_phase not in PHASES:
        raise ValueError(f"unknown phase {target_phase}")
    n = len(progress)
    grasped = [bool(row["grasped"]) for row in progress]
    predicate = [bool(row["predicate"]) for row in progress]
    first_grasp = next((i for i, value in enumerate(grasped) if value), None)
    first_done = next((i for i, value in enumerate(predicate) if value), None)

    if target_phase == "approach":
        return min(1, n - 1), target_phase, False
    if target_phase == "pre_grasp" and first_grasp is not None:
        return max(0, first_grasp - 2), target_phase, False
    if target_phase == "grasp_contact" and first_grasp is not None:
        return first_grasp, target_phase, False
    if target_phase == "transport" and first_grasp is not None:
        stop = first_done if first_done is not None else n
        candidates = list(range(first_grasp, max(first_grasp + 1, stop)))
        if candidates:
            index = min(candidates, key=lambda i: float(progress[i]["object_goal_distance"]))
            midpoint = candidates[len(candidates) // 2]
            return min(index, midpoint), target_phase, False
    if target_phase == "pre_place" and first_grasp is not None:
        stop = first_done if first_done is not None else n
        candidates = [i for i in range(first_grasp, stop) if grasped[i]]
        if candidates:
            restore_safe = [
                i for i in candidates
                if float(progress[i]["object_goal_distance"]) >= PRE_PLACE_RESTORE_MARGIN_M
            ]
            pool = restore_safe or candidates
            return min(pool, key=lambda i: float(progress[i]["object_goal_distance"])), target_phase, False

    fallback = min(n - 1, max(0, round((PHASES.index(target_phase) + 1) * (n - 1) / 6)))
    return fallback, f"normalized_progress_{PHASES.index(target_phase) + 1}_of_6", True


def stratified_token_groups(eligible: Sequence[int], scores: Sequence[float], seed: int) -> list[dict]:
    """Choose eight disjoint singleton groups spanning the attention distribution."""
    if len(eligible) < 8:
        raise ValueError(f"need at least 8 eligible visual tokens, got {len(eligible)}")
    ordered = sorted(eligible, key=lambda i: (-float(scores[i]), i))
    top = ordered[:3]
    remaining = [i for i in ordered if i not in top]
    center = len(ordered) // 2
    middle_pool = ordered[max(3, center - 8) : min(len(ordered) - 2, center + 8)]
    middle = [middle_pool[0], middle_pool[len(middle_pool) // 2]]
    low = list(reversed(ordered[-2:]))
    occupied = set(top + middle + low)
    random_pool = [i for i in remaining if i not in occupied]
    rng = np.random.default_rng(seed)
    random_token = int(rng.choice(random_pool))
    selected = [
        *(dict(stratum="top", token_indices=[i]) for i in top),
        *(dict(stratum="middle", token_indices=[i]) for i in middle),
        *(dict(stratum="low", token_indices=[i]) for i in low),
        dict(stratum="random", token_indices=[random_token]),
    ]
    for ordinal, row in enumerate(selected):
        row["group_id"] = f"g{ordinal:02d}_{row['stratum']}"
        row["attention_score"] = float(scores[row["token_indices"][0]])
    if len({row["token_indices"][0] for row in selected}) != 8:
        raise RuntimeError("candidate groups are not disjoint")
    return selected


def paired_effect(clean: Sequence[int], masked: Sequence[int]) -> dict:
    if len(clean) != len(masked) or not clean:
        raise ValueError("clean/masked outcomes must have equal nonzero length")
    differences = np.asarray(masked, dtype=float) - np.asarray(clean, dtype=float)
    tau = float(differences.mean())
    return {
        "tau": tau,
        "paired_differences": differences.astype(int).tolist(),
        "discordant_seeds": int(np.count_nonzero(differences)),
        "strong_anchor": tau < -0.4,
        "strong_nuisance": tau > 0.4,
        "neutral": abs(tau) <= 0.4,
    }
