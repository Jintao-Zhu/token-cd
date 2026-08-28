"""Small, dependency-light CoreAct metric and integrity helpers.

These functions are intentionally independent of the policy so that metric
semantics can be unit-tested before any checkpoint is loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np


def intervention_metrics(v_pos, v_neg, u_target, valid_mask=None, eps=1e-8):
    """Return I_G, Q_G and Q_rel over valid action elements."""
    vp, vn, ut = (np.asarray(x, dtype=np.float64) for x in (v_pos, v_neg, u_target))
    if vp.shape != vn.shape or vp.shape != ut.shape:
        raise ValueError("velocity and target tensors must have identical shapes")
    mask = np.ones(vp.shape, dtype=bool) if valid_mask is None else np.broadcast_to(valid_mask, vp.shape)
    if not np.any(mask):
        raise ValueError("valid mask contains no elements")
    denom = float(np.mean(vp[mask] ** 2)) + eps
    base = float(np.mean((vp[mask] - ut[mask]) ** 2))
    intervention = float(np.mean((vn[mask] - ut[mask]) ** 2))
    i_g = float(np.mean((vp[mask] - vn[mask]) ** 2) / denom)
    q_g = intervention - base
    return {"I_G": i_g, "Q_G": q_g, "Q_rel": q_g / (base + eps), "base_mse": base}


def deterministic_groups(n_tokens: int, n_groups: int, seed: int) -> list[list[int]]:
    if n_tokens < n_groups or n_tokens <= 0:
        raise ValueError("not enough eligible tokens")
    rng = np.random.default_rng(seed)
    return [[int(i)] for i in rng.choice(n_tokens, size=n_groups, replace=False)]


def split_manifest(rows: Iterable[Mapping], split_names=("mean_calibration", "development", "heldout")):
    """Reject episode overlap and duplicate state keys in a manifest."""
    rows = list(rows)
    by_episode: dict[str, set[str]] = {}
    seen = set()
    for row in rows:
        split = row.get("split")
        if split not in split_names:
            raise ValueError(f"unknown split {split!r}")
        episode = str(row["episode_id"])
        key = (episode, int(row["frame_id"]))
        if key in seen:
            raise ValueError(f"duplicate state {key}")
        seen.add(key)
        by_episode.setdefault(episode, set()).add(split)
    overlap = {ep: sorted(splits) for ep, splits in by_episode.items() if len(splits) > 1}
    if overlap:
        raise ValueError(f"episode leakage: {overlap}")
    return {name: sum(row.get("split") == name for row in rows) for name in split_names}


def hierarchical_cluster_bootstrap(
    rows: Iterable[Mapping],
    value_key: str,
    replicates: int,
    seed: int,
    statistic: Callable = np.median,
) -> np.ndarray:
    """Bootstrap state-level values by task, then episode, never by flow row."""
    rows = list(rows)
    if replicates <= 0 or not rows:
        raise ValueError("bootstrap requires rows and a positive replicate count")
    state_keys = [(row["task_id"], row["episode_id"], row["state_id"]) for row in rows]
    if len(state_keys) != len(set(state_keys)):
        raise ValueError("bootstrap input must contain one row per state, not flow rows")
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_id"]), {}).setdefault(str(row["episode_id"]), []).append(
            float(row[value_key])
        )
    tasks = sorted(grouped)
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        values = []
        for task in rng.choice(tasks, size=len(tasks), replace=True):
            episodes = sorted(grouped[task])
            for episode in rng.choice(episodes, size=len(episodes), replace=True):
                values.extend(grouped[task][episode])
        output[replicate] = statistic(np.asarray(values, dtype=np.float64))
    return output


def signed_effect_interpretation(metrics: Mapping, target_semantics_valid: bool) -> dict:
    """Prevent Q from being interpreted when the demonstration target is invalid."""
    if not target_semantics_valid:
        return {
            "target_semantics_valid": False,
            "I_G": float(metrics["I_G"]),
            "Q_G": None,
            "Q_rel": None,
            "label": "invalid_shuffled_demonstration_target",
        }
    q_rel = float(metrics["Q_rel"])
    label = "anchor" if q_rel > 0.05 else "nuisance_candidate" if q_rel < -0.05 else "sign_uncertain"
    return {
        "target_semantics_valid": True,
        "I_G": float(metrics["I_G"]),
        "Q_G": float(metrics["Q_G"]),
        "Q_rel": q_rel,
        "label": label,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
