#!/usr/bin/env python3
"""Shared primitives for the SAME_STATE_FORK_V1 mechanism diagnostic.

Design contract (frozen in artifacts/same_state_fork_v1/FROZEN_PROTOCOL.json):
  * A fork starts from an exactly restored simulator state that was reached by
    replaying a stored vanilla action sequence.
  * Three branches are compared at that state: clean (vanilla action),
    reconstruction-only (independently decoded with harmonic reconstruction),
    guided (frozen matched method).
  * Every branch executes ``d`` intervention steps and then hands control back
    to the vanilla policy for the remainder of the episode.
  * Per-dimension token analysis always uses the SAME decoding prefix.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

# Frozen method constants (must match libero_matched_rollout.py).
ATTENTION_LAYERS = (11,)
LAMBDA = 0.5
K = 8
KMEANS_SEED = 0

# Stage fractions of a vanilla episode, pre-registered before seeing results.
STAGE_FRACTIONS = (("approach", 0.25), ("manipulation", 0.55), ("finishing", 0.85))

# Intervention durations (policy decision steps).
DURATIONS = (1, 5, 15, 30, 50, -1)  # -1 == remaining


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def code_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def select_stage_steps(episode_len: int) -> list[dict]:
    """Deterministic, pre-registered stage selection.

    Missing stages are recorded as missing; no arbitrary frame is substituted.
    """
    stages = []
    for name, frac in STAGE_FRACTIONS:
        step = int(round(frac * episode_len))
        if step <= 0 or step >= episode_len:
            stages.append({"stage": name, "fraction": frac, "step": None, "status": "missing"})
        else:
            stages.append({"stage": name, "fraction": frac, "step": step, "status": "ok"})
    return stages


def resolve_duration(duration: int, remaining: int) -> int:
    if duration < 0:
        return int(remaining)
    return int(min(duration, remaining))


def action_deltas(reference: np.ndarray, other: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=float)
    other = np.asarray(other, dtype=float)
    return {
        "translation_delta": (other[:3] - reference[:3]).tolist(),
        "rotation_delta": (other[3:6] - reference[3:6]).tolist(),
        "gripper_delta": float(other[6] - reference[6]),
        "l2": float(np.linalg.norm(other - reference)),
    }


def action_bin_margin(logits: torch.Tensor, action_start: int) -> dict:
    """Top-1 margin inside the 256 action bins, per action dimension.

    ``logits`` is [7, vocab] from a teacher-forced forward at a fixed prefix.
    """
    bins = logits[:, action_start:action_start + 256].float()
    top2 = torch.topk(bins, k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()
    token_ids = bins.argmax(dim=-1).detach().cpu().numpy()
    return {"margin": margin.tolist(), "bin_index": token_ids.tolist()}


def identity_reconstruction_baseline(model, processor, image, instruction, inputs, clean_ids, clean_logits, unnorm_key):
    """Measure the no-op reconstruction error against the clean forward.

    This is the numerical noise floor that any reported branch difference must
    exceed to be meaningful.
    """
    from research.semantic_token_cd.libero_policy import forward_logits

    empty = forward_logits(model, processor, image, instruction, clean_ids, selected=(), mean=None)
    diff = (empty - clean_logits).abs()
    action_start = int(model.vocab_size) - 256
    bins = diff[:, action_start:action_start + 256]
    return {
        "identity_max_abs_logit_diff": float(diff.max().item()),
        "identity_max_abs_action_bin_diff": float(bins.max().item()),
        "identity_argmax_match": bool(
            torch.equal(empty.argmax(dim=-1), clean_logits.argmax(dim=-1))
        ),
    }


def cluster_bootstrap_ci(values, cluster_ids, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Cluster (episode-level) bootstrap CI for a mean."""
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    unique = np.unique(cluster_ids)
    if unique.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    index = {c: np.flatnonzero(cluster_ids == c) for c in unique}
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index[c] for c in pick])
        means[i] = values[idx].mean()
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(values.size),
        "n_clusters": int(unique.size),
    }


def mcnemar_exact(a_success, b_success) -> dict:
    """Exact McNemar for paired binary outcomes."""
    from math import comb

    a = np.asarray(a_success, dtype=bool)
    b = np.asarray(b_success, dtype=bool)
    rescue = int(np.sum(a & ~b))
    harm = int(np.sum(~a & b))
    n = rescue + harm
    if n == 0:
        p = 1.0
    else:
        k = min(rescue, harm)
        p = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
        p = min(1.0, p)
    return {"rescue": rescue, "harm": harm, "net": rescue - harm, "p": float(p), "n_discordant": n}
