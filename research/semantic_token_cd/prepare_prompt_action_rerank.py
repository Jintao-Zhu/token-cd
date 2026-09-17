"""Audit reusable Matched episodes, measure 3K saturation, and freeze v1 config."""
from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.prompt_action_rerank_protocol import (
    ACTION_LAYERS, ARM, ARTIFACT, CANDIDATE_MULTIPLIER, CANONICAL, LAMBDA,
    MATCHED_ROOT, PROMPT_LAYERS, PROTOCOL, SEEDS, TASKS, atomic_json,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    task_stats = {}
    all_m = []
    for task in TASKS:
        values = []
        for seed in SEEDS:
            path = MATCHED_ROOT[task] / f"episode_{seed:03d}_summary.json"
            row = json.loads(path.read_text())
            with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                expected = snapshot_sha(pickle.load(handle))
            checks = (
                row.get("canonical_snapshot_sha256") == expected,
                row.get("attention_layers") == [11],
                row.get("lambda") == LAMBDA,
                row.get("beta") == 0.0,
                row.get("all_guided_prefix") is True,
                row.get("all_non_target_bit_identical") is True,
            )
            if not all(checks):
                raise RuntimeError(f"Matched reuse audit failed: {path}")
            values.extend(int(step["m_t"]) for step in row["selector_trace"])
        array = np.asarray(values, dtype=np.int64)
        saturated = int(np.sum(CANDIDATE_MULTIPLIER * array >= 256))
        task_stats[task] = {
            "episodes": 100,
            "control_steps": int(array.size),
            "mean_m": float(array.mean()),
            "min_m": int(array.min()),
            "max_m": int(array.max()),
            "three_k_saturated_steps": saturated,
            "three_k_saturation_rate": saturated / int(array.size),
            "source": str(MATCHED_ROOT[task]),
        }
        all_m.extend(values)
    array = np.asarray(all_m, dtype=np.int64)
    saturation = float(np.mean(CANDIDATE_MULTIPLIER * array >= 256))
    if saturation >= 0.05:
        raise RuntimeError(f"3K saturation {saturation:.2%} is frequent; protocol must be revised")
    lock = {
        "protocol": PROTOCOL,
        "new_arm": ARM,
        "tasks": list(TASKS),
        "seeds": [0, 99],
        "new_episodes": 400,
        "selector": "Prompt-L11 Top-3K candidate pool; Action-L16-31 Top-K rerank",
        "prompt_attention_layers_zero_based": list(PROMPT_LAYERS),
        "action_attention_layers_zero_based": list(ACTION_LAYERS),
        "action_queries": "clean-positive teacher-forced A0..A5; gripper excluded",
        "candidate_multiplier": CANDIDATE_MULTIPLIER,
        "coverage": "own-state Standard-SHR K_t; final count exactly K_t",
        "lambda": LAMBDA,
        "beta": 0.0,
        "reconstruction": "16x16 four-neighbor Dirichlet harmonic",
        "negative_prefix": "shared clean positive prefix",
        "guidance": "1.5*z_positive - 0.5*z_negative for dimensions 0..5; clean gripper",
        "spatial_postprocessing": False,
        "matched_reuse_audit": task_stats,
        "all_control_steps": int(array.size),
        "all_mean_m": float(array.mean()),
        "all_three_k_saturated_steps": int(np.sum(CANDIDATE_MULTIPLIER * array >= 256)),
        "all_three_k_saturation_rate": saturation,
        "policy_code_sha256": sha(CANONICAL.parent.parent / "research/semantic_token_cd/prompt_attn_shr_policy.py"),
    }
    atomic_json(ARTIFACT / "CONFIG_LOCK.json", lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
