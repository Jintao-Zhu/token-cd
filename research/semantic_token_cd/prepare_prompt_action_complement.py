"""Freeze protocol configuration and audit all 400 reusable Original episodes."""
from __future__ import annotations

import hashlib
import json
import pickle

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.prompt_action_complement_protocol import (
    ACTION_LAYERS, ARMS, ARTIFACT, CANONICAL, LAMBDA, ORIGINAL_ROOT,
    PROMPT_LAYERS, PROTOCOL, SEEDS, TASKS, atomic_json,
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reused = {}
    for task in TASKS:
        count = 0
        for seed in SEEDS:
            result_path = ORIGINAL_ROOT[task] / f"episode_{seed:03d}_summary.json"
            row = json.loads(result_path.read_text())
            with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                expected = snapshot_sha(pickle.load(handle))
            checks = (
                row.get("canonical_snapshot_sha256") == expected,
                row.get("attention_layers") == [11], row.get("lambda") == LAMBDA,
                row.get("beta") == 0.0, row.get("all_guided_prefix") is True,
                row.get("all_non_target_bit_identical") is True,
            )
            if not all(checks): raise RuntimeError(f"Original reuse audit failed: {result_path}")
            count += 1
        reused[task] = {"count": count, "source": str(ORIGINAL_ROOT[task])}
    root = CANONICAL / "snapshots"
    lock = {
        "protocol": PROTOCOL, "tasks": list(TASKS), "seeds": [0, 99],
        "arms": list(ARMS), "prompt_attention_layers_zero_based": list(PROMPT_LAYERS),
        "action_attention_layers_zero_based": list(ACTION_LAYERS),
        "action_queries": "teacher-forced prediction positions A0..A5; gripper excluded",
        "core": "Top_(m-floor(m/4)) Prompt-L11", "supplement_r": "floor(m/4)",
        "lambda": LAMBDA, "beta": 0.0, "reconstruction": "16x16 four-neighbor Dirichlet harmonic",
        "closed_loop": {"per_task_per_arm": 100, "total": 1600,
                        "reused_original": 400, "new": 1200},
        "original_reuse_audit": reused,
        "policy_code_sha256": sha(CANONICAL.parent.parent / "research/semantic_token_cd/prompt_attn_shr_policy.py"),
    }
    atomic_json(ARTIFACT / "CONFIG_LOCK.json", lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__": main()
