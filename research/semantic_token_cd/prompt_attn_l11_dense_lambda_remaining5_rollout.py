#!/usr/bin/env python3
"""Run the existing audited L11 rollout under the remaining-five-task protocol."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import research.semantic_token_cd.prompt_attn_l11_dense_lambda_rollout as rollout
from research.semantic_token_cd.l11_dense_lambda_remaining5_protocol import (
    ARMS,
    ARM_LAMBDAS,
    ARTIFACT,
    CANONICAL,
    PROTOCOL,
    TASKS,
    atomic_json,
)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_config(artifact: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    payload = {
        "protocol_id": PROTOCOL,
        "created_date": "2026-09-15",
        "purpose": "replicate the 13-point L11-Matched lambda curve on the remaining five tasks",
        "tasks": list(TASKS),
        "seeds": [0, 99],
        "new_lambdas": list(ARM_LAMBDAS.values()),
        "reused_lambdas": [0.0, 0.5],
        "arms": ARM_LAMBDAS,
        "new_episodes": len(TASKS) * 100 * len(ARMS),
        "canonical_snapshot_artifact": str(CANONICAL),
        "locked_selector": {
            "mode": "prompt_attention",
            "layers": [11],
            "coverage": "own-state Standard-SHR matched m_t",
            "beta": 0.0,
        },
        "compute": {"gpus": [2, 3], "processes_per_gpu": 3},
        "code_version": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "policy_sha256": file_sha(repo / "research/semantic_token_cd/prompt_attn_shr_policy.py"),
            "base_rollout_sha256": file_sha(repo / "research/semantic_token_cd/prompt_attn_l11_dense_lambda_rollout.py"),
            "wrapper_sha256": file_sha(Path(__file__).resolve()),
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("remaining-five lambda config lock differs")
    atomic_json(path, payload)
    return payload["code_version"]


def main() -> None:
    rollout.ARMS = ARMS
    rollout.ARM_LAMBDAS = ARM_LAMBDAS
    rollout.ARTIFACT = ARTIFACT
    rollout.CANONICAL = CANONICAL
    rollout.PROTOCOL = PROTOCOL
    rollout.TASKS = TASKS
    rollout.atomic_json = atomic_json
    rollout.ensure_config = ensure_config
    rollout.main()


if __name__ == "__main__":
    main()
