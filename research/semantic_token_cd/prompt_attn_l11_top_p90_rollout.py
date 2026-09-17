"""Closed-loop L11 visual Top-p 0.90 extension rollout."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from research.semantic_token_cd import prompt_attn_l11_top_p_rollout as base
from research.semantic_token_cd.prompt_attn_l11_count_rollout import TASKS
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


PROTOCOL = "PROMPT_ATTN_L11_VISUAL_TOP_P90_EXTENSION_V1"
ARM_THRESHOLDS = {"l11_top_p90": 0.90}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_config(artifact: Path, canonical: Path, matched: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    policy_file = repo / "research/semantic_token_cd/prompt_attn_shr_policy.py"
    rollout_file = Path(__file__).resolve()
    original_rollout_file = repo / "research/semantic_token_cd/prompt_attn_l11_top_p_rollout.py"
    existing_sweep = repo / "artifacts/prompt_attn_l11_top_p_v1"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    code = {
        "git_commit": commit,
        "policy_sha256": file_sha(policy_file),
        "rollout_sha256": file_sha(rollout_file),
        "reused_top_p_rollout_sha256": file_sha(original_rollout_file),
    }
    payload = {
        "protocol_id": PROTOCOL,
        "purpose": "extend the frozen L11 visual Top-p sweep with threshold 0.90",
        "extension_of": {
            "protocol_id": "PROMPT_ATTN_L11_VISUAL_TOP_P_V1",
            "artifact": str(existing_sweep),
            "existing_arms": {"l11_top_p75": 0.75, "l11_top_p80": 0.80, "l11_top_p85": 0.85},
        },
        "tasks": list(TASKS),
        "seeds_by_task": {task: list(range(100, 200)) for task in TASKS},
        "new_arms": ARM_THRESHOLDS,
        "new_episode_count": 400,
        "reference_arm": {"arm": "l11_matched", "artifact": str(matched)},
        "canonical_snapshot_artifact": str(canonical),
        "count_bounds": [16, 64],
        "visual_attention_normalization": "R_i / sum over exactly 256 visual-token scores",
        "code_version": code,
        "locked_downstream": {
            "attention_layer": 11,
            "query": "full instruction excluding special/template/padding tokens",
            "head_query_aggregation": "equal arithmetic mean",
            "tie_break": "ascending visual-token index",
            "spatial_postprocessing": False,
            "harmonic": "16x16 four-neighbor Dirichlet beta=0/gamma=1",
            "prefix": "shared clean greedy prefix",
            "lambda": 0.5,
            "guided_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper": "clean positive dimension 6",
            "sampling": False,
        },
    }
    artifact.mkdir(parents=True, exist_ok=True)
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("TopP90 extension config lock differs")
    atomic_json(path, payload)
    return code


base.PROTOCOL = PROTOCOL
base.ARM_THRESHOLDS = ARM_THRESHOLDS
base.ensure_config = ensure_config


if __name__ == "__main__":
    base.main()
