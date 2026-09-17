"""Locked PA-Full-Joint-Top2K formal protocol."""
from research.semantic_token_cd.prompt_action_rerank_protocol import (
    ACTION_LAYERS, CANONICAL, LAMBDA, MATCHED_ROOT, PCD_SOURCE, PROMPT_LAYERS,
    REPO, SEEDS, TASKS, atomic_json,
)

PROTOCOL = "PA_FULL_JOINT_TOP2K_V1"
ARM = "pa_full_joint_top2k"
ARTIFACT = REPO / "artifacts/prompt_action_full_joint_top2k_v1"
CANDIDATE_MULTIPLIER = 2
