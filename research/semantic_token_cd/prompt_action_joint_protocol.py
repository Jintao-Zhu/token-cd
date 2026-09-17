"""Locked protocol for conservative Prompt/Action selection v1."""
from __future__ import annotations

from research.semantic_token_cd.prompt_action_rerank_protocol import (
    ACTION_LAYERS, CANONICAL, LAMBDA, MATCHED_ROOT, PCD_SOURCE, PROMPT_LAYERS,
    REPO, SEEDS, TASKS, atomic_json,
)

PROTOCOL = "PA_CONSTRAINED_JOINT_L11_V1"
ARTIFACT = REPO / "artifacts/prompt_action_constrained_joint_l11_v1"
ARMS = ("pa_a10", "pa_joint10")
ARM_MODE = {"pa_a10": "action_only", "pa_joint10": "joint"}
CANDIDATE_MULTIPLIER = 2
PROTECTED_FRACTION = 0.9
