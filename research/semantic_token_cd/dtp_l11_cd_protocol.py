"""Locked DTP-Fixed positive + L11-Matched harmonic negative protocol."""
from __future__ import annotations

from research.semantic_token_cd.prompt_action_rerank_protocol import (
    CANONICAL, MATCHED_ROOT, PCD_SOURCE, REPO, SEEDS, TASKS, atomic_json,
)

PROTOCOL = "DTP_FIXED_POSITIVE_L11_MATCHED_NEGATIVE_LAMBDA_SWEEP_V1"
ARTIFACT = REPO / "artifacts/dtp_fixed_positive_l11_matched_cd_v1"
LAMBDAS = (0.10, 0.25, 0.50)
ARMS = tuple(f"dtp_l11_lambda_{str(value).replace('.', 'p')}" for value in LAMBDAS)
ARM_TO_LAMBDA = dict(zip(ARMS, LAMBDAS))
DTP_LAYER = 11
DTP_K = 64
DTP_TAU = 0.5
DTP_MODE = "fixed_clean_dim0_single_mask"

