"""Locked high-lambda extension for pure L11 and DTP-positive/L11-negative CD."""
from __future__ import annotations

from research.semantic_token_cd.prompt_action_rerank_protocol import (
    CANONICAL, MATCHED_ROOT, PCD_SOURCE, REPO, SEEDS, TASKS, atomic_json,
)

PROTOCOL = "L11_MATCHED_AND_DTP_POSITIVE_HIGH_LAMBDA_V1"
ARTIFACT = REPO / "artifacts/l11_matched_dtp_positive_lambda_060_075_v1"
ARMS = (
    "l11_matched_lambda_0p6",
    "l11_matched_lambda_0p75",
    "dtp_l11_lambda_0p6",
    "dtp_l11_lambda_0p75",
)
ARM_CONFIG = {
    "l11_matched_lambda_0p6": {"family": "pure_l11", "lambda": 0.6},
    "l11_matched_lambda_0p75": {"family": "pure_l11", "lambda": 0.75},
    "dtp_l11_lambda_0p6": {"family": "dtp_l11", "lambda": 0.6},
    "dtp_l11_lambda_0p75": {"family": "dtp_l11", "lambda": 0.75},
}

