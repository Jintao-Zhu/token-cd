"""Locked protocol for the pure L11-Matched dense lambda sweep."""
from __future__ import annotations

from pathlib import Path

from research.semantic_token_cd.prompt_action_rerank_protocol import (
    CANONICAL,
    MATCHED_ROOT,
    PCD_SOURCE,
    REPO,
    SEEDS,
    TASKS,
    atomic_json,
)


PROTOCOL = "PROMPT_ATTN_L11_MATCHED_DENSE_LAMBDA_SWEEP_V1"
ARTIFACT = REPO / "artifacts/prompt_attn_l11_matched_dense_lambda_0_99_v1"
LAMBDAS = (0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45, 0.55)


def lambda_arm(lambd: float) -> str:
    return f"l11_matched_lambda_{int(round(lambd * 100)):03d}"


ARMS = tuple(lambda_arm(lambd) for lambd in LAMBDAS)
ARM_LAMBDAS = dict(zip(ARMS, LAMBDAS, strict=True))


EXISTING_LAMBDA_ROOTS = {
    0.0: {
        task: REPO / "artifacts/prompt_attn_l11_lambda_heterogeneity_discovery_v1/episodes" / task / "l11_positive_only"
        for task in TASKS
    },
    0.25: {
        "google_robot_open_drawer": REPO / "artifacts/prompt_attn_l11_lambda_heterogeneity_discovery_v1/episodes/google_robot_open_drawer/l11_fixed_025",
        "google_robot_close_drawer": REPO / "artifacts/prompt_attn_l11_adaptive_lambda_4task_0_299_v1/episodes/google_robot_close_drawer/l11_fixed_025",
        "google_robot_pick_coke_can": REPO / "artifacts/prompt_attn_l11_adaptive_lambda_4task_0_299_v1/episodes/google_robot_pick_coke_can/l11_fixed_025",
        "google_robot_move_near": REPO / "artifacts/prompt_attn_l11_adaptive_lambda_4task_0_299_v1/episodes/google_robot_move_near/l11_fixed_025",
    },
    0.5: MATCHED_ROOT,
    0.6: {
        task: REPO / "artifacts/l11_matched_dtp_positive_lambda_060_075_v1/closed_loop/episodes" / task / "l11_matched_lambda_0p6"
        for task in TASKS
    },
    0.75: {
        task: REPO / "artifacts/l11_matched_dtp_positive_lambda_060_075_v1/closed_loop/episodes" / task / "l11_matched_lambda_0p75"
        for task in TASKS
    },
}

