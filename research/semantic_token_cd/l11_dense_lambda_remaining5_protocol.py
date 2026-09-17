"""Locked protocol for the remaining-five-task L11-Matched lambda sweep."""
from __future__ import annotations

from pathlib import Path

from research.semantic_token_cd.prompt_action_rerank_protocol import atomic_json


REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
CANONICAL = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
ARTIFACT = REPO / "artifacts/prompt_attn_l11_matched_dense_lambda_remaining5_0_99_v1"
MATCHED = REPO / "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes"
PROTOCOL = "PROMPT_ATTN_L11_MATCHED_DENSE_LAMBDA_REMAINING5_V1"
TASKS = (
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
SEEDS = tuple(range(100))
NEW_LAMBDAS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.75)
ALL_LAMBDAS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.75)


def lambda_arm(value: float) -> str:
    return f"l11_matched_lambda_{int(round(value * 100)):03d}"


ARMS = tuple(lambda_arm(value) for value in NEW_LAMBDAS)
ARM_LAMBDAS = dict(zip(ARMS, NEW_LAMBDAS, strict=True))


def existing_summary(task: str, seed: int, value: float) -> Path:
    if value == 0.0:
        return CANONICAL / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json"
    if value == 0.5:
        return MATCHED / task / "prompt_single" / f"episode_{seed:03d}_summary.json"
    raise ValueError(value)
