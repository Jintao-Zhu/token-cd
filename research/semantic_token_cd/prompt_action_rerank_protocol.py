"""Locked protocol for PA-Rerank-L11-Matched v1."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
ARTIFACT = REPO / "artifacts/prompt_action_rerank_l11_matched_v1"
CANONICAL = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
PROTOCOL = "PA_RERANK_L11_MATCHED_V1"
ARM = "pa_rerank_l11_matched"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SEEDS = tuple(range(100))
PROMPT_LAYERS = (11,)
ACTION_LAYERS = tuple(range(16, 32))
CANDIDATE_MULTIPLIER = 3
LAMBDA = 0.5

MATCHED_ROOT = {
    "google_robot_open_drawer": REPO / "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_open_drawer/prompt_single",
    "google_robot_close_drawer": REPO / "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes/google_robot_close_drawer/prompt_single",
    "google_robot_pick_coke_can": REPO / "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_pick_coke_can/prompt_single",
    "google_robot_move_near": REPO / "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_move_near/prompt_single",
}
HH_ROOT = REPO / "artifacts/prompt_action_complement_v1/closed_loop/episodes"
VANILLA_ROOT = CANONICAL / "episodes"
SHR_ROOT = CANONICAL / "episodes"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)
