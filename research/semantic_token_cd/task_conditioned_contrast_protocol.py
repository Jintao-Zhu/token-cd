"""Frozen protocol constants for task-conditioned contrast + APC v1."""
from __future__ import annotations

from pathlib import Path


REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
ARTIFACT = REPO / "artifacts/task_conditioned_contrast_midsize_v1"
CANONICAL = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
OFFLINE_STATE_ROOT = REPO / "artifacts/l11_entity_budget_calibration_v1/states"

PROTOCOL = "TASK_CONDITIONED_CONTRAST_MIDSIZE_V1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
TASK_INDEX = {task: index for index, task in enumerate(TASKS)}
ARMS = ("C", "C_APC", "T", "T_APC")
FORMAL_SEEDS = tuple(range(100, 160))
OFFLINE_SEEDS_PER_TASK = 75
LAMBDA0 = 0.5
APC_BETA = 0.1
KMEANS_K = 8
KMEANS_SEED = 0
ATTENTION_LAYERS = (11,)

CONTROL_INSTRUCTIONS = {
    "google_robot_open_drawer": "open a drawer",
    "google_robot_close_drawer": "close a drawer",
    "google_robot_pick_coke_can": "pick up an object",
    "google_robot_move_near": "move an object near another object",
}

CONTROL_INSTRUCTIONS_ALT = {
    "google_robot_open_drawer": "pull open a drawer",
    "google_robot_close_drawer": "push closed a drawer",
    "google_robot_pick_coke_can": "grasp an object",
    "google_robot_move_near": "bring an object close to another object",
}

FORMAL_TASKS = list(TASKS)
FORMAL_ARMS = list(ARMS)
