"""Locked protocol for generic-frame target-specific Prompt attention."""
from __future__ import annotations

import json
import os
from pathlib import Path

from research.semantic_token_cd.xswap_protocol import CANONICAL, PCD_SOURCE, TASKS, atomic_json

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ARTIFACT = REPO_ROOT / "artifacts/target_specific_attention_shr_v1"
SOURCE_STATES = REPO_ROOT / "artifacts/prompt_attn_instr_swap_v1/runs/emitted_states"
PROTOCOL = "TARGET_SPECIFIC_ATTENTION_SHR_V1"
ARMS = ("correct", "target_diff", "target_boost", "reverse_diff")
NEW_ARMS = ARMS[1:]
OFFLINE_ARMS = ARMS + ("target_diff_alt",)
ATTENTION_LAYERS = (11,)
LAMBDA = 0.5


def generic_instructions(task: str) -> tuple[str, str]:
    """Return fixed primary and paraphrased grammatical generic frames."""
    if task == "google_robot_open_drawer":
        return "open a drawer", "open the drawer"
    if task == "google_robot_close_drawer":
        return "close a drawer", "close the drawer"
    if task == "google_robot_pick_coke_can":
        return "pick something up", "pick up an item"
    if task == "google_robot_move_near":
        return "move something near something else", "move one item near another"
    raise ValueError(task)


ARM_FORMULA = {
    "target_diff": ("target_diff", 1.0),
    "target_boost": ("target_boost", 1.0),
    "reverse_diff": ("reverse_diff", 1.0),
    "target_diff_alt": ("target_diff", 1.0),
}


def selector_config(task: str, arm: str) -> dict:
    if arm == "correct":
        return {"formula": None, "eta": None, "contrast_instruction": None}
    formula, eta = ARM_FORMULA[arm]
    primary, alternate = generic_instructions(task)
    contrast = alternate if arm == "target_diff_alt" else primary
    return {"formula": formula, "eta": eta, "contrast_instruction": contrast}


def write_lock() -> None:
    lock = {
        "protocol": PROTOCOL,
        "tasks": list(TASKS), "seeds": "0-99", "arms": list(ARMS),
        "new_closed_loop_episodes": 1200, "reused_correct_episodes": 400,
        "attention_layers": list(ATTENTION_LAYERS), "lambda": LAMBDA,
        "generic_instructions": {task: list(generic_instructions(task)) for task in TASKS},
        "formulae": {
            "target_diff": "P-Q", "target_boost": "P+(P-Q)=2P-Q",
            "reverse_diff": "Q-P",
        },
        "attention_normalization": "P and Q independently normalized over 256 visual tokens",
        "coverage": "own-state Standard-SHR matched count; generic prompt never controls m",
        "decode_instruction": "real task instruction for clean and harmonic negative branches",
        "reconstruction": "16x16 four-neighbor Dirichlet harmonic beta=0 gamma=1",
        "guidance": "lambda=0.5 dimensions 0..5; gripper clean",
        "offline_gate": "5 fixed episodes x 3 fixed trajectory states x 4 tasks = 60 states",
    }
    atomic_json(ARTIFACT / "CONFIG_LOCK.json", lock)

