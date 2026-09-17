"""Locked protocol helpers for the L11 state-conditioned policy probe."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


PROTOCOL = "PROMPT_ATTN_L11_INITIAL_STATE_PROBE_V1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SHORT_TASKS = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
SEEDS = tuple(range(100))
FEATURE_NAMES = (
    "proprio",
    "visual_mean",
    "visual_std",
    "visual_l11_weighted",
    "prompt_hidden_mean",
    "action_context_hidden",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_outcomes(path: Path) -> dict[tuple[str, int], dict[str, bool]]:
    outcomes = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            seed = int(row["seed"])
            if task not in TASKS or seed not in SEEDS:
                continue
            outcomes[(task, seed)] = {
                arm: row[arm].strip().lower() == "true"
                for arm in ("vanilla", "shr", "l11_matched")
            }
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    missing = sorted(expected - outcomes.keys())
    if missing:
        raise RuntimeError(f"missing paired outcomes, first entries: {missing[:5]}")
    return outcomes


def feature_path(artifact: Path, task: str, seed: int) -> Path:
    return artifact / "features" / task / f"seed_{seed:03d}.npz"


def metadata_path(artifact: Path, task: str, seed: int) -> Path:
    return artifact / "features" / task / f"seed_{seed:03d}.json"

