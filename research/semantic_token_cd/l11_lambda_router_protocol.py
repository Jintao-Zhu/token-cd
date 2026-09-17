"""Locked protocol helpers for held-out L11 lambda routing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np


PROTOCOL = "PROMPT_ATTN_L11_LAMBDA_ROUTER_CONFIRM_V1"
ROUTER_PROTOCOL = "PROMPT_ATTN_L11_LAMBDA_ROUTER_CLOSED_LOOP_V1"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SHORT = {
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
SEEDS = tuple(range(100, 200))
LAMBDAS = np.asarray([0.0, 0.25, 0.5], dtype=np.float32)
ARM_NAMES = {
    0.0: "l11_positive_only",
    0.25: "l11_fixed_025",
    0.5: "l11_fixed_050",
}
FEATURE_NAMES = ("prompt_hidden_mean", "action_context_hidden")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_seeds(specification: str) -> list[int]:
    seeds = []
    for part in specification.split(","):
        part = part.strip()
        if "-" in part:
            low, high = map(int, part.split("-", 1))
            seeds.extend(range(low, high + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(seed not in SEEDS for seed in result):
        raise ValueError("confirmation seeds must be within 100..199")
    return result


def feature_path(artifact: Path, task: str, seed: int) -> Path:
    return artifact / "features" / task / f"seed_{seed:03d}.npz"


def feature_metadata_path(artifact: Path, task: str, seed: int) -> Path:
    return artifact / "features" / task / f"seed_{seed:03d}.json"


def arm_summary_path(artifact: Path, task: str, arm: str, seed: int) -> Path:
    return artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"


def arm_arrays_path(artifact: Path, task: str, arm: str, seed: int) -> Path:
    return artifact / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"


def load_frozen_model(discovery: Path) -> tuple[dict, Path]:
    results_path = discovery / "analysis" / "DISCOVERY_RESULTS.json"
    results = json.loads(results_path.read_text())
    if not results["go_no_go"]["passed"]:
        raise RuntimeError("discovery did not pass its locked Go criteria")
    info = results.get("frozen_model")
    if not info:
        raise RuntimeError("discovery did not freeze a model")
    model_path = Path(info["path"])
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if sha256(model_path) != info["sha256"]:
        raise RuntimeError("frozen model hash mismatch")
    payload = joblib.load(model_path)
    if payload["feature_set"] != "prompt_action":
        raise RuntimeError("unexpected frozen feature set")
    if not np.array_equal(np.asarray(payload["lambdas"]), LAMBDAS):
        raise RuntimeError("unexpected frozen lambda set")
    return payload, model_path


def load_feature_vector(path: Path, task: str) -> np.ndarray:
    task_index = (
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_pick_coke_can",
        "google_robot_move_near",
    ).index(task)
    with np.load(path) as arrays:
        parts = [np.eye(4, dtype=np.float32)[task_index]]
        parts.extend(
            np.asarray(arrays[name], dtype=np.float32).reshape(-1)
            for name in FEATURE_NAMES
        )
        return np.concatenate(parts)[None, :]


def predict_scores(model: dict, vector: np.ndarray) -> np.ndarray:
    transform = model["state_transform"]
    varying = np.asarray(transform["varying"], dtype=bool)
    state = transform["scaler"].transform(vector[:, varying])
    state = transform["pca"].transform(state)
    rows = []
    for lambda_index in range(len(LAMBDAS)):
        one_hot = np.eye(len(LAMBDAS), dtype=np.float32)[lambda_index]
        rows.append(np.concatenate([state[0], one_hot, np.kron(one_hot, state[0])]))
    return model["classifier"].predict_proba(np.stack(rows))[:, 1]


def choose_lambda(model: dict, feature_file: Path, task: str) -> tuple[float, list[float]]:
    scores = predict_scores(model, load_feature_vector(feature_file, task))
    margin = float(model.get("switch_margin", 0.0))
    weaker = int(np.argmax(scores[:2]))
    choice = weaker if scores[weaker] > scores[2] + margin else 2
    return float(LAMBDAS[choice]), [float(value) for value in scores]
