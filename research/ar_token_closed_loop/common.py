from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@dataclass(frozen=True)
class Progress:
    eef_object_distance: float
    object_goal_distance: float
    grasped: bool
    predicate: bool


def progress(env) -> Progress:
    inner = env.env
    goals = inner.parsed_problem["goal_state"]
    if len(goals) != 1 or len(goals[0]) != 3 or goals[0][0] not in ("in", "on"):
        raise RuntimeError(f"Unsupported goal structure: {goals}")
    goal = goals[0]
    object_name, goal_name = goal[1], goal[2]
    object_position = inner.object_states_dict[object_name].get_geom_state()["pos"]
    goal_position = inner.object_states_dict[goal_name].get_geom_state()["pos"]
    eef_position = inner.sim.data.site_xpos[inner.robots[0].eef_site_id]
    grasped = inner._check_grasp(inner.robots[0].gripper, inner.get_object(object_name))
    return Progress(
        eef_object_distance=float(np.linalg.norm(eef_position - object_position)),
        object_goal_distance=float(np.linalg.norm(object_position - goal_position)),
        grasped=bool(grasped),
        predicate=bool(inner._eval_predicate(goal)),
    )


def progress_dict(env) -> dict:
    return asdict(progress(env))


def make_env(task, get_libero_path, env_class):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = env_class(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    return env
