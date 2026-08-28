"""Direct LIBERO runtime exposing aligned RGB, instance segmentation, and progress metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from libero.libero import benchmark
from libero.libero.envs import SegmentationRenderEnv

from lerobot.envs.libero import LiberoEnv
from research.coreact_closed_loop.runtime import env_config


@dataclass(frozen=True)
class ProgressSnapshot:
    eef_object_distance: float
    object_goal_distance: float
    grasped: bool
    predicate: bool


def make_segmented_env(suite: str, task_id: int) -> LiberoEnv:
    cfg = env_config(suite, task_id)
    task_suite = benchmark.get_benchmark_dict()[suite]()
    env = LiberoEnv(
        task_suite=task_suite,
        task_id=task_id,
        task_suite_name=suite,
        episode_length=30,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        init_states=True,
        control_mode="relative",
    )
    env._env = SegmentationRenderEnv(
        bddl_file_name=env._task_bddl_file,
        camera_heights=256,
        camera_widths=256,
        camera_segmentations="instance",
    )
    return env


def raw_observation(env: LiberoEnv) -> dict:
    if env._env is None:
        raise RuntimeError("segmented environment is not initialized")
    return env._env.env._get_observations()


def batched_observation(observation: dict) -> dict:
    """Add one batch axis to every ndarray from the direct, non-vectorized environment."""
    output = {}
    for key, value in observation.items():
        if isinstance(value, dict):
            output[key] = batched_observation(value)
        elif isinstance(value, np.ndarray):
            output[key] = np.expand_dims(value, axis=0)
        else:
            output[key] = value
    return output


def task_entities(env: LiberoEnv) -> tuple[str, str, list]:
    inner = env._env.env
    goals = inner.parsed_problem["goal_state"]
    if len(goals) != 1 or len(goals[0]) != 3 or goals[0][0] not in ("in", "on"):
        raise ValueError(f"diagnostic supports one binary in/on goal, got {goals}")
    return goals[0][1], goals[0][2], goals[0]


def progress_snapshot(env: LiberoEnv) -> ProgressSnapshot:
    if env._env is None:
        raise RuntimeError("segmented environment is not initialized")
    inner = env._env.env
    object_name, goal_name, goal = task_entities(env)
    object_position = inner.object_states_dict[object_name].get_geom_state()["pos"]
    goal_position = inner.object_states_dict[goal_name].get_geom_state()["pos"]
    eef_position = inner.sim.data.site_xpos[inner.robots[0].eef_site_id]
    grasped = inner._check_grasp(inner.robots[0].gripper, inner.get_object(object_name))
    return ProgressSnapshot(
        eef_object_distance=float(np.linalg.norm(eef_position - object_position)),
        object_goal_distance=float(np.linalg.norm(object_position - goal_position)),
        grasped=bool(grasped),
        predicate=bool(inner._eval_predicate(goal)),
    )


def step_without_autoreset(env: LiberoEnv, action: np.ndarray) -> tuple[dict, ProgressSnapshot]:
    if env._env is None:
        raise RuntimeError("segmented environment is not initialized")
    raw, _, _, _ = env._env.step(action)
    return env._format_raw_obs(raw), progress_snapshot(env)
