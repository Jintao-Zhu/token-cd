#!/usr/bin/env python3
"""Official OpenVLA LIBERO-Spatial vanilla rollout, paired with matched runner.

This is the no-intervention control for `libero_matched_rollout.py`.  It uses
exactly the same task registry, episode/seed convention, initial-state indexing,
image preprocessing, environment action postprocessing, and the locked official protocol.  The
only difference is that the action is the official clean OpenVLA action:
`predict_action(...)` with no KMeans, no L11 attention, and no token masking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.libero_runtime import (
    encode_video,
    load_policy,
    predict_action,
    prepare_agentview,
    prepare_env_action,
    set_determinism,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MAX_STEPS = 220
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
CHECKPOINT = Path(
    "/home/leju-suzhou/zjt_ws/checkpoints/libero/"
    "openvla-7b-finetuned-libero-spatial"
)
CODE_DIR = Path(
    "/home/leju-suzhou/zjt_ws/token-cd/"
    "third_party/openvla/prismatic/extern/hf"
)


def parse_episodes(value: str) -> list[int]:
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            episodes.extend(range(lo, hi + 1))
        else:
            episodes.append(int(part))
    return sorted(set(episodes))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--unnorm-key", default="libero_spatial")
    p.add_argument("--dataset-statistics", type=Path, default=None)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--settle-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--video-dir", type=Path, default=None)
    a = p.parse_args()

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    episodes = parse_episodes(a.episodes)
    suite = benchmark.get_benchmark_dict()[a.suite]()
    task = next(
        t for i in range(suite.n_tasks)
        if (t := suite.get_task(i)).name == a.task
    )
    task_index = next(
        i for i in range(suite.n_tasks)
        if suite.get_task(i).name == a.task
    )
    bddl = str(
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
    )
    init_states = suite.get_task_init_states(task_index)
    max_steps = int(
        a.max_steps
        if a.max_steps is not None
        else DEFAULT_MAX_STEPS.get(a.suite, MAX_STEPS)
    )

    set_determinism(7)
    model, processor = load_policy(
        a.checkpoint,
        CODE_DIR,
        device=f"cuda:{a.gpu}",
        dataset_statistics_path=a.dataset_statistics,
        unnorm_key=a.unnorm_key,
    )
    root = a.artifact.resolve() / task.name
    root.mkdir(parents=True, exist_ok=True)

    for ep in episodes:
        out_path = root / f"episode_{ep:03d}.json"
        if out_path.exists():
            print(json.dumps({"skip": True, "task": task.name, "episode": ep}), flush=True)
            continue

        init_index = ep % len(init_states)
        init = init_states[init_index]
        env.seed(int(a.env_seed))
        env.reset()
        obs = env.set_init_state(init)
        for _ in range(int(a.settle_steps)):
            obs, _reward, _done, _info = env.step([0, 0, 0, 0, 0, 0, -1])
        initial_state = np.asarray(env.get_sim_state()).copy()
        trajectory: list[np.ndarray] = []
        video_frames: list[np.ndarray] = []
        done = False
        started = time.monotonic()

        for _ in range(max_steps):
            _, image = prepare_agentview(obs)
            if a.video_dir is not None:
                video_frames.append(np.asarray(image.copy()))
            raw_action = predict_action(
                model,
                processor,
                image,
                task.language,
                unnorm_key=a.unnorm_key,
            )
            action = prepare_env_action(raw_action)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(
                    f"invalid vanilla action task={task.name} episode={ep}: {action}"
                )
            obs, _reward, done, _info = env.step(action.tolist())
            trajectory.append(action.copy())
            if done:
                break

        success = bool(env.check_success())
        if a.video_dir is not None and video_frames:
            encode_video(
                video_frames,
                a.video_dir.resolve() / task.name / f"episode_{ep:03d}.mp4",
                fps=30,
            )
        action_array = np.asarray(trajectory, dtype=np.float64)
        summary = {
            "protocol_id": f"{a.suite.upper()}_OFFICIAL_VANILLA_V1",
            "benchmark": a.suite,
            "task_id": int(task_index),
            "unnorm_key": a.unnorm_key,
            "dataset_statistics_path": str(a.dataset_statistics) if a.dataset_statistics else None,
            "task": task.name,
            "instruction": task.language,
            "episode": int(ep),
            "init_state_index": int(init_index),
            "env_seed": int(a.env_seed),
            "settle_steps": int(a.settle_steps),
            "max_policy_steps": int(max_steps),
            "success": success,
            "steps": len(trajectory),
            "runtime_seconds": time.monotonic() - started,
            "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
            "trajectory": action_array.tolist(),
            "control_steps": len(trajectory),
            "done": bool(done),
            "gpu_id": int(a.gpu),
        }
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps({
                "task": task.name,
                "episode": ep,
                "success": success,
                "steps": len(trajectory),
            }),
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    main()
