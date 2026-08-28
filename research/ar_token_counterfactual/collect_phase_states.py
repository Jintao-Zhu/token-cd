"""Collect outcome-blind phase-proxy states and independent calibration images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .libero_runtime import load_policy, predict_action, prepare_agentview, prepare_env_action, set_determinism


PHASES = ("approach", "pre_grasp", "grasp_contact", "transport_place")
PHASE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
PHASE_COUNTS = {"approach": 13, "pre_grasp": 13, "grasp_contact": 12, "transport_place": 12}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_png(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def make_env(task, get_libero_path, env_class):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = env_class(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    selected = json.loads((artifact / "selected_tasks.json").read_text())["task_ids"]
    if len(selected) != 3:
        raise RuntimeError("Exactly three baseline-qualified tasks are required")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    calibration_tasks = [task_id for task_id in range(10) if task_id not in selected][:5]
    calibration_manifest = artifact / "calibration_manifest.jsonl"
    existing_calibration = set()
    if calibration_manifest.exists():
        existing_calibration = {json.loads(line)["calibration_id"] for line in calibration_manifest.read_text().splitlines() if line}
    for task_id in calibration_tasks:
        task = suite.get_task(task_id)
        for init_index in range(20, 50):
            calibration_id = f"task{task_id:02d}_init{init_index:02d}"
            if calibration_id in existing_calibration:
                continue
            env = make_env(task, get_libero_path, OffScreenRenderEnv)
            try:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[init_index])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                _, image = prepare_agentview(obs)
            finally:
                env.close()
            image_path = artifact / "calibration_images" / f"{calibration_id}.png"
            save_png(image, image_path)
            append_jsonl(calibration_manifest, {
                "calibration_id": calibration_id,
                "task_id": task_id,
                "init_state_index": init_index,
                "task_description": task.language,
                "image": str(image_path.relative_to(artifact)),
                "split": "position_mean_calibration",
            })

    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)
    snapshot_manifest = artifact / "phase1_state_manifest.jsonl"
    existing = set()
    if snapshot_manifest.exists():
        existing = {json.loads(line)["snapshot_id"] for line in snapshot_manifest.read_text().splitlines() if line}
    for task_id in selected:
        task = suite.get_task(task_id)
        for init_index in range(13):
            required = [phase for phase in PHASES if init_index < PHASE_COUNTS[phase]]
            ids = {phase: f"task{task_id:02d}_init{init_index:02d}_{phase}" for phase in required}
            if all(snapshot_id in existing for snapshot_id in ids.values()):
                continue
            env = make_env(task, get_libero_path, OffScreenRenderEnv)
            frames = []
            try:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[init_index])
                for _ in range(10):
                    obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                steps = 0
                while steps < 220 and not done:
                    _, image = prepare_agentview(obs)
                    frames.append(image.copy())
                    action = prepare_env_action(predict_action(model, processor, image, task.language))
                    obs, _, done, _ = env.step(action.tolist())
                    steps += 1
            finally:
                env.close()
            if not frames:
                raise RuntimeError(f"No frames collected for task {task_id}, init {init_index}")
            for phase, fraction in zip(PHASES, PHASE_FRACTIONS):
                if phase not in required or ids[phase] in existing:
                    continue
                frame_index = int(round(fraction * (len(frames) - 1)))
                image_path = artifact / "phase1_images" / f"{ids[phase]}.png"
                save_png(frames[frame_index], image_path)
                append_jsonl(snapshot_manifest, {
                    "snapshot_id": ids[phase],
                    "task_id": task_id,
                    "init_state_index": init_index,
                    "task_description": task.language,
                    "phase": phase,
                    "phase_source": "trajectory_quartile_proxy",
                    "phase_fraction": fraction,
                    "trajectory_frame_index": frame_index,
                    "trajectory_policy_steps": len(frames),
                    "image": str(image_path.relative_to(artifact)),
                    "split": "phase1",
                })
                existing.add(ids[phase])
            print(json.dumps({"task_id": task_id, "init_state_index": init_index, "trajectory_complete": True}), flush=True)


if __name__ == "__main__":
    main()
