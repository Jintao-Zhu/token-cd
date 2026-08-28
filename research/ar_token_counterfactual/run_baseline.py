"""Run the locked official OpenVLA LIBERO-Spatial baseline qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .libero_runtime import encode_video, load_policy, predict_action, prepare_agentview, prepare_env_action, set_determinism


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["episode_id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    code_dir = workspace / "third_party/openvla/prismatic/extern/hf"

    set_determinism(7)
    model, processor = load_policy(checkpoint, code_dir)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    results_path = artifact / ("smoke_results.jsonl" if args.smoke else "baseline_results.jsonl")
    done_ids = completed_ids(results_path)
    manifest = [json.loads(line) for line in (artifact / "baseline_manifest.jsonl").read_text().splitlines() if line]
    if args.smoke:
        manifest = manifest[:1]

    for entry in manifest:
        episode_id = entry["episode_id"]
        if episode_id in done_ids:
            continue
        task_id = entry["task_id"]
        init_state_index = entry["init_state_index"]
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        frames: list[np.ndarray] = []
        actions: list[list[float]] = []
        success = False
        error = None
        started = time.perf_counter()
        try:
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(task_id)[init_state_index])
            for _ in range(10):
                obs, _, success, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            policy_steps = 0
            while policy_steps < 220 and not success:
                frame, policy_image = prepare_agentview(obs)
                frames.append(frame)
                raw_action = predict_action(model, processor, policy_image, task.language)
                env_action = prepare_env_action(raw_action)
                actions.append(env_action.tolist())
                obs, _, success, _ = env.step(env_action.tolist())
                policy_steps += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            policy_steps = len(actions)
        finally:
            env.close()

        video_path = artifact / "videos" / f"{episode_id}_success{int(bool(success))}.mp4"
        video_error = None
        if frames:
            try:
                encode_video(frames, video_path)
            except Exception as exc:
                video_error = f"{type(exc).__name__}: {exc}"
        action_array = np.asarray(actions, dtype=np.float64)
        row = {
            **entry,
            "task_description": task.language,
            "success": bool(success),
            "policy_steps": policy_steps,
            "runtime_seconds": time.perf_counter() - started,
            "error": error,
            "video_error": video_error,
            "nonfinite_action": bool(action_array.size and not np.isfinite(action_array).all()),
            "action_sha256": hashlib.sha256(action_array.tobytes()).hexdigest(),
            "video": str(video_path.relative_to(artifact)) if frames else None,
        }
        append_jsonl(results_path, row)
        print(json.dumps({"episode_id": episode_id, "completed": True, "error": error}), flush=True)


if __name__ == "__main__":
    main()
