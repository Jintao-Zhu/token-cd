"""Paired vanilla/matched pi0.5 evaluation on LIBERO-Long (``libero_10``)."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


MAX_STEPS = 520
DEFAULT_ARMS = ("vanilla", "matched")


def parse_ids(value: str, *, upper_bound: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(upper_bound))
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lower, upper = map(int, part.split("-", 1))
            values.extend(range(lower, upper + 1))
        else:
            values.append(int(part))
    result = sorted(set(values))
    if not result or result[0] < 0 or result[-1] >= upper_bound:
        raise ValueError(f"IDs must be nonempty and in [0, {upper_bound - 1}]")
    return result


def axis_angle(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] * value[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros((3,), dtype=np.float64)
    return value[:3] * 2.0 * math.acos(value[3]) / denominator


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            keys.add((int(row["task_id"]), int(row["init_state_id"]), str(row["arm"])))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--init-states", default="0")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=20261020)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--layer-index", type=int, default=None)
    parser.add_argument("--active-steps", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument("--record-stride", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    result_path = artifact / "episodes.jsonl"
    done_keys = completed_keys(result_path)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_ids = parse_ids(args.tasks, upper_bound=suite.n_tasks)
    init_ids = parse_ids(args.init_states, upper_bound=50)
    arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
    if not arms or any(arm not in DEFAULT_ARMS for arm in arms):
        raise ValueError(f"arms must be drawn from {DEFAULT_ARMS}")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        for init_state_id in init_ids:
            if init_state_id >= len(initial_states):
                raise ValueError(f"task {task_id} has only {len(initial_states)} initial states")
            for arm in arms:
                key = (task_id, init_state_id, arm)
                if key in done_keys:
                    continue
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=256,
                    camera_widths=256,
                )
                env.seed(args.env_seed)
                actions: list[np.ndarray] = []
                selected_counts: list[int] = []
                error = None
                success = False
                initial_state_hash = None
                initial_rgb_hash = None
                started = time.monotonic()
                try:
                    env.reset()
                    observation = env.set_init_state(initial_states[init_state_id])
                    for _ in range(args.settle_steps):
                        observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    initial_state = np.asarray(env.get_sim_state()).copy()
                    initial_state_hash = hashlib.sha256(initial_state.tobytes()).hexdigest()
                    initial_rgb = np.ascontiguousarray(observation["agentview_image"])
                    initial_rgb_hash = hashlib.sha256(initial_rgb.tobytes()).hexdigest()
                    action_queue: deque[np.ndarray] = deque()
                    replan_index = 0
                    policy_steps = 0
                    frames: list[np.ndarray] = []
                    while policy_steps < args.max_steps and not success:
                        if not action_queue:
                            base_image = np.ascontiguousarray(
                                observation["agentview_image"][::-1, ::-1]
                            )
                            wrist_image = np.ascontiguousarray(
                                observation["robot0_eye_in_hand_image"][::-1, ::-1]
                            )
                            base_image = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(base_image, 224, 224)
                            )
                            wrist_image = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_image, 224, 224)
                            )
                            state = np.concatenate(
                                (
                                    observation["robot0_eef_pos"],
                                    axis_angle(observation["robot0_eef_quat"]),
                                    observation["robot0_gripper_qpos"],
                                )
                            ).astype(np.float32)
                            request_seed = (
                                args.noise_seed
                                + task_id * 1_000_000
                                + init_state_id * 10_000
                                + replan_index
                            )
                            request = {
                                "observation/image": base_image,
                                "observation/wrist_image": wrist_image,
                                "observation/state": state,
                                "prompt": str(task.language),
                                "pi05_arm": arm,
                                "pi05_noise_seed": request_seed,
                            }
                            if arm == "matched":
                                if args.guidance_scale is not None:
                                    request["pi05_guidance_scale"] = args.guidance_scale
                                if args.layer_index is not None:
                                    request["pi05_layer_index"] = args.layer_index
                                if args.active_steps is not None:
                                    request["pi05_active_steps"] = args.active_steps
                            if args.num_steps is not None:
                                request["pi05_num_steps"] = args.num_steps
                            response = client.infer(request)
                            chunk = np.asarray(response["actions"], dtype=np.float32)
                            if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                                raise RuntimeError(f"invalid action chunk {chunk.shape}")
                            action_queue.extend(chunk[: args.replan_steps])
                            paired_trace = response.get("pi05_paired_trace", {})
                            if "selected_token_count" in paired_trace:
                                selected_counts.append(int(paired_trace["selected_token_count"]))
                            replan_index += 1
                        action = np.asarray(action_queue.popleft(), dtype=np.float32)
                        observation, _, success, _ = env.step(action.tolist())
                        actions.append(action.copy())
                        if args.record_dir is not None and policy_steps % max(
                            1, args.record_stride
                        ) == 0:
                            frames.append(
                                np.ascontiguousarray(observation["agentview_image"][::-1])
                            )
                        policy_steps += 1
                    success = bool(success or env.check_success())
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    env.close()

                action_array = np.asarray(actions, dtype=np.float32)
                video_path = None
                if args.record_dir is not None and frames:
                    import imageio.v2 as imageio

                    args.record_dir.mkdir(parents=True, exist_ok=True)
                    label = "success" if success else "failure"
                    video_path = (
                        args.record_dir
                        / f"task{task_id:02d}_init{init_state_id:02d}_{arm}_{label}.mp4"
                    )
                    imageio.mimwrite(
                        video_path,
                        list(frames),
                        fps=20,
                        macro_block_size=None,
                    )
                record = {
                    "protocol": "PI05_LIBERO_LONG_PAIRED_V1",
                    "suite": "libero_10",
                    "task_id": task_id,
                    "task_name": task.name,
                    "instruction": task.language,
                    "init_state_id": init_state_id,
                    "arm": arm,
                    "env_seed": args.env_seed,
                    "base_noise_seed": args.noise_seed,
                    "settle_steps": args.settle_steps,
                    "replan_steps": args.replan_steps,
                    "max_steps": args.max_steps,
                    "guidance_scale_requested": args.guidance_scale,
                    "layer_index_requested": args.layer_index,
                    "active_steps_requested": args.active_steps,
                    "num_steps_requested": args.num_steps,
                    "video_path": str(video_path) if video_path else None,
                    "success": bool(success),
                    "policy_steps": len(actions),
                    "replans": int(math.ceil(len(actions) / args.replan_steps)) if actions else 0,
                    "runtime_seconds": time.monotonic() - started,
                    "initial_state_sha256": initial_state_hash,
                    "initial_rgb_sha256": initial_rgb_hash,
                    "action_sha256": hashlib.sha256(action_array.tobytes()).hexdigest(),
                    "actions_finite": bool(not action_array.size or np.isfinite(action_array).all()),
                    "mean_selected_token_count": (
                        float(np.mean(selected_counts)) if selected_counts else None
                    ),
                    "error": error,
                }
                append_jsonl(result_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
