"""Paired vanilla/VTCG rollout on robosuite long-horizon tasks (PickPlace / NutAssembly).

Why these tasks: LIBERO tops out at 2-3 goal atoms, so it cannot express
"several objects into one place". robosuite ships PickPlace (4 objects into 2
bins) and NutAssembly (4 nuts onto 4 pegs) with the SAME 7-dim OSC_POSE +
gripper action space LIBERO uses, so the pi0.5 server needs no new adapter.

Pairing discipline matches the LIBERO harness: identical simulator initial
state per pair (captured right after reset and restored for the second arm)
and identical action noise at every replan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

DEFAULT_ARMS = ("vanilla", "matched")


def axis_angle(quaternion) -> np.ndarray:
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


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            keys.add((int(row["seed"]), str(row["arm"])))
    return keys


def prepare_observation(obs, flip: bool):
    image = np.ascontiguousarray(obs["agentview_image"])
    if flip:
        image = np.ascontiguousarray(image[::-1])
    return image


def run_episode(env, client, image_tools, arm, seed, args):
    done = False
    success = False
    policy_steps = 0
    replan_index = 0
    queue = []
    started = time.monotonic()
    state_hash = None
    while policy_steps < args.max_steps and not done:
        if not queue:
            obs = env._get_observations(force_update=True)
            image = prepare_observation(obs, args.image_flip)
            image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, 224, 224)
            )
            wrist = np.zeros_like(image)
            if "robot0_eye_in_hand_image" in obs:
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
                if args.image_flip:
                    wrist = np.ascontiguousarray(wrist[::-1])
                wrist = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist, 224, 224)
                )
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    axis_angle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32)
            request_seed = args.noise_seed + seed * 10_000 + replan_index
            request = {
                "observation/image": image,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": args.prompt,
                "pi05_arm": arm,
                "pi05_noise_seed": request_seed,
            }
            if arm == "matched":
                request["pi05_guidance_scale"] = args.guidance_scale
                request["pi05_layer_index"] = args.layer_index
            response = client.infer(request)
            chunk = np.asarray(response["actions"], dtype=np.float32)
            queue.extend(chunk[: args.replan_steps])
            replan_index += 1
        action = np.asarray(queue.pop(0), dtype=np.float32)
        obs, reward, done, info = env.step(action)
        policy_steps += 1
        if env._check_success():
            success = True
            break
    return {
        "arm": arm,
        "success": bool(success),
        "policy_steps": policy_steps,
        "replans": replan_index,
        "runtime_seconds": time.monotonic() - started,
    }


def parse_seeds(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--env-name", default="PickPlace")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--seeds", default="0-49")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--prompt", default="place the objects into the bins")
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--noise-seed", type=int, default=20261022)
    parser.add_argument("--guidance-scale", type=float, default=0.5)
    parser.add_argument("--layer-index", type=int, default=10)
    parser.add_argument("--image-flip", type=int, default=1)
    parser.add_argument("--save-frame", type=Path, default=None)
    args = parser.parse_args()

    import robosuite as suite
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    result_path = artifact / "episodes.jsonl"
    seen = done_keys(result_path)
    seeds = parse_seeds(args.seeds)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        args.env_name,
        robots="Panda",
        controller_configs=controller_configs,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
        reward_shaping=False,
        hard_reset=False,
        ignore_done=True,
    )
    try:
        for seed in seeds:
            np.random.seed(seed)
            env.reset()
            init_state = env.sim.get_state().flatten()
            init_hash = hashlib.sha256(init_state.tobytes()).hexdigest()
            if args.save_frame is not None:
                import imageio.v2 as imageio

                raw = env._get_observations(force_update=True)["agentview_image"]
                args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(str(args.save_frame), np.ascontiguousarray(raw[::-1]))
            for arm in arms:
                if (seed, arm) in seen:
                    continue
                env.reset()
                env.sim.set_state_from_flattened(init_state)
                env.sim.forward()
                env._get_observations(force_update=True)
                outcome = run_episode(env, client, image_tools, arm, seed, args)
                record = {
                    "protocol": "PI05_ROBOSUITE_PAIRED_V1",
                    "env_name": args.env_name,
                    "seed": seed,
                    "init_state_sha256": init_hash,
                    "prompt": args.prompt,
                    "max_steps": args.max_steps,
                    "replan_steps": args.replan_steps,
                    "noise_seed": args.noise_seed,
                    **outcome,
                }
                append_jsonl(result_path, record)
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "arm": arm,
                            "success": record["success"],
                            "steps": record["policy_steps"],
                            "secs": round(record["runtime_seconds"], 1),
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
