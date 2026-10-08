"""Paired SIMPLER Google-Robot rollouts for pi0-FAST Vanilla and L6-Matched."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
TOKEN_CD = ROOT / "token-cd"
SIMPLER = ROOT / "SimplerEnv"
SPATIAL_SIMPLER = ROOT / "SpatialVLA-SimplerEnv"
PCD_SOURCE = ROOT / "pcd_openvla_simpler_box_31b027e/source"
DEFAULT_OUT = TOKEN_CD / "artifacts/pi0_fast_l6_matched_simpler_google800_v1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ARMS = ("vanilla", "l6_matched")


def _digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _environment(task: str, rt_depth: int = 0):
    import gymnasium as gym
    import simpler_env

    render_config = {
        "rt_samples_per_pixel": 32,
        "rt_max_path_depth": int(rt_depth),
        "rt_use_denoiser": True,
    }
    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd",
            prepackaged_config=True,
            distractor_config="less",
            shader_dir="rt",
            render_config=render_config,
            renderer_kwargs=renderer_kwargs,
        )
    env_id, base_kwargs = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base_kwargs)
    kwargs.update(
        obs_mode="rgbd",
        prepackaged_config=True,
        shader_dir="rt",
        render_config=render_config,
        renderer_kwargs=renderer_kwargs,
    )
    return gym.make(env_id, **kwargs)


def _eef_pos(agent_obs: dict, env) -> np.ndarray:
    if "eef_pos" in agent_obs:
        return np.asarray(agent_obs["eef_pos"], dtype=np.float32)
    # GoogleRobotStaticBase's public proprioception omits eef_pos. Rebuild the
    # same canonical Fractal/Simpler encoding used by OpenPI's SIMPLER adapter:
    # TCP pose in robot-base coordinates, wxyz quaternion, gripper openness.
    agent = env.unwrapped.agent
    pose = agent.robot.get_pose().inv() * env.unwrapped.tcp.get_pose()
    openness = 1.0 - float(agent.get_gripper_closedness())
    return np.concatenate((np.asarray(pose.p), np.asarray(pose.q), [openness])).astype(np.float32)


def _raw_policy_input(image: np.ndarray, instruction: str, agent_obs: dict, env) -> dict:
    eef = _eef_pos(agent_obs, env)
    quat_xyzw = eef[3:7][[1, 2, 3, 0]]
    state = np.concatenate((eef[:3], quat_xyzw, [1.0 - float(eef[7])])).astype(np.float32)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    import cv2

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return {
        "observation/primary_image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": state,
        "prompt": instruction,
    }


class RolloutPolicy:
    def __init__(self, host: str, port: int):
        os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        for path in (PCD_SOURCE, SPATIAL_SIMPLER, SIMPLER, ROOT / "SpatialVLA/ManiSkill2_real2sim", TOKEN_CD):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        # Match the official OpenPI serving path. Keep the environment's
        # synchronous server from timing out while JAX compiles the first request.
        from websockets.sync import client as websocket_client

        original_connect = websocket_client.connect

        def connect_without_keepalive(*args, **kwargs):
            kwargs.setdefault("ping_interval", None)
            kwargs.setdefault("ping_timeout", None)
            return original_connect(*args, **kwargs)

        websocket_client.connect = connect_without_keepalive
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.exec_horizon = 4

    def chunk(self, arm: str, raw: dict, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        request = {**raw, "cd_arm": arm, "cd_seed": int(seed)}
        result = self.client.infer(request)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if arm == "l6_matched":
            trace = dict(result["pi0_fast_matched_cd_trace"])
            # A malformed FAST payload uses OpenPI's documented zero-DCT
            # fallback. Accept it only when the direct path numerically agrees
            # with OpenPI's official decoder output.
            if not trace.get("openpi_official_decoder_agrees"):
                raise RuntimeError(f"L6 FAST decoder validation failed: {trace}")
            if trace.get("selector_layer_label") != "L6" or not trace.get("selected_count_equals_matched"):
                raise RuntimeError(f"L6-Matched invariant failed: {trace}")
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid {arm} action chunk: {actions.shape}")
        trace = dict(result.get("pi0_fast_matched_cd_trace", {
            "official_openpi_policy_infer": True,
            "output_action_shape": list(actions.shape),
            "output_action_finite": bool(np.isfinite(actions).all()),
        }))
        return actions, trace


class ActionChunk:
    def __init__(self):
        from collections import deque

        self.plan = deque()
        self.exec_horizon = 4
        self.previous_gripper_openness = None
        self.sticky_gripper_action = 0.0
        self.gripper_repeat_remaining = 0

    def next(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        raw = np.asarray(chunk, dtype=np.float32)
        action = np.zeros((7,), dtype=np.float32)
        action[:3] = raw[:3]
        action[3:6] = Rotation.from_euler("xyz", raw[3:6]).as_rotvec().astype(np.float32)
        openness = float(raw[6])
        if self.previous_gripper_openness is None:
            gripper_delta = 0.0
            self.previous_gripper_openness = openness
        else:
            gripper_delta = self.previous_gripper_openness - openness
            if abs(gripper_delta) > 0.5 and self.gripper_repeat_remaining == 0:
                self.sticky_gripper_action = gripper_delta
                self.gripper_repeat_remaining = 10
                self.previous_gripper_openness = openness
        if self.gripper_repeat_remaining > 0:
            gripper_delta = self.sticky_gripper_action
            self.gripper_repeat_remaining -= 1
        action[6] = np.clip(gripper_delta, -1.0, 1.0)
        return action


def _episode(policy: RolloutPolicy, env, task: str, seed: int, arm: str, out: Path, rt_depth: int = 0) -> dict[str, Any]:
    started = time.monotonic()
    obs, reset_info = env.reset(seed=seed)
    instruction = str(env.unwrapped.get_language_instruction())
    camera = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"
    initial_image = np.asarray(obs["image"][camera]["rgb"])
    initial_agent = _eef_pos(obs["agent"], env)
    rgb_sha = _digest(initial_image)
    state_sha = _digest(initial_agent)
    chunks: list[dict[str, Any]] = []
    action_chunk = ActionChunk()
    success = False
    terminated = truncated = False
    last_info: dict[str, Any] = {}
    steps = 0
    max_steps = 250
    while not (terminated or truncated) and steps < max_steps:
        image = obs["image"][camera]["rgb"]
        raw = _raw_policy_input(image, instruction, obs["agent"], env)
        if steps % action_chunk.exec_horizon == 0:
            replan_seed = seed * 1009 + steps // action_chunk.exec_horizon
            chunk, trace = policy.chunk(arm, raw, replan_seed)
            if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                raise RuntimeError(f"invalid action chunk {chunk.shape} for {task}/{seed}/{arm}")
            action_chunk.plan.extend(chunk[: action_chunk.exec_horizon])
            chunks.append({"step": steps, "trace": trace})
        raw_action = action_chunk.plan.popleft()
        action = action_chunk.next(raw_action)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)
        success = bool(info.get("success", success))
        steps += 1
    if steps >= max_steps and not (terminated or truncated):
        truncated = True
    result = {
        "status": "complete",
        "task": task,
        "seed": int(seed),
        "arm": arm,
        "success": bool(success),
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "renderer": {"shader_dir": "rt", "rt_max_path_depth": int(rt_depth), "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
        "checkpoint": str(ROOT / "checkpoints/openpi_fast_finetuned/fractal"),
        "instruction": instruction,
        "environment_id": getattr(getattr(env, "spec", None), "id", None),
        "initial_rgb_sha256": rgb_sha,
        "initial_agent_state_sha256": state_sha,
        "reset_info": reset_info,
        "final_info": last_info,
        "replan_count": len(chunks),
        "chunks": chunks,
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    path = out / "episodes" / task / f"{arm}_seed{seed:03d}.json"
    _atomic_json(path, result)
    print(json.dumps({k: result[k] for k in ("task", "seed", "arm", "success", "steps", "runtime_seconds", "initial_rgb_sha256")}, default=str), flush=True)
    return result


def build_manifest(out: Path, workers: int = 9, seed_start: int = 0, seed_end: int = 99) -> list[list[dict[str, Any]]]:
    """Shard seed-paired jobs in contiguous task blocks across all workers."""
    if workers != 9:
        raise ValueError("this run is configured for exactly 9 rollout workers (3 clients per GPU)")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    worker = 0
    for task in TASKS:
        seeds = list(range(seed_start, seed_end + 1))
        cursor = 0
        for slot in range(workers):
            remaining_slots = workers - slot
            count = (len(seeds) - cursor + remaining_slots - 1) // remaining_slots
            for seed in seeds[cursor : cursor + count]:
                shards[worker].append({"task": task, "seed": seed, "arms": list(ARMS)})
            cursor += count
            worker = (worker + 1) % workers
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rt-depth", type=int, default=int(os.environ.get("PI0FAST_RT_DEPTH", "0")))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/openpi_fast_finetuned/fractal")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        shards = build_manifest(args.out)
        for i, jobs in enumerate(shards):
            _atomic_json(args.out / "manifests" / f"worker_{i}.json", {"worker_id": i, "jobs": jobs})
        protocol = {
            "experiment": "pi0fast_simpler_google_vanilla_vs_l6_matched",
            "total_episodes": 800,
            "tasks": list(TASKS),
            "arms": list(ARMS),
            "seeds": [0, 99],
            "workers": 9,
            "gpu_worker_map": {str(w): [1, 1, 1, 2, 2, 2, 3, 3, 3][w] for w in range(9)},
            "renderer": {"shader_dir": "rt", "rt_max_path_depth": 0, "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
            "checkpoint": str(args.checkpoint),
            "selector": "L6 prompt-to-primary-view attention, zero-based layer index 6; dynamic entity-matched budget; harmonic negative; lambda=0.5",
            "pairing": "Vanilla and L6-Matched reset the same task seed; both initial RGB and agent-state hashes logged",
        }
        _atomic_json(args.out / "protocol.json", protocol)
        print(json.dumps({"output": str(args.out), "episodes": 800, "shards": [len(s) * 2 for s in shards]}))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    shard_payload = json.loads(args.shard.read_text())
    jobs = shard_payload["jobs"]
    policy = RolloutPolicy(args.host, args.port)
    print(json.dumps({"event": "model_ready", "gpu": args.gpu, "worker": args.worker_id, "jobs": len(jobs)}), flush=True)
    current_task = None
    env = None
    try:
        for job in jobs:
            task, seed = job["task"], int(job["seed"])
            if task != current_task:
                if env is not None:
                    env.close()
                env = _environment(task, args.rt_depth)
                current_task = task
                print(json.dumps({"event": "environment_ready", "task": task, "gpu": args.gpu, "worker": args.worker_id}), flush=True)
            for arm in job["arms"]:
                output = args.out / "episodes" / task / f"{arm}_seed{seed:03d}.json"
                if output.exists():
                    try:
                        if json.loads(output.read_text()).get("status") == "complete":
                            continue
                    except Exception:
                        pass
                try:
                    _episode(policy, env, task, seed, arm, args.out, args.rt_depth)
                except BaseException as exc:
                    failure = {
                        "status": "error", "task": task, "seed": seed, "arm": arm,
                        "gpu": args.gpu, "worker_id": args.worker_id,
                        "error_type": type(exc).__name__, "error": repr(exc),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    _atomic_json(output, failure)
                    raise
    finally:
        if env is not None:
            env.close()
    print(json.dumps({"event": "worker_complete", "gpu": args.gpu, "worker": args.worker_id}), flush=True)


if __name__ == "__main__":
    main()
