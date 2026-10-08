"""One-episode SIMPLER vanilla smoke test for pi0-FAST via the OpenPI websocket API."""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path


ROOT = Path("/home/leju-suzhou/zjt_ws")
TOKEN_CD = ROOT / "token-cd"
SIMPLER = ROOT / "SimplerEnv"
SPATIAL_SIMPLER = ROOT / "SpatialVLA-SimplerEnv"
PCD_SOURCE = ROOT / "pcd_openvla_simpler_box_31b027e/source"
OUT = TOKEN_CD / "artifacts/pi0_fast_simpler_vanilla_smoke_v1"


class Pi0FastClient:
    def __init__(self, host: str, port: int, exec_horizon: int = 4):
        # The synchronous OpenPI server performs JAX inference on its event-loop
        # thread. A long first compile can exceed websockets' default keepalive
        # deadline on a loaded GPU, so disable heartbeat expiry for this smoke.
        from websockets.sync import client as websocket_client

        original_connect = websocket_client.connect

        def connect_without_keepalive(*args, **kwargs):
            kwargs.setdefault("ping_interval", None)
            kwargs.setdefault("ping_timeout", None)
            return original_connect(*args, **kwargs)

        websocket_client.connect = connect_without_keepalive
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.exec_horizon = exec_horizon
        self.plan: collections.deque = collections.deque()
        self.task = None
        self.last_stats: dict = {}
        self.previous_gripper_openness = None
        self.sticky_gripper_action = 0.0
        self.gripper_repeat_remaining = 0

    def reset(self, task: str) -> None:
        self.plan.clear()
        self.task = task
        self.previous_gripper_openness = None
        self.sticky_gripper_action = 0.0
        self.gripper_repeat_remaining = 0

    def step(self, image, instruction: str, agent_obs: dict):
        import cv2
        import numpy as np
        from scipy.spatial.transform import Rotation

        if instruction != self.task:
            self.reset(instruction)
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[0] == 3 and image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        if not self.plan:
            eef = np.asarray(agent_obs["eef_pos"], dtype=np.float32)
            qpos = np.asarray(agent_obs["qpos"], dtype=np.float32)
            # Match the existing SIMPLER Fractal/Google Robot Pi0 adapter:
            # xyz + xyzw quaternion + gripper closedness.
            quat_xyzw = eef[3:7][[1, 2, 3, 0]]
            state = np.concatenate((eef[:3], quat_xyzw, [1.0 - float(eef[7])])).astype(np.float32)
            output = self.client.infer(
                {
                    "observation/primary_image": image,
                    "observation/wrist_image": np.zeros_like(image),
                    "observation/state": state,
                    "prompt": instruction,
                }
            )
            chunk = np.asarray(output["actions"], dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] < 7 or not np.isfinite(chunk).all():
                raise RuntimeError(f"Invalid FAST-decoded action chunk: {chunk.shape}")
            self.last_stats = {
                "chunk_shape": list(chunk.shape),
                "chunk_min": float(np.min(chunk)),
                "chunk_max": float(np.max(chunk)),
                "chunk_mean_abs": float(np.mean(np.abs(chunk))),
            }
            self.plan.extend(chunk[: self.exec_horizon])

        raw = np.asarray(self.plan.popleft(), dtype=np.float32)
        # Policy output is unnormalized with the Fractal/Google Robot statistics.
        # Fractal's rotation representation is Euler delta; SIMPLER's controller
        # expects an axis-angle vector. Match the existing SIMPLER Pi0 adapter.
        action = np.zeros((7,), dtype=np.float32)
        action[:3] = raw[:3]
        action[3:6] = Rotation.from_euler("xyz", raw[3:6]).as_rotvec().astype(np.float32)
        openness = float(raw[6])  # Fractal convention: 1=open, 0=closed.
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
        return raw[:7].copy(), action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--inference-gpu", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/openpi-assets/openpi-assets/checkpoints/pi0_fast_base",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8117)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for path in (PCD_SOURCE, SPATIAL_SIMPLER, SIMPLER, ROOT / "SpatialVLA/ManiSkill2_real2sim", TOKEN_CD):
        if str(path) not in os.sys.path:
            os.sys.path.insert(0, str(path))

    import gymnasium as gym
    import numpy as np
    import simpler_env

    render_config = {
        "rt_samples_per_pixel": 32,
        "rt_max_path_depth": 0,
        "rt_use_denoiser": True,
    }
    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    if args.task == "google_robot_pick_coke_can":
        env = gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd",
            prepackaged_config=True,
            distractor_config="less",
            shader_dir="rt",
            render_config=render_config,
            renderer_kwargs=renderer_kwargs,
        )
        env_id = "GraspSingleOpenedCokeCanDistractorInScene-v0"
    else:
        env_id, base_kwargs = simpler_env.ENVIRONMENT_MAP[args.task]
        kwargs = dict(base_kwargs)
        kwargs.update(
            obs_mode="rgbd",
            prepackaged_config=True,
            shader_dir="rt",
            render_config=render_config,
            renderer_kwargs=renderer_kwargs,
        )
        env = gym.make(env_id, **kwargs)

    policy = Pi0FastClient(args.host, args.port)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    obs, reset_info = env.reset(seed=args.seed)
    instruction = env.unwrapped.get_language_instruction()
    trace = []
    last_info = {}
    success = False
    terminated = truncated = False
    step = 0
    try:
        while not (terminated or truncated) and step < args.max_steps:
            camera_name = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"
            image = obs["image"][camera_name]["rgb"]
            raw, action = policy.step(image, instruction, obs["agent"])
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            success = bool(info.get("success", success))
            trace.append(
                {
                    "step": step,
                    "raw_fast_action": raw.tolist(),
                    "simper_action": action.tolist(),
                    "chunk_stats": dict(policy.last_stats),
                    "success": success,
                }
            )
            step += 1
    finally:
        env.close()

    summary = {
        "model": "Physical Intelligence OpenPI pi0-FAST",
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "seed": args.seed,
        "renderer": {"shader_dir": "rt", **render_config},
        "renderer_gpu": args.gpu,
        "inference_gpu": args.gpu if args.inference_gpu is None else args.inference_gpu,
        "action_adapter": "Fractal/Google Robot state/action stats; xyz action direct; Euler rotation converted to axis-angle; sticky gripper delta; zero wrist-image pad; technical smoke only",
        "instruction": instruction,
        "environment_id": env_id,
        "steps": step,
        "smoke_step_limit": args.max_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": success,
        "final_info": last_info,
        "runtime_s": round(time.monotonic() - started, 3),
        "trace": trace,
    }
    path = args.output / f"{args.task}_seed{args.seed}_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, default=str), flush=True)


if __name__ == "__main__":
    main()
