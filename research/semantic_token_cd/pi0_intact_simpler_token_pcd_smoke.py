"""Short SIMPLER execution smoke for the INTACT Pi0 token-PCD adapter.

This is deliberately not a success-rate experiment.  It performs one short
closed-loop segment on the Bridge/Spoon task to exercise renderer reset,
Bridge state preprocessing, Pi0 token-PCD inference, action decoding, and a
second observation/replan.  It runs on one explicitly selected GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
CHECKPOINT = ROOT / "checkpoints/pi0_intact_bridge_simpler"
BRIDGE_STATS = ROOT / "open-pi-zero/config/bridge_statistics.json"
TASK_TO_ENV = {
    "widowx_carrot_on_plate": "PutCarrotOnPlateInScene-v0",
    "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v0",
    "widowx_spoon_on_towel": "PutSpoonOnTableClothInScene-v0",
    "widowx_stack_cube": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
}
DEFAULT_OUTPUT = REPO / "artifacts/pi0_intact_token_pcd_simpler_smoke_v1"


def normalize_bound(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (values - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)


def bridge_state(obs: dict, stats: dict) -> np.ndarray:
    from src.utils.geometry import mat2euler, quat2mat

    eef = np.asarray(obs["agent"]["eef_pos"], dtype=np.float64)
    # The Bridge demonstrations express EEF orientation relative to this
    # top-down frame; this matches the published BridgeSimplerAdapter.
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    euler = mat2euler(quat2mat(eef[3:7]) @ default_rot.T)
    raw = np.concatenate([eef[:3], euler, [eef[7]]]).astype(np.float32)
    proprio = stats["proprio"]
    return normalize_bound(raw, np.asarray(proprio["p01"]), np.asarray(proprio["p99"])).astype(np.float32)


def prepare_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    # Match INTACT's Bridge adapter: resize the 3rd-view image to 224 square.
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    return torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float()[None].to(device) / 255.0


def decode_bridge_actions(actions: np.ndarray, stats: dict, action_space) -> np.ndarray:
    from src.utils.geometry import euler2axangle

    normalized = np.asarray(actions, dtype=np.float64).copy()
    if normalized.ndim != 2 or normalized.shape[1] != 7 or not np.isfinite(normalized).all():
        raise ValueError(f"expected finite normalized Bridge actions [T,7], got {normalized.shape}")
    # Bridge action dimensions 0:6 are bound-normalized; gripper stays in
    # [0,1] and is discretized exactly as in BridgeSimplerAdapter.
    normalized[:, :6] = np.clip(normalized[:, :6], -1.0, 1.0)
    action_stats = stats["action"]
    low = np.asarray(action_stats["p01"], dtype=np.float64)
    high = np.asarray(action_stats["p99"], dtype=np.float64)
    raw = np.concatenate(
        [((normalized[:, :6] + 1.0) / 2.0) * (high[:6] - low[:6]) + low[:6], normalized[:, 6:7]],
        axis=1,
    )
    env_actions = np.empty((len(raw), 7), dtype=np.float32)
    for index, row in enumerate(raw):
        axis, angle = euler2axangle(*row[3:6])
        gripper = 1.0 if row[6] > 0.5 else -1.0
        env_actions[index] = np.concatenate([row[:3], axis * angle, [gripper]]).astype(np.float32)
    if not all(action_space.contains(action) for action in env_actions):
        raise ValueError(f"decoded actions exceed SIMPLER action space: {env_actions.tolist()}")
    return env_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--task", choices=tuple(TASK_TO_ENV), default="widowx_spoon_on_towel")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=2, help="small candidate count for a technical smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.gpu}; got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from lerobot.common.constants import OBS_ROBOT
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from research.semantic_token_cd.pi0_intact_token_pcd import Pi0IntactTokenPCDInference
    import gymnasium as gym
    import mani_skill2_real2sim.envs  # noqa: F401
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    stats = json.loads(BRIDGE_STATS.read_text())
    env = gym.make(
        TASK_TO_ENV[args.task],
        obs_mode="rgbd",
        prepackaged_config=True,
        shader_dir="rt",
        renderer_kwargs={"device": "cuda:0", "offscreen_only": True},
        render_config={"rt_max_path_depth": 8, "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
    )
    try:
        observation, _ = env.reset(seed=args.seed)
        instruction = env.get_language_instruction()
        policy = PI0Policy.from_pretrained(str(CHECKPOINT), strict=False).to(device).eval()
        inference = Pi0IntactTokenPCDInference(
            policy,
            layer_index=2,
            n_candidates=args.candidates,
            pcd_alpha=0.2,
            seed=args.seed,
        )
        inference.reset(instruction, seed=args.seed)

        outputs = []
        executed = 0
        started = time.monotonic()
        # Two policy chunks: the first is executed fully, then the updated
        # simulator observation is fed back for a second replan/action.
        for replan in range(2):
            image = get_image_from_maniskill2_obs_dict(env, observation)
            state = bridge_state(observation, stats)
            batch = {
                "observation.images.top": prepare_image(image, device),
                OBS_ROBOT: torch.from_numpy(state)[None].to(device),
                "task": [instruction],
            }
            normalized_actions, trace = inference.predict_action_chunk(batch, seed=args.seed + replan)
            env_actions = decode_bridge_actions(
                normalized_actions[0].detach().float().cpu().numpy(), stats, env.action_space
            )
            n_execute = 4 if replan == 0 else 1
            step_records = []
            for action in env_actions[:n_execute]:
                observation, reward, success, truncated, info = env.step(action)
                executed += 1
                step_records.append({
                    "action": action.astype(float).tolist(),
                    "reward": float(reward),
                    "success": bool(success),
                    "truncated": bool(truncated),
                })
                if truncated:
                    break
            outputs.append({
                "replan_index": replan,
                "state_normalized": state.astype(float).tolist(),
                "pcd_trace": trace,
                "decoded_actions": env_actions.astype(float).tolist(),
                "executed_steps": step_records,
            })
            if step_records[-1]["truncated"]:
                break
            instruction = env.get_language_instruction()

        result = {
            "protocol_id": "PI0_INTACT_TOKEN_PCD_SIMPLER_SMOKE_V1",
            "status": "PASS",
            "task": args.task,
            "seed": args.seed,
            "gpu": args.gpu,
            "renderer": {"shader": "rt", "path_depth": 8, "samples_per_pixel": 32},
            "checkpoint": str(CHECKPOINT),
            "candidate_count_for_smoke": args.candidates,
            "prompt_attention_layer_0based": 2,
            "action_steps_executed": executed,
            "elapsed_seconds": time.monotonic() - started,
            "replans": outputs,
            "note": "Technical pipeline smoke only; not a success-rate episode or method efficacy result.",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        out = args.output / f"{args.task}_seed_{args.seed:03d}_gpu{args.gpu}.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "status": "PASS",
            "artifact": str(out),
            "task": args.task,
            "steps": executed,
            "replans": len(outputs),
            "elapsed_seconds": round(result["elapsed_seconds"], 2),
            "renderer": result["renderer"],
        }), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
