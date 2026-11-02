"""Full-episode SIMPLER WidowX rollout with the INTACT Bridge pi0 (flow-matching) checkpoint.

Purpose: discriminate between two explanations for the pi0-FAST WidowX failures:
  (a) our rollout / action+state interface is wrong, or
  (b) the pi0-FAST Bridge checkpoint simply cannot drive this embodiment.

This runner uses the *reference* Bridge interface that the INTACT authors validated in SIMPLER
(taken from research/semantic_token_cd/pi0_intact_simpler_token_pcd_smoke.py):
  * R_topdown conjugation of the EEF orientation + p01/p99 state normalisation  (bridge_state)
  * action clipping to [-1, 1], p01/p99 de-normalisation, euler2axangle, binarised gripper
    (decode_bridge_actions)
  * single top camera, 224x224, [0,1] (prepare_image)
and the standard LeRobot PI0Policy.select_action path (the checkpoint's config.json declares
normalization_mapping = IDENTITY, so this runner does the normalisation itself).

Usage (inside the INTACT runtime):
    CUDA_VISIBLE_DEVICES=0 python pi0_intact_widowx_rollout.py --gpu 0 \
        --task widowx_carrot_on_plate --seeds 0-3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
TOKEN_CD = ROOT / "token-cd"
DEFAULT_OUT = TOKEN_CD / "runs/pi0_intact_widowx_rollout"


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.gpu}; got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import gymnasium as gym
    import mani_skill2_real2sim.envs  # noqa: F401
    import torch
    from lerobot.common.constants import OBS_ROBOT
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    sys.path.insert(0, str(TOKEN_CD))
    from research.semantic_token_cd.pi0_intact_simpler_token_pcd_smoke import (  # noqa: E402
        BRIDGE_STATS,
        CHECKPOINT,
        TASK_TO_ENV,
        bridge_state,
        decode_bridge_actions,
        prepare_image,
    )

    if args.task not in TASK_TO_ENV:
        raise SystemExit(f"unknown task {args.task}; known: {list(TASK_TO_ENV)}")

    torch.set_num_threads(1)
    # The INTACT venv inherits a torch/cuDNN combination whose cuDNN SDPA backend fails with
    # "No execution plans support the graph" on the first SigLIP attention call.  Fall back to
    # the flash / memory-efficient / math backends instead.
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception as exc:                                               # noqa: BLE001
        print(json.dumps({"event": "sdpa_backend_note", "detail": repr(exc)}), flush=True)
    device = torch.device("cuda:0")
    stats = json.loads(BRIDGE_STATS.read_text())

    def atomic(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
        os.replace(tmp, path)

    print(json.dumps({"event": "loading_policy", "checkpoint": str(CHECKPOINT), "task": args.task}), flush=True)
    policy = PI0Policy.from_pretrained(str(CHECKPOINT), strict=False).to(device).eval()
    print(json.dumps({"event": "policy_ready", "task": args.task}), flush=True)

    env = gym.make(
        TASK_TO_ENV[args.task],
        obs_mode="rgbd",
        prepackaged_config=True,
        shader_dir="rt",
        renderer_kwargs={"device": "cuda:0", "offscreen_only": True},
        render_config={"rt_max_path_depth": 8, "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
    )
    flags = ("success", "moved_correct_obj", "moved_wrong_obj", "is_src_obj_grasped", "src_on_target")
    try:
        for seed in _parse_seeds(args.seeds):
            out = args.out / args.task / f"intact_seed{seed:03d}.json"
            if out.exists() and json.loads(out.read_text()).get("status") == "complete":
                print(json.dumps({"event": "skip", "task": args.task, "seed": seed}), flush=True)
                continue
            started = time.monotonic()
            obs, reset_info = env.reset(seed=seed)
            instruction = env.get_language_instruction()
            policy.reset()
            steps = 0
            terminated = truncated = False
            info: dict = {}
            success = False
            try:
                while not (terminated or truncated) and steps < args.max_steps:
                    image = get_image_from_maniskill2_obs_dict(env, obs)
                    state = bridge_state(obs, stats)
                    batch = {
                        "observation.images.top": prepare_image(image, device),
                        OBS_ROBOT: torch.from_numpy(state)[None].to(device),
                        "task": [instruction],
                    }
                    with torch.inference_mode():
                        action = policy.select_action(batch)
                    env_actions = decode_bridge_actions(
                        action.detach().float().cpu().numpy(), stats, env.action_space
                    )
                    obs, _reward, terminated, truncated, info = env.step(env_actions[0])
                    success = bool(info.get("success", success))
                    steps += 1
                    if steps % 4 == 0 and not (terminated or truncated):
                        instruction = env.get_language_instruction()
                status = "complete"
                error = None
                trace = None
            except BaseException as exc:                                   # noqa: BLE001
                import traceback
                status, error = "error", f"{type(exc).__name__}: {exc}"
                trace = traceback.format_exc()[-2000:]
            row = {
                "status": status,
                "protocol_id": "PI0_INTACT_WIDOWX_FULL_EPISODE_V1",
                "task": args.task,
                "seed": seed,
                "success": bool(success),
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "runtime_seconds": round(time.monotonic() - started, 3),
                "instruction": instruction,
                "final_info": {k: str(info.get(k)) for k in flags if k in info},
                "checkpoint": str(CHECKPOINT),
                "renderer": {"shader": "rt", "path_depth": 8, "samples_per_pixel": 32, "denoiser": True},
                "error": error,
                "traceback": trace,
            }
            atomic(out, row)
            print(json.dumps({k: row[k] for k in ("task", "seed", "success", "steps", "status", "runtime_seconds")}), flush=True)
    finally:
        env.close()
    print(json.dumps({"event": "worker_complete", "task": args.task}), flush=True)


if __name__ == "__main__":
    main()
