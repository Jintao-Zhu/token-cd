"""Check that a fresh depth0 worker reproduces a full-RT worker shard's first seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import simpler_env

from research.semantic_token_cd.univla_l14_google_rollout import capture_snapshot, restore_snapshot

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
FULL = ROOT / "artifacts/univla_fractal_google_ft_30k_vanilla_vs_l14matched_rtfull_100_v1/episodes"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    env_id, base = simpler_env.ENVIRONMENT_MAP[args.task]
    kwargs = dict(base)
    kwargs.update({
        "obs_mode": "rgbd", "prepackaged_config": True, "shader_dir": "rt",
        "render_config": {"rt_samples_per_pixel": 32, "rt_max_path_depth": 0, "rt_use_denoiser": True},
        "renderer_kwargs": {"device": "cuda:0", "offscreen_only": True},
    })
    env = gym.make(env_id, **kwargs)
    try:
        snapshot = capture_snapshot(env, args.seed)
        _, _, state_sha, rgb_sha = restore_snapshot(env, args.seed, snapshot)
        ref = json.loads((FULL / args.task / "vanilla" / f"seed_{args.seed:03d}.json").read_text())
        result = {
            "task": args.task, "seed": args.seed,
            "state_match": state_sha == ref["initial_state_sha256"],
            "instruction_match": snapshot["instruction"] == ref["instruction"],
            "rgb_diff_expected": rgb_sha != ref["initial_rgb_sha256"],
            "state_sha256": state_sha, "reference_state_sha256": ref["initial_state_sha256"],
        }
        print(json.dumps(result), flush=True)
        if not result["state_match"] or not result["instruction_match"]:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
