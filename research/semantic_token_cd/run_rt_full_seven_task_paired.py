#!/usr/bin/env python3
"""Run historical Vanilla / dynamic VTCG on SIMPLER RT path depth 8.

This thin wrapper deliberately reuses the audited depth-0 rollout implementation
so policy, action decoding, snapshot restore, logging, and matched-budget logic
stay identical. The only changed experimental variable is renderer path depth.
"""
from __future__ import annotations

import sys

from research.semantic_token_cd import replay_simpler_l11_rank_effects as replay
from research.semantic_token_cd import run_rt_depth0_seven_task_paired as rollout

PROTOCOL = "SIMPLER_RT_FULL_7TASK_L11_MATCHED_PAIRED_0_299_V1"
RENDERER = "SAPIEN RT full depth8, spp=32, denoiser=true"


def make_rt_full_environment(task: str):
    import gymnasium as gym
    import simpler_env

    render = {"rt_samples_per_pixel": 32, "rt_max_path_depth": 8,
              "rt_use_denoiser": True}
    renderer = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0", obs_mode="rgbd",
            prepackaged_config=True, distractor_config="less", shader_dir="rt",
            render_config=render, renderer_kwargs=renderer,
        )
    env_id, base = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base)
    kwargs.update({
        "prepackaged_config": True, "shader_dir": "rt",
        "render_config": render, "renderer_kwargs": renderer,
    })
    return gym.make(env_id, obs_mode="rgbd", **kwargs)


def main() -> None:
    # The shared rollout checks historical RT0 records for identity. For this
    # cross-renderer study, validate against this run's paired RT8 outputs
    # instead; canonical snapshot and simulator-state checks remain unchanged.
    replay.make_rt_depth0_environment = make_rt_full_environment
    rollout.PROTOCOL = PROTOCOL
    rollout.RENDERER = RENDERER
    rollout.RT0_REFERENCE = rollout.ROOT / "artifacts/simpler_rt_full_7task_4200_v1/episodes"
    rollout.main()


if __name__ == "__main__":
    main()
