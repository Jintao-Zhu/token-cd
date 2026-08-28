from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_closed_loop.runtime import prepare
from research.coreact_trained_weak.run_libero10_quality_screen import env_config
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_w1_slg_rollout.sampler import ACTIVE_STEPS, sample_w1_slg_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    protocol = json.loads((artifact / "protocol.lock.json").read_text())
    config, policy, preprocessor, _ = load_policy(Path(protocol["strong"]["checkpoint"]))
    cfg = env_config(0)
    env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=config)
    env = cfg.create_envs(n_envs=1, use_async_envs=False)["libero_10"][0]
    try:
        inner = env.envs[0]
        inner.init_state_id = 0
        observation, _ = env.reset(seed=860_000_000)
        batch = prepare(policy, preprocessor, env_pre, observation, inner.task_description)
        noise = torch.randn(
            (1, config.chunk_size, config.max_action_dim),
            generator=torch.Generator(device=batch["state"].device).manual_seed(870_000_000),
            device=batch["state"].device, dtype=batch["state"].dtype,
        )
        with torch.inference_mode():
            official = policy.model.sample_actions(
                batch["images"], batch["image_masks"], batch["lang_tokens"],
                batch["lang_masks"], batch["state"], noise=noise,
            )
            custom, strong_trace = sample_w1_slg_actions(
                policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"],
                batch["lang_masks"], batch["state"], noise, arm="strong",
            )
            traces = {}
            for arm in ("low_w1", "full_w1", "high_w1"):
                _, trace = sample_w1_slg_actions(
                    policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"],
                    batch["lang_masks"], batch["state"], noise, arm=arm,
                )
                traces[arm] = trace
    finally:
        env.close()
    parity = float((official - custom).abs().max())
    checks = {
        "official_custom_strong_max_abs_lt_1e_6": parity < 1e-6,
        "strong_no_active_steps": strong_trace["active_steps"] == [],
        **{f"{arm}_mask_exact": traces[arm]["active_steps"] == sorted(ACTIVE_STEPS[arm]) for arm in traces},
        **{f"{arm}_scale_contract": traces[arm]["scale_contract"] == [1.0] * 15 + [0.0] for arm in traces},
        **{f"{arm}_finite": traces[arm]["all_output_finite"] for arm in traces},
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "official_custom_strong_max_abs": parity, "checks": checks}
    (artifact / "sampler_dry_run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
