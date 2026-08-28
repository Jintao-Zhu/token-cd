from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.coreact_closed_loop.runtime import make_task_env, prepare
from research.coreact_trained_weak.runtime import load_locked_pair
from research.coreact_trained_weak.sampler import applied_correction, sample_trained_weak_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    _, strong, weak = load_locked_pair(artifact)
    config, strong_policy, preprocessor, _ = strong
    _, weak_policy, _, _ = weak
    env, env_preprocessor, _ = make_task_env("libero_spatial", 4, config)
    try:
        inner = env.envs[0]
        inner.init_state_id = 10
        observation, _ = env.reset(seed=630_004_100)
        batch = prepare(strong_policy, preprocessor, env_preprocessor, observation, inner.task_description)
        noise = torch.randn(
            (1, config.chunk_size, config.max_action_dim),
            generator=torch.Generator(device="cuda").manual_seed(640_004_100),
            device="cuda",
        )
        native = strong_policy.model.sample_actions(
            batch["images"], batch["image_masks"], batch["lang_tokens"],
            batch["lang_masks"], batch["state"], noise=noise,
        )
        vanilla, trace_v = sample_trained_weak_actions(
            strong_policy.model, weak_policy.model,
            batch["images"], batch["image_masks"], batch["lang_tokens"],
            batch["lang_masks"], batch["state"], noise, arm="V", lambda_value=0.25,
            return_debug=True,
        )
        outputs = {}
        for arm in ("W", "M", "G"):
            outputs[arm], _ = sample_trained_weak_actions(
                strong_policy.model, weak_policy.model,
                batch["images"], batch["image_masks"], batch["lang_tokens"],
                batch["lang_masks"], batch["state"], noise, arm=arm, lambda_value=0.25,
            )
        debug = trace_v.pop("_debug")
        correction, scale = applied_correction(debug["strong"], debug["weak"], 0.25)
        result = {
            "status": "PASS",
            "native_vanilla_max_abs": float((native - vanilla).abs().max()),
            "all_outputs_finite": all(bool(torch.isfinite(value).all()) for value in [native, vanilla, *outputs.values()]),
            "strong_weak_not_identical": bool((outputs["W"] - vanilla).abs().max() > 0),
            "midpoint_not_identical": bool((outputs["M"] - vanilla).abs().max() > 0),
            "cfg_not_identical": bool((outputs["G"] - vanilla).abs().max() > 0),
            "step0_correction_ratio": float(torch.linalg.vector_norm(correction) / (torch.linalg.vector_norm(debug["strong"]) + 1e-12)),
            "step0_clip_scale": float(scale),
            "fresh_init_state": 10,
        }
        if result["native_vanilla_max_abs"] > 1e-5 or not all(
            result[key] for key in ("all_outputs_finite", "strong_weak_not_identical", "midpoint_not_identical", "cfg_not_identical")
        ):
            raise RuntimeError(result)
        (artifact / "trained_weak_sampler_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        env.close()


if __name__ == "__main__":
    main()
