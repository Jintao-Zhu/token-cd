#!/usr/bin/env python3
"""Run mandatory real-batch gates for the direct masked-prefix sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.coreact_closed_loop.guidance import GuidanceConfig, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_revision.masked_sampler import sample_masked_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    tasks = json.loads((artifact / "task_manifest.json").read_text())
    task = next(row for row in tasks if row["suite"] == "libero_spatial")
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(
        workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",
        weights_only=True,
        map_location="cpu",
    )
    env, env_preprocessor, _ = make_task_env(task["suite"], task["task_id"], config)
    try:
        env.envs[0].init_state_id = 0
        observation, _ = env.reset(seed=52_070_001)
        prepared = prepare(policy, preprocessor, env_preprocessor, observation, task["canonical_language"])
        generator = torch.Generator(device=prepared["state"].device).manual_seed(52_070_002_000)
        noise = torch.randn(
            (1, config.chunk_size, config.max_action_dim), generator=generator,
            dtype=prepared["state"].dtype, device=prepared["state"].device,
        )
        with torch.inference_mode():
            native_a = policy.model.sample_actions(
                prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                prepared["lang_masks"], prepared["state"], noise=noise,
            )
            native_b = policy.model.sample_actions(
                prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                prepared["lang_masks"], prepared["state"], noise=noise,
            )
            masked_a, trace_a = sample_masked_actions(
                policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"],
                config=GuidanceConfig(selection="top"), selection_seed=52_070_003_000,
            )
            masked_b, trace_b = sample_masked_actions(
                policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"],
                config=GuidanceConfig(selection="top"), selection_seed=52_070_003_000,
            )
        checks = {
            "policy_eval": not policy.training,
            "all_parameters_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
            "native_determinism_max_abs": float((native_a - native_b).abs().max()),
            "masked_determinism_max_abs": float((masked_a - masked_b).abs().max()),
            "masked_differs_from_native": bool((masked_a - native_a).abs().max() > 0),
            "exactly_eight_changed": len(trace_a["changed_indices"]) == 8
            and len(set(trace_a["changed_indices"])) == 8,
            "selected_equals_changed": sorted(trace_a["selected_indices"]) == sorted(trace_a["changed_indices"]),
            "protected_tokens_untouched": trace_a["protected_tokens_untouched"],
            "selection_repeatable": trace_a["selected_indices"] == trace_b["selected_indices"],
            "prefix_repeatable": trace_a["prefix_sha256"] == trace_b["prefix_sha256"],
            "masked_prefix_repeatable": trace_a["masked_prefix_sha256"] == trace_b["masked_prefix_sha256"],
            "attention_layer_count": trace_a["attention_layer_count"],
            "same_noise_sha256": tensor_sha256(noise),
            "all_outputs_finite": bool(torch.isfinite(native_a).all() and torch.isfinite(masked_a).all()),
            "two_real_camera_masks_valid": len(prepared["images"]) == 2
            and all(bool(mask.all()) for mask in prepared["image_masks"]),
            "state_input_dim": int(prepared["batch"]["observation.state"].shape[-1]),
            "state_model_dim": int(prepared["state"].shape[-1]),
        }
        passed = (
            checks["policy_eval"] and checks["all_parameters_frozen"]
            and checks["native_determinism_max_abs"] <= 1e-6
            and checks["masked_determinism_max_abs"] <= 1e-6
            and checks["masked_differs_from_native"] and checks["exactly_eight_changed"]
            and checks["selected_equals_changed"] and checks["protected_tokens_untouched"]
            and checks["selection_repeatable"] and checks["prefix_repeatable"]
            and checks["masked_prefix_repeatable"] and checks["attention_layer_count"] == 16
            and checks["all_outputs_finite"] and checks["two_real_camera_masks_valid"]
            and checks["state_input_dim"] == 8 and checks["state_model_dim"] == 32
        )
        report = {
            "gate": "direct_mask_sampler_integrity", "pass": passed, "tolerance": 1e-6,
            "checks": checks,
            "output_hashes": {"native": tensor_sha256(native_a), "masked_top8": tensor_sha256(masked_a)},
            "selected_tokens": trace_a["selected_tokens"],
        }
        (artifact / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        lines = ["# Direct Mask Sampler Integrity", "", f"Overall: {'PASS' if passed else 'FAIL'}", "", "| Check | Value |", "|---|---:|"]
        lines.extend(f"| {key} | {value} |" for key, value in checks.items())
        (artifact / "integrity_report.md").write_text("\n".join(lines) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
