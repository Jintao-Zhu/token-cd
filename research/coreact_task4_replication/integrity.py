#!/usr/bin/env python3
"""Mandatory real-state gate for the task-4 three-condition replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_revision.masked_sampler import sample_masked_actions


def prepared_hash(prepared: dict) -> str:
    values = [*prepared["images"], *prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"]]
    return hashlib.sha256("".join(tensor_sha256(value) for value in values).encode("ascii")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True); parser.add_argument("--reset-count", type=int, default=1); args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    task = json.loads((artifact / "task_manifest.json").read_text())
    spec = json.loads((artifact / "episode_manifest.jsonl").read_text().splitlines()[0])
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    env, env_preprocessor, _ = make_task_env(task["suite"], task["task_id"], config)
    try:
        env.envs[0].init_state_id = spec["init_state_id"]
        for _ in range(args.reset_count):
            observation, _ = env.reset(seed=spec["reset_seed"])
        prepared = prepare(policy, preprocessor, env_preprocessor, observation, task["language"])
        generator = torch.Generator(device=prepared["state"].device).manual_seed(spec["action_noise_seed"] * 1000)
        noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, dtype=prepared["state"].dtype, device=prepared["state"].device)
        guidance_config = GuidanceConfig(selection="top", group_count=8, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10)
        with torch.inference_mode():
            native_a = policy.model.sample_actions(prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise=noise)
            native_b = policy.model.sample_actions(prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise=noise)
            guided_a, guided_trace_a = sample_coreact_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"], config=guidance_config, selection_seed=spec["selection_seed"] * 1000)
            guided_b, guided_trace_b = sample_coreact_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"], config=guidance_config, selection_seed=spec["selection_seed"] * 1000)
            masked_a, masked_trace_a = sample_masked_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"], config=guidance_config, selection_seed=spec["selection_seed"] * 1000)
            masked_b, masked_trace_b = sample_masked_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"], config=guidance_config, selection_seed=spec["selection_seed"] * 1000)
        selected = guided_trace_a["selected_indices"]
        protected = all(token["modality"] == "visual" and token["intervention_allowed"] for token in guided_trace_a["selected_tokens"])
        checks = {
            "policy_eval": not policy.training,
            "all_parameters_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
            "native_determinism_max_abs": float((native_a - native_b).abs().max()),
            "guidance_determinism_max_abs": float((guided_a - guided_b).abs().max()),
            "mask_only_determinism_max_abs": float((masked_a - masked_b).abs().max()),
            "exactly_eight_visual_tokens_changed": len(selected) == len(set(selected)) == 8 and len(guided_trace_a["changed_indices"]) == 8 and len(masked_trace_a["changed_indices"]) == 8,
            "guidance_selected_equals_changed": sorted(selected) == sorted(guided_trace_a["changed_indices"]),
            "mask_selected_equals_changed": sorted(masked_trace_a["selected_indices"]) == sorted(masked_trace_a["changed_indices"]),
            "coreact_mask_top8_identical": selected == masked_trace_a["selected_indices"],
            "native_prefix_hash_identical": guided_trace_a["prefix_sha256"] == masked_trace_a["prefix_sha256"],
            "masked_prefix_hash_identical": guided_trace_a["negative_prefix_sha256"] == masked_trace_a["masked_prefix_sha256"],
            "protected_language_state_special_padding_untouched": protected and guided_trace_a["protected_tokens_untouched"] and masked_trace_a["protected_tokens_untouched"],
            "same_noise_sha256": tensor_sha256(noise),
            "all_actions_finite": bool(torch.isfinite(native_a).all() and torch.isfinite(guided_a).all() and torch.isfinite(masked_a).all()),
            "all_velocity_steps_finite": all(row["finite"] for row in guided_trace_a["step_traces"] + masked_trace_a["step_traces"]),
            "guidance_virtual_dimensions_zero": all(row["virtual_guidance_norm"] == 0 for row in guided_trace_a["step_traces"]),
            "attention_layers": guided_trace_a["attention_layer_count"],
            "two_real_camera_inputs": len(prepared["images"]) == 2 and all(bool(mask.all()) for mask in prepared["image_masks"]),
        }
        passed = all(value if isinstance(value, bool) else True for value in checks.values()) and max(checks["native_determinism_max_abs"], checks["guidance_determinism_max_abs"], checks["mask_only_determinism_max_abs"]) <= 1e-6 and checks["attention_layers"] == 16
        report = {
            "gate": "task4_coreact_and_mask_only_integrity", "pass": passed, "tolerance": 1e-6,
            "checks": checks, "prepared_input_sha256": prepared_hash(prepared),
            "prefix_hashes": {"native": guided_trace_a["prefix_sha256"], "masked": guided_trace_a["negative_prefix_sha256"]},
            "selected_indices": selected, "selected_tokens": guided_trace_a["selected_tokens"],
            "output_hashes": {"native": tensor_sha256(native_a), "guided": tensor_sha256(guided_a), "mask_only": tensor_sha256(masked_a)},
        }
        (artifact / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (artifact / "integrity_report.md").write_text("# Task-4 Replication Integrity\n\nOverall: **" + ("PASS" if passed else "FAIL") + "**\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in checks.items()) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        if not passed: raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__": main()
