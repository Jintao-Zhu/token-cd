#!/usr/bin/env python3
"""Mandatory repeated real-state gate for consistency-guided rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.coreact_closed_loop.guidance import GuidanceConfig, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_revision.consistency_guidance import sample_consistency_guided_actions
from research.coreact_revision.run_online_consistency_qualification import fit_task_heldout_classifier


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, required=True); p.add_argument("--artifact", type=Path, required=True); args = p.parse_args()
    w, artifact = args.workspace.resolve(), args.artifact.resolve(); spec = json.loads((artifact / "episode_manifest.jsonl").read_text().splitlines()[1])
    source = w / "artifacts/coreact_region_sign_multitask_v1_20260808_093747"; ablation = w / "artifacts/coreact_action_consistency_ablation_v1_20260808_113116"
    classifier, _, numeric = fit_task_heldout_classifier(source / "raw_effects.jsonl", ablation / "consistency_scores.jsonl", heldout_task=spec["task_id"])
    cfg, policy, preprocessor, _ = load_policy_and_processors(w); means = torch.load(w / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    env, env_pre, _ = make_task_env(spec["suite"], spec["task_id"], cfg)
    try:
        env.envs[0].init_state_id = spec["init_state_id"]; observation, _ = env.reset(seed=spec["reset_seed"]); prepared = prepare(policy, preprocessor, env_pre, observation, spec["language"])
        noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=torch.Generator(device=prepared["state"].device).manual_seed(spec["action_noise_seed"] * 1000), device=prepared["state"].device, dtype=prepared["state"].dtype)
        g = GuidanceConfig(group_count=8, guidance_scale=.5, trust_region_kappa=.25, action_dim=7, num_steps=10)
        with torch.inference_mode():
            a, ta = sample_consistency_guided_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means, classifier, numeric, config=g)
            b, tb = sample_consistency_guided_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means, classifier, numeric, config=g)
        checks = {"policy_eval": not policy.training, "parameters_frozen": not any(p.requires_grad for p in policy.parameters()), "output_repeat_max_abs": float((a-b).abs().max()), "selected_repeatable": ta["selected_indices"] == tb["selected_indices"], "exactly_8_changed": len(ta["changed_indices"]) == len(set(ta["changed_indices"])) == 8, "selected_equals_changed": sorted(ta["selected_indices"]) == sorted(ta["changed_indices"]), "prefix_repeatable": ta["prefix_sha256"] == tb["prefix_sha256"], "negative_prefix_repeatable": ta["negative_prefix_sha256"] == tb["negative_prefix_sha256"], "same_noise": ta["noise_sha256"] == tb["noise_sha256"] == tensor_sha256(noise), "protected_tokens_untouched": ta["protected_tokens_untouched"] and tb["protected_tokens_untouched"], "all_outputs_finite": ta["all_output_finite"] and tb["all_output_finite"] and bool(torch.isfinite(a).all()), "attention_layers": ta["attention_layer_count"], "candidate_count": len(ta["candidate_indices"])}
        passed = bool(checks["policy_eval"] and checks["parameters_frozen"] and checks["output_repeat_max_abs"] <= 1e-6 and checks["selected_repeatable"] and checks["exactly_8_changed"] and checks["selected_equals_changed"] and checks["prefix_repeatable"] and checks["negative_prefix_repeatable"] and checks["same_noise"] and checks["protected_tokens_untouched"] and checks["all_outputs_finite"] and checks["attention_layers"] == 16 and checks["candidate_count"] == 32)
        report = {"gate": "consistency_closed_loop_sampler_integrity", "pass": passed, "checks": checks, "selected_indices": ta["selected_indices"], "selected_probabilities": ta["selected_probabilities"], "output_sha256": tensor_sha256(a)}
        (artifact / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); print(json.dumps(report, indent=2, sort_keys=True))
        if not passed: raise SystemExit(1)
    finally: env.close()


if __name__ == "__main__": main()
