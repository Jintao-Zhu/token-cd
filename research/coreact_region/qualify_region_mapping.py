#!/usr/bin/env python3
"""Qualify segmentation, connector topology, region selection, and fixed-mask parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector

from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors, prepare
from research.coreact_region.fixed_mask_sampler import prepare_ranked_prefix, sample_fixed_mask_actions
from research.coreact_region.region_mapping import protected_instance_names
from research.coreact_region.run_region_diagnostic import build_regions, bytes_hash
from research.coreact_region.segmented_runtime import (
    batched_observation,
    make_segmented_env,
    progress_snapshot,
    raw_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    spec = json.loads((artifact / "episode_manifest.jsonl").read_text().splitlines()[0])
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    env = make_segmented_env(spec["suite"], spec["task_id"])
    env_preprocessor, _ = make_env_pre_post_processors(
        env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=config
    )
    try:
        env.init_state_id = spec["init_state_id"]
        observation, _ = env.reset(seed=spec["reset_seed"])
        raw = raw_observation(env)
        rgb_exact = all(
            np.array_equal(observation["pixels"][dst], raw[src])
            for dst, src in (("image", "agentview_image"), ("image2", "robot0_eye_in_hand_image"))
        )
        prepared = prepare(
            policy, preprocessor, env_preprocessor, batched_observation(observation), spec["language"]
        )
        generator = torch.Generator(device=prepared["state"].device).manual_seed(spec["action_noise_seed"] * 1000)
        noise = torch.randn(
            (1, config.chunk_size, config.max_action_dim), generator=generator,
            dtype=prepared["state"].dtype, device=prepared["state"].device,
        )
        with torch.inference_mode():
            ranked = prepare_ranked_prefix(
                policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                prepared["lang_masks"], prepared["state"], noise,
            )
            mapped, groups, count = build_regions(env, raw, ranked, spec["selection_seed"])
            masked_a, trace_a = sample_fixed_mask_actions(
                policy.model, ranked, noise, means["visual_position_mean"], groups["relevant_high"]
            )
            masked_b, trace_b = sample_fixed_mask_actions(
                policy.model, ranked, noise, means["visual_position_mean"], groups["relevant_high"]
            )

        synthetic = torch.arange(32 * 32).reshape(1, 32 * 32, 1)
        connector = object.__new__(SmolVLMConnector)
        shuffled = SmolVLMConnector.pixel_shuffle(connector, synthetic, 4)
        expected_first = [0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99]
        regions_by_name = {}
        for row in mapped:
            regions_by_name[row["region"]] = regions_by_name.get(row["region"], 0) + 1
        instances = env._env.env.model.instances_to_ids
        relevant_names = sorted(env._env.obj_of_interest)
        protected_names = sorted(protected_instance_names(instances))
        progress = progress_snapshot(env)
        checks = {
            "policy_eval": not policy.training,
            "all_parameters_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
            "rgb_segmentation_same_raw_observation": rgb_exact,
            "two_segmentation_views_256_square": all(raw[key].shape == (256, 256, 1) for key in ("agentview_segmentation_instance", "robot0_eye_in_hand_segmentation_instance")),
            "connector_uses_no_resampler": not policy.model.vlm_with_expert.get_vlm_model().config.text_config.use_resampler,
            "connector_scale_factor": int(policy.model.vlm_with_expert.get_vlm_model().config.scale_factor),
            "connector_first_token_source_patches": shuffled[0, 0].tolist(),
            "connector_topology_expected": shuffled.shape == (1, 64, 16) and shuffled[0, 0].tolist() == expected_first,
            "visual_prefix_tokens": len(mapped),
            "matched_group_count": count,
            "all_four_groups_same_count": len({len(value) for value in groups.values()}) == 1,
            "group_indices_unique": all(len(value) == len(set(value)) for value in groups.values()),
            "protected_selected": any(row["region"] == "protected" and row["prefix_index"] in {i for values in groups.values() for i in values} for row in mapped),
            "fixed_mask_determinism_max_abs": float((masked_a - masked_b).abs().max()),
            "selected_equals_changed": sorted(trace_a["selected_indices"]) == sorted(trace_a["changed_indices"]),
            "fixed_mask_output_finite": bool(torch.isfinite(masked_a).all()),
            "attention_layer_count": ranked["attention_layer_count"],
            "progress_metrics_finite": all(np.isfinite(value) for value in (progress.eef_object_distance, progress.object_goal_distance)),
        }
        passed = (
            checks["policy_eval"] and checks["all_parameters_frozen"] and checks["rgb_segmentation_same_raw_observation"]
            and checks["two_segmentation_views_256_square"] and checks["connector_uses_no_resampler"]
            and checks["connector_scale_factor"] == 4 and checks["connector_topology_expected"]
            and checks["visual_prefix_tokens"] == 128 and checks["matched_group_count"] >= 2
            and checks["all_four_groups_same_count"] and checks["group_indices_unique"]
            and not checks["protected_selected"] and checks["fixed_mask_determinism_max_abs"] <= 1e-6
            and checks["selected_equals_changed"] and checks["fixed_mask_output_finite"]
            and checks["attention_layer_count"] == 16 and checks["progress_metrics_finite"]
        )
        report = {
            "gate": "segmentation_connector_region_qualification", "pass": passed,
            "checks": checks, "regions_by_name": regions_by_name,
            "relevant_instance_names": relevant_names, "protected_instance_names": protected_names,
            "groups": groups, "initial_sim_state_sha256": bytes_hash(env._env.get_sim_state()),
            "noise_sha256": tensor_sha256(noise), "masked_output_sha256": tensor_sha256(masked_a),
            "mapping_example": mapped,
            "interpretation": "Token regions refer to ordered source footprints; vision tokens are globally contextualized and are not strict local receptive fields.",
        }
        (artifact / "qualification_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
