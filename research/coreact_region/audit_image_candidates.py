#!/usr/bin/env python3
"""Qualify clean-selected critical states and pre-encoder image interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors, prepare
from research.coreact_region.audit_effect_candidates import restore
from research.coreact_region.effect_existence import array_sha256
from research.coreact_region.image_intervention import (
    CAMERAS,
    SEMANTICS,
    apply_image_condition,
    build_image_masks,
)
from research.coreact_region.prepare_image_effect_existence import CONDITIONS
from research.coreact_region.segmented_runtime import batched_observation, make_segmented_env, raw_observation


def save_preview(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write preview {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    extraction = json.loads((artifact / "critical_extraction_gate.json").read_text())
    if not extraction.get("candidate_minimum_met"):
        raise RuntimeError("clean critical-state count gate failed")
    if (artifact / "rollout_manifest.jsonl").exists():
        raise RuntimeError("image rollout manifest is already locked")
    critical = read_jsonl(artifact / "critical_state_manifest.jsonl")
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    audits, failures, manifest = [], [], []
    first_qualification = None
    for ordinal, spec in enumerate(critical, 1):
        audit_path = artifact / "candidate_audits" / f"{spec['critical_state_id']}.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if not audit.get("candidate_valid") or audit.get("sim_state_sha256") != spec["sim_state_sha256"]:
                raise RuntimeError(f"cached candidate audit is invalid: {audit_path}")
            audits.append(audit)
            for condition in CONDITIONS:
                manifest.append({
                    **spec, "condition": condition,
                    "episode_id": f"image_effect__{spec['critical_state_id']}__{condition}",
                    "pair_id": spec["critical_state_id"],
                    "candidate_audit_path": str(audit_path.relative_to(artifact)),
                    "mask_bundle_sha256": audit["bundle_sha256"],
                })
            print(f"{ordinal}/{len(critical)} {spec['critical_state_id']} cached", flush=True)
            continue
        env = make_segmented_env(spec["suite"], spec["task_id"])
        env_preprocessor, _ = make_env_pre_post_processors(
            env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=config
        )
        try:
            env.init_state_id = spec["init_state_id"]
            env.reset(seed=spec["reset_seed"])
            state = np.load(artifact / spec["state_path"], allow_pickle=False)
            observation = restore(env, state)
            raw = raw_observation(env)
            restored_hash = array_sha256(np.asarray(env._env.get_sim_state()))
            rgb_aligned = all(
                np.array_equal(observation["pixels"][observation_key], raw[rgb_key])
                for _, observation_key, rgb_key, _ in CAMERAS
            )
            bundle, mask_audit = build_image_masks(env, raw, random_seed=spec["selection_seed"])
            counts = {
                key: value for key, value in mask_audit["camera_counts"].items()
                if key != "two_view_totals"
            }
            totals = mask_audit["camera_counts"]["two_view_totals"]
            target_visible = sum(row["target_exact_pixels"] for row in counts.values()) > 0
            goal_visible = sum(row["goal_exact_pixels"] for row in counts.values()) > 0
            area_matched = (
                totals["target_pixels"] == totals["background_match_target_pixels"]
                and totals["goal_pixels"] == totals["background_match_goal_pixels"]
            )
            backgrounds_clean = all(
                row["background_target_overlap_excluded"] == 0
                and row["background_goal_overlap_excluded"] == 0
                for row in counts.values()
            )
            replacement_audits = {}
            for condition in CONDITIONS:
                changed, trace = apply_image_condition(observation, bundle, condition)
                if condition != "vanilla":
                    if any(row["changed_outside_mask"] for row in trace["cameras"].values()):
                        raise RuntimeError("image replacement escaped its mask")
                replacement_audits[condition] = trace
            valid = (
                restored_hash == spec["sim_state_sha256"] and rgb_aligned and target_visible
                and goal_visible and area_matched and backgrounds_clean
            )
            audit = {
                **spec, **mask_audit, "restored_sim_state_sha256": restored_hash,
                "exact_state_restoration": restored_hash == spec["sim_state_sha256"],
                "rgb_segmentation_same_raw_observation": rgb_aligned,
                "target_visible_in_at_least_one_camera": target_visible,
                "goal_visible_in_at_least_one_camera": goal_visible,
                "two_view_background_area_exact": area_matched,
                "background_masks_exclude_target_goal_and_protected": backgrounds_clean,
                "replacement_audits": replacement_audits, "candidate_valid": valid,
            }
            write_json(audit_path, audit)
            if not valid:
                failures.append({"critical_state_id": spec["critical_state_id"], "reason": "state, alignment, visibility, or mask-area failure"})
                continue
            for condition in CONDITIONS:
                manifest.append({
                    **spec, "condition": condition,
                    "episode_id": f"image_effect__{spec['critical_state_id']}__{condition}",
                    "pair_id": spec["critical_state_id"],
                    "candidate_audit_path": str(audit_path.relative_to(artifact)),
                    "mask_bundle_sha256": mask_audit["bundle_sha256"],
                })
            if first_qualification is None:
                generator = torch.Generator(device="cuda").manual_seed(spec["rollout_noise_seed"] * 1000)
                smoke_rows, vanilla_actions = [], None
                for condition in CONDITIONS:
                    changed, trace = apply_image_condition(observation, bundle, condition)
                    prepared = prepare(
                        policy, preprocessor, env_preprocessor, batched_observation(changed), spec["language"]
                    )
                    noise = torch.randn(
                        (1, config.chunk_size, config.max_action_dim), generator=generator,
                        dtype=prepared["state"].dtype, device=prepared["state"].device,
                    )
                    generator.manual_seed(spec["rollout_noise_seed"] * 1000)
                    with torch.inference_mode():
                        actions = policy.model.sample_actions(
                            prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                            prepared["lang_masks"], prepared["state"], noise=noise,
                        )
                    if condition == "vanilla":
                        vanilla_actions = actions
                    smoke_rows.append({
                        "condition": condition, "raw_trace": trace,
                        "preprocessed_image_sha256": [tensor_sha256(image) for image in prepared["images"]],
                        "actions_sha256": tensor_sha256(actions),
                        "action_rmse_vs_vanilla": None if vanilla_actions is None else float(
                            torch.sqrt(torch.mean((actions - vanilla_actions) ** 2))
                        ),
                        "finite": bool(torch.isfinite(actions).all()),
                    })
                    for _, observation_key, _, _ in CAMERAS:
                        save_preview(
                            artifact / "qualification_previews" / condition / f"{observation_key}.png",
                            changed["pixels"][observation_key],
                        )
                first_qualification = {
                    "critical_state_id": spec["critical_state_id"], "conditions": smoke_rows,
                    "policy_eval": not policy.training,
                    "all_parameters_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
                }
            audits.append(audit)
            print(
                f"{ordinal}/{len(critical)} {spec['critical_state_id']} "
                f"target={sum(x['target_pixels'] for x in counts.values())} "
                f"goal={sum(x['goal_pixels'] for x in counts.values())}", flush=True,
            )
        finally:
            env.close()
    phase_suite = {
        suite: {phase: sum(row["suite"] == suite and row["phase"] == phase for row in audits) for phase in ("pre_grasp", "pre_place", "near_reach", "near_place")}
        for suite in sorted({row["suite"] for row in audits})
    }
    true_pre_place_each_suite = all(row["pre_place"] >= 1 for row in phase_suite.values())
    smoke_pass = (
        first_qualification is not None and first_qualification["policy_eval"]
        and first_qualification["all_parameters_frozen"]
        and all(row["finite"] for row in first_qualification["conditions"])
    )
    passed = (
        not failures and len(audits) >= 6 and true_pre_place_each_suite and smoke_pass
        and all(sum(1 for row in audits if row["suite"] == suite) >= 2 for suite in phase_suite)
    )
    with (artifact / "rollout_manifest.jsonl").open("x") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    gate = {
        "gate": "pre_encoder_image_intervention_qualification", "pass": passed,
        "critical_states": len(audits), "phase_counts_by_suite": phase_suite,
        "true_pre_place_each_suite": true_pre_place_each_suite,
        "rollout_episodes": len(manifest), "failures": failures,
        "forward_smoke": first_qualification,
    }
    write_json(artifact / "candidate_gate.json", gate)
    (artifact / "candidate.lock.yaml").write_text(yaml.safe_dump(gate, sort_keys=False))
    print(json.dumps({key: gate[key] for key in ("pass", "critical_states", "phase_counts_by_suite", "rollout_episodes", "failures")}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
