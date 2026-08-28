#!/usr/bin/env python3
"""Verify snapshot restoration/interventions and lock the complete rollout manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_causal_dataset.common import canonical_sha256, stratified_token_groups
from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors, prepare
from research.coreact_region.audit_effect_candidates import restore
from research.coreact_region.effect_existence import array_sha256
from research.coreact_region.fixed_mask_sampler import prepare_ranked_prefix, sample_fixed_mask_actions
from research.coreact_region.segmented_runtime import batched_observation, make_segmented_env, progress_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if (artifact / "rollout_manifest.jsonl").exists():
        raise RuntimeError("rollout manifest already locked")
    plan = read_jsonl(artifact / "snapshot_plan.jsonl")
    records = []
    for spec in plan:
        path = artifact / "clean_extraction" / spec["snapshot_id"] / "trajectory.json"
        if not path.exists():
            raise RuntimeError(f"missing clean extraction {path}")
        records.append(json.loads(path.read_text()))
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",
                       weights_only=True, map_location="cpu")["visual_position_mean"]
    failures, audits, manifest = [], [], []
    for ordinal, spec in enumerate(records, 1):
        env = make_segmented_env(spec["suite"], spec["task_id"])
        env_preprocessor, _ = make_env_pre_post_processors(
            env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=config
        )
        try:
            env.init_state_id = spec["init_state_id"]
            env.reset(seed=spec["reset_seed"])
            state = np.load(artifact / spec["state_path"], allow_pickle=False)
            observation_a = restore(env, state)
            state_hash_a = array_sha256(np.asarray(env._env.get_sim_state()))
            restored_progress_a = progress_snapshot(env)
            prepared_a = prepare(policy, preprocessor, env_preprocessor,
                                 batched_observation(observation_a), spec["language"])
            observation_b = restore(env, state)
            state_hash_b = array_sha256(np.asarray(env._env.get_sim_state()))
            restored_progress_b = progress_snapshot(env)
            prepared_b = prepare(policy, preprocessor, env_preprocessor,
                                 batched_observation(observation_b), spec["language"])
            image_hashes_a = [tensor_sha256(x) for x in prepared_a["images"]]
            image_hashes_b = [tensor_sha256(x) for x in prepared_b["images"]]
            generator = torch.Generator(device=prepared_a["state"].device).manual_seed(spec["proposal_seed"])
            noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator,
                                device=prepared_a["state"].device, dtype=prepared_a["state"].dtype)
            with torch.inference_mode():
                ranked = prepare_ranked_prefix(
                    policy.model, prepared_a["images"], prepared_a["image_masks"],
                    prepared_a["lang_tokens"], prepared_a["lang_masks"], prepared_a["state"], noise,
                )
                clean_a = policy.model.sample_actions(
                    prepared_a["images"], prepared_a["image_masks"], prepared_a["lang_tokens"],
                    prepared_a["lang_masks"], prepared_a["state"], noise=noise,
                )
                clean_b = policy.model.sample_actions(
                    prepared_a["images"], prepared_a["image_masks"], prepared_a["lang_tokens"],
                    prepared_a["lang_masks"], prepared_a["state"], noise=noise,
                )
            eligible = [row.index for row in ranked["span_map"]
                        if row.modality == "visual" and row.intervention_allowed]
            groups = stratified_token_groups(eligible, ranked["scores"], spec["proposal_seed"] + 1)
            group_traces = []
            for group in groups:
                masked_a, trace_a = sample_fixed_mask_actions(
                    policy.model, ranked, noise, means, group["token_indices"]
                )
                masked_b, trace_b = sample_fixed_mask_actions(
                    policy.model, ranked, noise, means, group["token_indices"]
                )
                group_traces.append({
                    **group, "changed_indices": trace_a["changed_indices"],
                    "masked_prefix_sha256": trace_a["masked_prefix_sha256"],
                    "masked_repeat_max_abs_diff": float((masked_a - masked_b).abs().max()),
                    "masked_finite": bool(torch.isfinite(masked_a).all()),
                    "protected_tokens_untouched": trace_a["changed_indices"] == group["token_indices"],
                    "repeat_hash_identity": trace_a["masked_prefix_sha256"] == trace_b["masked_prefix_sha256"],
                })
            valid = all([
                state_hash_a == spec["sim_state_sha256"] == state_hash_b,
                image_hashes_a == image_hashes_b,
                restored_progress_a == restored_progress_b,
                restored_progress_a.grasped == spec["selected_boundary"]["grasped"],
                restored_progress_a.predicate == spec["selected_boundary"]["predicate"],
                abs(restored_progress_a.eef_object_distance
                    - spec["selected_boundary"]["eef_object_distance"]) <= 1e-3,
                abs(restored_progress_a.object_goal_distance
                    - spec["selected_boundary"]["object_goal_distance"]) <= 1e-3,
                float((clean_a - clean_b).abs().max()) <= 1e-6,
                bool(torch.isfinite(clean_a).all()),
                len(eligible) == 128,
                all(row["masked_repeat_max_abs_diff"] <= 1e-6 and row["masked_finite"]
                    and row["protected_tokens_untouched"] and row["repeat_hash_identity"] for row in group_traces),
            ])
            audit = {
                "snapshot_id": spec["snapshot_id"], "valid": valid,
                "exact_state_restore": state_hash_a == spec["sim_state_sha256"] == state_hash_b,
                "deterministic_observation": image_hashes_a == image_hashes_b,
                "restored_progress_parity": {
                    "pass": (
                        restored_progress_a == restored_progress_b
                        and restored_progress_a.grasped == spec["selected_boundary"]["grasped"]
                        and restored_progress_a.predicate == spec["selected_boundary"]["predicate"]
                        and abs(restored_progress_a.eef_object_distance
                                - spec["selected_boundary"]["eef_object_distance"]) <= 1e-3
                        and abs(restored_progress_a.object_goal_distance
                                - spec["selected_boundary"]["object_goal_distance"]) <= 1e-3
                    ),
                    "saved": {key: spec["selected_boundary"][key] for key in (
                        "eef_object_distance", "object_goal_distance", "grasped", "predicate"
                    )},
                    "restored": restored_progress_a.__dict__,
                    "geometry_tolerance_m": 0.001,
                },
                "state_sha256": state_hash_a, "image_sha256": image_hashes_a,
                "native_prefix_sha256": tensor_sha256(ranked["prefix"]),
                "clean_repeat_max_abs_diff": float((clean_a - clean_b).abs().max()),
                "clean_finite": bool(torch.isfinite(clean_a).all()),
                "eligible_visual_tokens": len(eligible), "groups": group_traces,
                "group_definition_sha256": canonical_sha256(groups),
            }
            audit_path = artifact / "candidate_audits" / f"{spec['snapshot_id']}.json"
            write_json(audit_path, audit)
            audits.append(audit)
            if not valid:
                failures.append({"snapshot_id": spec["snapshot_id"], "audit": str(audit_path)})
                continue
            for replicate, rollout_seed in enumerate(spec["rollout_seeds"]):
                manifest.append({
                    **{key: spec[key] for key in ("snapshot_id", "suite", "task_id", "task_name", "language",
                                                   "init_state_id", "reset_seed", "state_path", "sim_state_sha256",
                                                   "assigned_phase", "phase_fallback", "selected_boundary")},
                    "episode_id": f"causal__{spec['snapshot_id']}__clean__r{replicate}",
                    "condition": "clean", "group_id": None, "token_indices": [], "stratum": None,
                    "attention_score": None, "replicate": replicate, "rollout_seed": rollout_seed,
                    "candidate_audit_path": str(audit_path.relative_to(artifact)),
                })
                for group in groups:
                    manifest.append({
                        **{key: spec[key] for key in ("snapshot_id", "suite", "task_id", "task_name", "language",
                                                       "init_state_id", "reset_seed", "state_path", "sim_state_sha256",
                                                       "assigned_phase", "phase_fallback", "selected_boundary")},
                        "episode_id": f"causal__{spec['snapshot_id']}__{group['group_id']}__r{replicate}",
                        "condition": "masked", "replicate": replicate, "rollout_seed": rollout_seed,
                        "candidate_audit_path": str(audit_path.relative_to(artifact)), **group,
                    })
            print(f"{ordinal}/{len(records)} {spec['snapshot_id']} valid={valid}", flush=True)
        finally:
            env.close()
    passed = not failures and len(audits) == 45 and len(manifest) == 1215
    gate = {
        "gate": "snapshot_and_candidate_integrity", "pass": passed,
        "snapshots_expected": 45, "snapshots_audited": len(audits),
        "rollouts_expected": 1215, "rollouts_locked": len(manifest), "failures": failures,
        "all_groups_singleton": all(len(g["token_indices"]) == 1 for a in audits for g in a["groups"]),
        "attention_is_proposal_only": True,
    }
    write_json(artifact / "integrity_gate.json", gate)
    if passed:
        with (artifact / "rollout_manifest.jsonl").open("x") as stream:
            for row in manifest:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        (artifact / "rollout_manifest.lock.yaml").write_text(yaml.safe_dump({
            "rows": len(manifest), "sha256": canonical_sha256(manifest),
            "conditions": {"clean": 135, "masked": 1080}, "gate": gate,
        }, sort_keys=False))
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
