#!/usr/bin/env python3
"""Audit actual token counts at clean-selected critical states and lock rollout conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors, prepare
from research.coreact_region.effect_existence import array_sha256, build_effect_groups
from research.coreact_region.fixed_mask_sampler import prepare_ranked_prefix
from research.coreact_region.segmented_runtime import (
    batched_observation,
    make_segmented_env,
    raw_observation,
)


def restore(env, state: np.ndarray) -> dict:
    raw = env._env.regenerate_obs_from_state(state)
    return env._format_raw_obs(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    extraction = json.loads((artifact / "critical_extraction_gate.json").read_text())
    if not extraction.get("candidate_minimum_met"):
        raise RuntimeError("critical-state minimum gate failed")
    if (artifact / "rollout_manifest.jsonl").exists():
        raise RuntimeError("candidate rollout manifest already locked")
    critical = read_jsonl(artifact / "critical_state_manifest.jsonl")
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    audits, failures, rollout_rows = [], [], []
    for ordinal, spec in enumerate(critical):
        env = make_segmented_env(spec["suite"], spec["task_id"])
        env_preprocessor, _ = make_env_pre_post_processors(
            env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=config
        )
        try:
            env.init_state_id = spec["init_state_id"]
            env.reset(seed=spec["reset_seed"])
            state = np.load(artifact / spec["state_path"], allow_pickle=False)
            observation = restore(env, state)
            restored_hash = array_sha256(np.asarray(env._env.get_sim_state()))
            prepared = prepare(
                policy, preprocessor, env_preprocessor, batched_observation(observation), spec["language"]
            )
            generator = torch.Generator(device=prepared["state"].device).manual_seed(
                spec["rollout_noise_seed"] * 1000
            )
            noise = torch.randn(
                (1, config.chunk_size, config.max_action_dim), generator=generator,
                dtype=prepared["state"].dtype, device=prepared["state"].device,
            )
            with torch.inference_mode():
                ranked = prepare_ranked_prefix(
                    policy.model, prepared["images"], prepared["image_masks"],
                    prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise,
                )
            mapped, groups, audit = build_effect_groups(
                env, raw_observation(env), ranked, random_seed=spec["selection_seed"]
            )
            required_controls = [
                "full_target", "background_random_match_full_target"
            ]
            valid = restored_hash == spec["sim_state_sha256"] and all(
                name in groups and groups[name] for name in required_controls
            )
            row = {
                **spec, **audit, "restored_sim_state_sha256": restored_hash,
                "exact_state_restoration": restored_hash == spec["sim_state_sha256"],
                "groups": groups, "mapping": mapped, "candidate_valid": valid,
            }
            audit_path = artifact / "candidate_audits" / f"{spec['critical_state_id']}.json"
            write_json(audit_path, row)
            row["candidate_audit_path"] = str(audit_path.relative_to(artifact))
            audits.append(row)
            if not valid:
                failures.append({"critical_state_id": spec["critical_state_id"], "reason": "state restoration or full-target/background group failure"})
                continue
            conditions = [("clean", None, None)]
            for group_name in sorted(groups):
                if group_name.startswith("background_random_match_"):
                    continue
                conditions.append((group_name, group_name, "position_mean"))
                background_name = f"background_random_match_{group_name}"
                if background_name in groups:
                    conditions.append((background_name, background_name, "position_mean"))
            conditions.extend([
                ("full_target_zero", "full_target", "zero"),
                ("background_random_match_full_target_zero", "background_random_match_full_target", "zero"),
            ])
            for condition, group_name, replacement in conditions:
                rollout_rows.append({
                    **spec, "condition": condition, "group_name": group_name,
                    "replacement_type": replacement,
                    "candidate_audit_path": str(audit_path.relative_to(artifact)),
                    "episode_id": f"effect__{spec['critical_state_id']}__{condition}",
                    "pair_id": spec["critical_state_id"],
                })
            print(
                f"{ordinal + 1}/{len(critical)} {spec['critical_state_id']} "
                f"target={audit['target_token_count']} relevant={audit['task_relevant_token_count']} "
                f"k={audit['supported_k']}", flush=True,
            )
        finally:
            env.close()
    with (artifact / "rollout_manifest.jsonl").open("x") as stream:
        for row in rollout_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    counts = {
        suite: sum(row["candidate_valid"] and row["suite"] == suite for row in audits)
        for suite in sorted({row["suite"] for row in audits})
    }
    passed = not failures and len(audits) >= 6 and all(count >= 2 for count in counts.values())
    gate = {
        "gate": "effect_candidate_audit", "pass": passed,
        "critical_states": len(audits), "valid_by_suite": counts,
        "rollout_conditions": len(rollout_rows), "failures": failures,
        "actual_counts": [{key: row[key] for key in (
            "critical_state_id", "suite", "phase", "target_token_count",
            "task_relevant_token_count", "eligible_background_token_count", "supported_k"
        )} for row in audits],
        "no_groups_filled_with_background": True,
    }
    write_json(artifact / "candidate_gate.json", gate)
    (artifact / "candidate.lock.yaml").write_text(yaml.safe_dump(gate, sort_keys=False))
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
