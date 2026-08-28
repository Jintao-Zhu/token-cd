#!/usr/bin/env python3
"""Audit new confirmation states and lock persistent image conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_region.audit_effect_candidates import restore
from research.coreact_region.effect_existence import array_sha256
from research.coreact_region.image_intervention import CAMERAS, apply_image_condition, build_image_masks
from research.coreact_region.prepare_persistent_image_effect import CONDITIONS
from research.coreact_region.segmented_runtime import make_segmented_env, raw_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if (artifact / "rollout_manifest.jsonl").exists():
        raise RuntimeError("persistent rollout manifest already locked")
    critical = read_jsonl(artifact / "critical_state_manifest.jsonl")
    audits, failures, manifest = [], [], []
    for ordinal, spec in enumerate(critical, 1):
        audit_path = artifact / "candidate_audits" / f"{spec['critical_state_id']}.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
        else:
            env = make_segmented_env(spec["suite"], spec["task_id"])
            try:
                env.init_state_id = spec["init_state_id"]
                env.reset(seed=spec["reset_seed"])
                state = np.load(artifact / spec["state_path"], allow_pickle=False)
                observation = restore(env, state)
                raw = raw_observation(env)
                restored = array_sha256(np.asarray(env._env.get_sim_state()))
                aligned = all(np.array_equal(observation["pixels"][obs], raw[rgb]) for _, obs, rgb, _ in CAMERAS)
                bundle, mask_audit = build_image_masks(env, raw, random_seed=spec["selection_seed"] * 100)
                totals = mask_audit["camera_counts"]["two_view_totals"]
                area = totals["target_pixels"] == totals["background_match_target_pixels"] and totals["goal_pixels"] == totals["background_match_goal_pixels"]
                traces = {
                    name: apply_image_condition(observation, bundle, cfg["base_condition"])[1]
                    for name, cfg in CONDITIONS.items()
                }
                valid = restored == spec["sim_state_sha256"] and aligned and area
                audit = {**spec, **mask_audit, "restored_sim_state_sha256": restored, "rgb_aligned": aligned, "joint_area_exact": area, "condition_traces": traces, "candidate_valid": valid}
                write_json(audit_path, audit)
            finally:
                env.close()
        audits.append(audit)
        if not audit.get("candidate_valid"):
            failures.append({"critical_state_id": spec["critical_state_id"], "reason": "candidate invalid"})
            continue
        for condition, config in CONDITIONS.items():
            manifest.append({
                **spec, "condition": condition, **config,
                "episode_id": f"persistent_image__{spec['critical_state_id']}__{condition}",
                "pair_id": spec["critical_state_id"],
                "initial_mask_bundle_sha256": audit["bundle_sha256"],
            })
        print(f"{ordinal}/{len(critical)} {spec['critical_state_id']}", flush=True)
    phases = {suite: {phase: sum(r["suite"] == suite and r["phase"] == phase for r in audits) for phase in ("pre_grasp", "pre_place", "near_reach", "near_place")} for suite in sorted({r["suite"] for r in audits})}
    passed = not failures and len(audits) >= 6 and all(v["pre_place"] >= 1 for v in phases.values())
    with (artifact / "rollout_manifest.jsonl").open("x") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    gate = {"gate": "persistent_image_new_state_qualification", "pass": passed, "critical_states": len(audits), "phase_counts_by_suite": phases, "rollout_episodes": len(manifest), "failures": failures}
    write_json(artifact / "candidate_gate.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
