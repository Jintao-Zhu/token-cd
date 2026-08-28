#!/usr/bin/env python3
"""Audit and decide the new-state persistent image intervention gate."""

from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from research.coreact_closed_loop.audit_qualification import video_frames
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_region.effect_existence import HORIZONS


COMPONENTS = ("composite_progress", "reach_progress", "transport_progress", "grasp_ever", "predicate_ever")


def summary(rows, condition, horizon):
    selected = [r for r in rows if r["condition"] == condition and r["horizon"] == horizon]
    values = np.asarray([r["D_composite_progress"] for r in selected])
    return {"condition": condition, "horizon": horizon, "n": len(selected), "mean_D": float(np.mean(values)), "median_D": float(np.median(values)), "fraction_D_gt_0_05": float(np.mean(values > .05)), "median_D_by_suite": {s: float(np.median([r["D_composite_progress"] for r in selected if r["suite"] == s])) for s in sorted({r["suite"] for r in selected})}, "median_D_by_phase": {p: float(np.median([r["D_composite_progress"] for r in selected if r["phase"] == p])) for p in sorted({r["phase"] for r in selected})}, "median_first_chunk_action_rmse": float(np.median([r["first_chunk_action_rmse"] for r in selected]))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args(); artifact = args.artifact.resolve()
    specs = read_jsonl(artifact / "rollout_manifest.jsonl"); records, failures = {}, []
    for spec in specs:
        path = artifact / "episodes" / spec["episode_id"] / "episode.json"
        if not path.exists(): failures.append({"episode_id": spec["episode_id"], "reason": "missing"}); continue
        record = json.loads(path.read_text()); expected = list(range(spec["intervention_replans"]))
        if not (record["control_steps"] == 60 and record["all_outputs_finite"] and record["initial_sim_state_sha256"] == spec["sim_state_sha256"] and record["initial_mask_bundle_sha256_runtime"] == spec["initial_mask_bundle_sha256"] and record["actual_intervention_replans"] == expected and video_frames(artifact / record["video"]) == 60 and len(read_jsonl(artifact / record["step_log"])) == 60): failures.append({"episode_id": spec["episode_id"], "reason": "integrity mismatch"})
        records[spec["episode_id"]] = record
    grouped = defaultdict(dict)
    for r in records.values(): grouped[r["pair_id"]][r["condition"]] = r
    effects = []
    for pair_id, conditions in grouped.items():
        if "vanilla" not in conditions: failures.append({"episode_id": pair_id, "reason": "missing vanilla"}); continue
        vanilla = conditions["vanilla"]; va = np.asarray([r["model_action"] for r in read_jsonl(artifact / vanilla["step_log"])])
        for condition, masked in conditions.items():
            if condition == "vanilla": continue
            if vanilla["initial_sim_state_sha256"] != masked["initial_sim_state_sha256"] or vanilla["noise_sha256_by_replan"] != masked["noise_sha256_by_replan"] or vanilla["initial_mask_bundle_sha256_runtime"] != masked["initial_mask_bundle_sha256_runtime"]: failures.append({"episode_id": masked["episode_id"], "reason": "paired mismatch"}); continue
            ma = np.asarray([r["model_action"] for r in read_jsonl(artifact / masked["step_log"])]); delta = va[:10] - ma[:10]
            for h in HORIZONS:
                vp, mp = vanilla["progress_by_horizon"][str(h)], masked["progress_by_horizon"][str(h)]
                row = {"pair_id": pair_id, "suite": vanilla["suite"], "phase": vanilla["phase"], "condition": condition, "horizon": h, "first_chunk_action_rmse": float(np.sqrt(np.mean(delta**2)))}
                for c in COMPONENTS: row[f"vanilla_{c}"] = float(vp[c]); row[f"mask_{c}"] = float(mp[c]); row[f"D_{c}"] = float(vp[c]) - float(mp[c])
                effects.append(row)
    primary = [r for r in effects if r["condition"] == "target_inpaint_r3" and r["horizon"] == 60]
    p = summary(effects, "target_inpaint_r3", 60); r2 = summary(effects, "target_inpaint_r2", 60); bg = summary(effects, "background_match_target_inpaint_r3", 60); blur = summary(effects, "target_blur_r3", 60)
    variation = sum(abs(r["D_reach_progress"]) > .05 or abs(r["D_transport_progress"]) > .05 or r["D_grasp_ever"] != 0 or r["D_predicate_ever"] != 0 for r in primary)
    suite_positive = len(p["median_D_by_suite"]) == 2 and all(v > 0 for v in p["median_D_by_suite"].values())
    passed = not failures and len(records) == len(specs) and p["median_D"] > .05 and p["fraction_D_gt_0_05"] >= .5 and suite_positive and variation >= 4 and p["median_D"] > bg["median_D"]
    decision = "PERSISTENT_IMAGE_DYNAMIC_RANGE_ESTABLISHED" if passed else "PERSISTENT_IMAGE_DYNAMIC_RANGE_NOT_ESTABLISHED"
    result = {"expected_episodes": len(specs), "actual_episodes": len(records), "integrity_failures": failures, "primary_target_inpaint_r3_H60": p, "target_inpaint_r2_H60": r2, "target_blur_r3_H60": blur, "matched_background_r3_H60": bg, "component_variation_count": variation, "suite_direction_positive": suite_positive, "gate_pass": passed, "decision": decision, "all_summaries": [summary(effects,c,h) for c in sorted({r["condition"] for r in effects}) for h in HORIZONS]}
    write_json(artifact / "analysis_summary.json", result); write_json(artifact / "decision.json", {"status": decision, "gate_pass": passed})
    with (artifact / "paired_effects.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=list(effects[0])); writer.writeheader(); writer.writerows(effects)
    with (artifact / "missingness.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=("episode_id","reason")); writer.writeheader(); writer.writerows(failures)
    figdir = artifact / "figures"; figdir.mkdir(exist_ok=True); fig, ax = plt.subplots(figsize=(8,4.5))
    for c in ("target_inpaint_r2","target_inpaint_r3","target_blur_r3","background_match_target_inpaint_r3"): ax.plot(HORIZONS,[summary(effects,c,h)["median_D"] for h in HORIZONS],marker="o",label=c)
    ax.axhline(.05,color="black",linestyle="--"); ax.axhline(0,color="gray",linewidth=.8); ax.set(xlabel="Horizon",ylabel="Median D"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figdir / "persistent_effect_by_horizon.png",dpi=180); plt.close(fig)
    (artifact / "report.md").write_text(f"# Persistent Image Effect Gate\n\n- Episodes: {len(records)}/{len(specs)}\n- Integrity failures: {len(failures)}\n- Target inpaint r3 H60 median D: {p['median_D']}\n- Fraction D > 0.05: {p['fraction_D_gt_0_05']}\n- Suite medians: {json.dumps(p['median_D_by_suite'],sort_keys=True)}\n- Target inpaint r2 median D: {r2['median_D']}\n- Matched background r3 median D: {bg['median_D']}\n- Component variation: {variation}/{len(primary)}\n\nDecision: `{decision}`\n\nFailure means the tested outcome endpoint still lacks stable dynamic range under three replans; it does not compare attribution methods.\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
