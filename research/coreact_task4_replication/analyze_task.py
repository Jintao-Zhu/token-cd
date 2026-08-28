#!/usr/bin/env python3
"""Analyze one 50-state LIBERO-Spatial three-condition replication."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from research.coreact_closed_loop.audit_qualification import video_frames
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json


CONDITIONS = ("vanilla", "coreact_top8", "top8_mask_only")
COMPARISONS = (("coreact_top8", "vanilla"), ("top8_mask_only", "vanilla"), ("coreact_top8", "top8_mask_only"))


def paired_comparison(rows: list[dict], first: str, second: str) -> dict:
    a = np.asarray([row[f"{first}_success"] for row in rows], dtype=float)
    b = np.asarray([row[f"{second}_success"] for row in rows], dtype=float)
    delta = a - b
    rng = np.random.default_rng(77_007_007)
    estimates = [float(np.mean(delta[rng.integers(0, len(delta), len(delta))])) for _ in range(2000)]
    first_only = int(np.sum((a == 1) & (b == 0))); second_only = int(np.sum((a == 0) & (b == 1)))
    discordant = first_only + second_only
    return {
        "comparison": f"{first}_minus_{second}", "success_rate_difference": float(np.mean(delta)),
        "paired_state_bootstrap_95_ci": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "exact_mcnemar_p_value": float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0,
        f"{first}_only_success": first_only, f"{second}_only_success": second_only, "discordant_total": discordant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args()
    artifact = args.artifact.resolve(); protocol = __import__("yaml").safe_load((artifact / "protocol.lock.yaml").read_text())
    specs = read_jsonl(artifact / "episode_manifest.jsonl"); ids = [r["episode_id"] for r in specs]; failures = []
    dirs = {p.name for p in (artifact / "episodes").iterdir() if p.is_dir()}
    if dirs != set(ids): failures.append({"episode_id": "aggregate", "reason": "episode directory set mismatch", "missing": sorted(set(ids)-dirs), "extra": sorted(dirs-set(ids))})
    records = {}
    for spec in specs:
        path = artifact / "episodes" / spec["episode_id"] / "episode.json"
        if not path.exists(): failures.append({"episode_id": spec["episode_id"], "reason": "missing episode"}); continue
        record = json.loads(path.read_text()); records[spec["episode_id"]] = record
        if record["episode_id"] != spec["episode_id"] or record["status"] != "complete": failures.append({"episode_id": spec["episode_id"], "reason": "record identity/status"})
        if not record["all_sampler_outputs_finite"] or record["nonfinite_action_count"] != 0: failures.append({"episode_id": spec["episode_id"], "reason": "finite failure"})
        if video_frames(artifact / record["video"]) is None or len(read_jsonl(artifact / record["step_log"])) != record["control_steps"]: failures.append({"episode_id": spec["episode_id"], "reason": "video/step mismatch"})
    grouped = defaultdict(dict)
    for record in records.values(): grouped[record["pair_id"]][record["condition"]] = record
    pairs = []
    for pair_id, conditions in sorted(grouped.items()):
        if set(conditions) != set(CONDITIONS): failures.append({"episode_id": pair_id, "reason": "incomplete pair"}); continue
        ref = conditions["vanilla"]
        keys = ("suite", "task_id", "init_state_id", "reset_seed", "action_noise_seed", "selection_seed", "language")
        if any(conditions[c][k] != ref[k] for c in CONDITIONS for k in keys): failures.append({"episode_id": pair_id, "reason": "pair identity"}); continue
        if len({conditions[c]["initial_sim_state_sha256"] for c in CONDITIONS}) != 1 or len({conditions[c]["initial_prepared_input_sha256"] for c in CONDITIONS}) != 1: failures.append({"episode_id": pair_id, "reason": "state/preprocessing identity"}); continue
        m = min(conditions[c]["replans"] for c in CONDITIONS)
        if any(conditions[c]["noise_sha256_by_replan"][:m] != ref["noise_sha256_by_replan"][:m] for c in CONDITIONS): failures.append({"episode_id": pair_id, "reason": "noise identity"}); continue
        g=conditions["coreact_top8"]["replan_traces"][0]; q=conditions["top8_mask_only"]["replan_traces"][0]
        if g["selected_indices"] != q["selected_indices"] or g["prefix_sha256"] != q["prefix_sha256"] or g["negative_prefix_sha256"] != q["masked_prefix_sha256"]: failures.append({"episode_id": pair_id, "reason": "top8/prefix identity"}); continue
        row={"pair_id":pair_id,"init_state_id":ref["init_state_id"]}
        for c in CONDITIONS:
            r=conditions[c]; row.update({f"{c}_success":int(r["success"]),f"{c}_control_steps":r["control_steps"],f"{c}_action_total_variation":r["action_total_variation"],f"{c}_chunk_boundary_discontinuity":r["mean_chunk_boundary_discontinuity"],f"{c}_latency_median_per_replan":r["policy_seconds_median_per_replan"],f"{c}_nonfinite_count":r["nonfinite_action_count"],f"{c}_bound_violation_count":r["normalized_bound_violation_count"]})
        pairs.append(row)
    if len(specs)!=150 or len(records)!=150 or len(pairs)!=50 or len(set(ids))!=150: failures.append({"episode_id":"aggregate","reason":"expected 150 records and 50 pairs"})
    if failures: write_json(artifact/"analysis_integrity_failure.json",{"failures":failures,"episodes":len(records),"pairs":len(pairs)}); raise SystemExit(1)
    rates={c:float(np.mean([r[f"{c}_success"] for r in pairs])) for c in CONDITIONS}; comparisons=[paired_comparison(pairs,a,b) for a,b in COMPARISONS]; lookup={r["comparison"]:r for r in comparisons}
    patterns=Counter("".join(str(r[f"{c}_success"]) for c in CONDITIONS) for r in pairs)
    secondary={c:{m:{"mean":float(np.mean([r[f"{c}_{m}"] for r in pairs])),"median":float(np.median([r[f"{c}_{m}"] for r in pairs]))} for m in ("control_steps","action_total_variation","chunk_boundary_discontinuity","latency_median_per_replan","nonfinite_count","bound_violation_count")} for c in CONDITIONS}
    md=lookup["top8_mask_only_minus_vanilla"]; gd=lookup["coreact_top8_minus_vanilla"]
    answers={"top8_mask_only_lowers_success":md["success_rate_difference"]<0,"top8_mask_only_clearly_lowers_success":md["paired_state_bootstrap_95_ci"][1]<0,"coreact_guidance_worse_than_vanilla":gd["success_rate_difference"]<0,"unreliable_negative_branch_pattern_supported":md["success_rate_difference"]>=0 and gd["success_rate_difference"]<0,"necessary_information_pattern_supported":md["paired_state_bootstrap_95_ci"][1]<0}
    status="TASK_MASK_TOP8_DAMAGING" if answers["top8_mask_only_clearly_lowers_success"] else ("TASK_GUIDANCE_WORSE_MASK_NOT_WORSE" if answers["unreliable_negative_branch_pattern_supported"] else "TASK_MIXED_OR_NULL_DEVELOPMENT_RESULT")
    summary={"stage":"task_level_development_replication","task_id":protocol["task_id"],"task":protocol["task"],"episodes":150,"complete_pairs":50,"missingness_rate":0.0,"success_rates":rates,"comparisons":comparisons,"all_outcome_patterns":dict(sorted(patterns.items())),"secondary_metrics":secondary,"direct_answers":answers,"decision":status,"integrity_failures":[]}
    write_json(artifact/"analysis_summary.json",summary); write_json(artifact/"decision.json",{"status":status,"direct_answers":answers})
    with (artifact/"paired_results.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    with (artifact/"missingness.csv").open("w",newline="") as f: csv.DictWriter(f,fieldnames=("episode_id","reason")).writeheader()
    (artifact/"report.md").write_text(f"# LIBERO-Spatial Task {protocol['task_id']} Top-8 Development Replication\n\nTask: {protocol['task']}\n\n## Integrity\n\n- Episodes: 150/150\n- Complete pairs: 50/50\n- Missing or integrity failures: 0\n\n## Success\n\n- Vanilla: {rates['vanilla']:.1%}\n- CoreAct top-8: {rates['coreact_top8']:.1%}\n- Top-8 mask-only: {rates['top8_mask_only']:.1%}\n\n## Comparisons\n\n```json\n{json.dumps(comparisons,indent=2,sort_keys=True)}\n```\n\nDecision: `{status}`. This is development evidence, not independent confirmation.\n")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__ == "__main__": main()
