#!/usr/bin/env python3
"""Analyze the completed paired consistency-guidance development pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--artifact", type=Path, required=True); args = p.parse_args(); artifact = args.artifact.resolve()
    manifest = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines() if line.strip()]
    if len(manifest) % 2 or len({row["episode_id"] for row in manifest}) != len(manifest): raise RuntimeError("manifest must contain an even number of unique episodes")
    records = []
    for spec in manifest:
        path = artifact / "episodes" / spec["episode_id"] / "episode.json"
        if not path.exists(): raise RuntimeError(f"missing {path}")
        row = json.loads(path.read_text())
        for key in ("pair_id", "condition", "init_state_id", "reset_seed", "action_noise_seed", "language"):
            if row[key] != spec[key]: raise RuntimeError(f"paired identity mismatch {spec['episode_id']} {key}")
        records.append(row)
    by_pair = {}
    integrity_failures = []
    for row in records:
        by_pair.setdefault(row["pair_id"], {})[row["condition"]] = row
        if row["condition"] == "consistency_guided":
            for trace in row["replan_traces"]:
                if sorted(trace["selected_indices"]) != sorted(trace["changed_indices"]): integrity_failures.append([row["episode_id"], trace["replan"], "selected_changed"])
                if not trace["protected_tokens_untouched"] or not trace["all_output_finite"]: integrity_failures.append([row["episode_id"], trace["replan"], "protected_or_finite"])
    expected_pairs = len(manifest) // 2
    if len(by_pair) != expected_pairs or any(set(pair) != {"vanilla", "consistency_guided"} for pair in by_pair.values()): raise RuntimeError(f"expected {expected_pairs} complete pairs")
    paired = []
    for pair_id in sorted(by_pair):
        v, g = by_pair[pair_id]["vanilla"], by_pair[pair_id]["consistency_guided"]
        paired.append({"pair_id": pair_id, "init_state_id": v["init_state_id"], "vanilla_success": int(v["success"]), "consistency_guided_success": int(g["success"]), "success_difference": int(g["success"]) - int(v["success"]), "vanilla_steps": v["control_steps"], "guided_steps": g["control_steps"], "vanilla_action_tv": v["action_total_variation"], "guided_action_tv": g["action_total_variation"], "vanilla_latency_s_per_replan": v["policy_seconds_median_per_replan"], "guided_latency_s_per_replan": g["policy_seconds_median_per_replan"]})
    with (artifact / "paired_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    diffs = np.asarray([row["success_difference"] for row in paired], dtype=float); rng = np.random.default_rng(8675310); boot = np.asarray([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)])
    v_success = sum(row["vanilla_success"] for row in paired); g_success = sum(row["consistency_guided_success"] for row in paired)
    vanilla_only = sum(row["vanilla_success"] == 1 and row["consistency_guided_success"] == 0 for row in paired); guided_only = sum(row["vanilla_success"] == 0 and row["consistency_guided_success"] == 1 for row in paired); discordant = vanilla_only + guided_only
    mcnemar = float(binomtest(guided_only, discordant, .5).pvalue) if discordant else 1.0
    passed = not integrity_failures and len(records) == len(manifest)
    if not passed: decision = "FAILED_INTEGRITY"
    elif g_success > v_success: decision = "PROCEED_TO_BROADER_DEVELOPMENT"
    elif g_success < v_success: decision = "DO_NOT_SCALE_CURRENT_CONSISTENCY_GUIDANCE"
    else: decision = "INCONCLUSIVE_SMALL_PILOT"
    result = {"complete_episodes": len(records), "complete_pairs": len(paired), "integrity_pass": passed, "integrity_failures": integrity_failures, "vanilla": {"successes": v_success, "episodes": expected_pairs, "success_rate": v_success / expected_pairs}, "consistency_guided": {"successes": g_success, "episodes": expected_pairs, "success_rate": g_success / expected_pairs}, "guided_minus_vanilla": {"point_estimate": float(diffs.mean()), "paired_bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], "bootstrap_replicates": 2000, "vanilla_only_success": vanilla_only, "guided_only_success": guided_only, "mcnemar_exact_p": mcnemar}, "secondary": {"vanilla_mean_steps": float(np.mean([r["vanilla_steps"] for r in paired])), "guided_mean_steps": float(np.mean([r["guided_steps"] for r in paired])), "vanilla_mean_action_tv": float(np.mean([r["vanilla_action_tv"] for r in paired])), "guided_mean_action_tv": float(np.mean([r["guided_action_tv"] for r in paired])), "vanilla_median_latency_s_per_replan": float(np.median([r["vanilla_latency_s_per_replan"] for r in paired])), "guided_median_latency_s_per_replan": float(np.median([r["guided_latency_s_per_replan"] for r in paired]))}, "decision": decision, "scope": f"{expected_pairs}-pair single-task development replication; not confirmation"}
    (artifact / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"status": decision, "selective_guidance_confirmed": False, "reason": "Five-pair single-task pilot is directional evidence only."}, indent=2, sort_keys=True) + "\n")
    report = f"""# Consistency-Guided Closed-Loop Pilot\n\nIntegrity: **{'PASS' if passed else 'FAIL'}**. Completed {len(records)}/{len(manifest)} episodes and {len(paired)}/{expected_pairs} paired init states.\n\n| Condition | Successes | Rate |\n|---|---:|---:|\n| Vanilla | {v_success}/{expected_pairs} | {v_success/expected_pairs:.3f} |\n| Consistency-guided | {g_success}/{expected_pairs} | {g_success/expected_pairs:.3f} |\n\nGuided minus vanilla: `{diffs.mean():.3f}`; paired bootstrap 95% CI `[{np.quantile(boot,.025):.3f}, {np.quantile(boot,.975):.3f}]`; exact McNemar p=`{mcnemar:.4f}`. Discordant pairs: vanilla-only `{vanilla_only}`, guided-only `{guided_only}`.\n\nMedian policy latency per replan was `{result['secondary']['vanilla_median_latency_s_per_replan']:.4f}s` vanilla and `{result['secondary']['guided_median_latency_s_per_replan']:.4f}s` guided.\n\nDecision: **{decision}**. This is a {expected_pairs}-pair, single-task development result and cannot establish general success improvement. No parameter was changed after reading outcomes.\n"""
    (artifact / "report.md").write_text(report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
