#!/usr/bin/env python3
"""Paired analysis for the locked LIBERO-Spatial task-4 replication."""

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
COMPARISONS = (
    ("coreact_top8", "vanilla"),
    ("top8_mask_only", "vanilla"),
    ("coreact_top8", "top8_mask_only"),
)


def paired_comparison(rows: list[dict], first: str, second: str) -> dict:
    a = np.asarray([row[f"{first}_success"] for row in rows], dtype=float)
    b = np.asarray([row[f"{second}_success"] for row in rows], dtype=float)
    delta = a - b
    rng = np.random.default_rng(44_004_004)
    estimates = [float(np.mean(delta[rng.integers(0, len(delta), len(delta))])) for _ in range(2000)]
    first_only = int(np.sum((a == 1) & (b == 0)))
    second_only = int(np.sum((a == 0) & (b == 1)))
    discordant = first_only + second_only
    p_value = float(binomtest(first_only, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "comparison": f"{first}_minus_{second}",
        "success_rate_difference": float(np.mean(delta)),
        "paired_bootstrap_95_ci": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "exact_mcnemar_p_value": p_value,
        f"{first}_only_success": first_only,
        f"{second}_only_success": second_only,
        "discordant_total": discordant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args(); artifact = args.artifact.resolve()
    specs = read_jsonl(artifact / "episode_manifest.jsonl")
    duplicate_specs = [key for key,count in Counter(row["episode_id"] for row in specs).items() if count != 1]
    records, failures = {}, []
    for spec in specs:
        path = artifact / "episodes" / spec["episode_id"] / "episode.json"
        if not path.exists(): failures.append({"episode_id": spec["episode_id"], "reason": "missing episode"}); continue
        record = json.loads(path.read_text())
        if record["status"] != "complete" or not record["all_sampler_outputs_finite"] or record["nonfinite_action_count"] != 0:
            failures.append({"episode_id": spec["episode_id"], "reason": "status or finite failure"})
        if video_frames(artifact / record["video"]) is None or len(read_jsonl(artifact / record["step_log"])) != record["control_steps"]:
            failures.append({"episode_id": spec["episode_id"], "reason": "video or step log mismatch"})
        records[spec["episode_id"]] = record
    grouped = defaultdict(dict)
    for record in records.values(): grouped[record["pair_id"]][record["condition"]] = record
    pairs = []
    for pair_id, conditions in sorted(grouped.items()):
        if set(conditions) != set(CONDITIONS): failures.append({"episode_id": pair_id, "reason": "incomplete three-condition pair"}); continue
        reference = conditions["vanilla"]
        if any(conditions[c][key] != reference[key] for c in CONDITIONS for key in ("suite","task_id","init_state_id","reset_seed","action_noise_seed","selection_seed","language")):
            failures.append({"episode_id": pair_id, "reason": "manifest identity mismatch"}); continue
        if len({conditions[c]["initial_sim_state_sha256"] for c in CONDITIONS}) != 1 or len({conditions[c]["initial_prepared_input_sha256"] for c in CONDITIONS}) != 1:
            failures.append({"episode_id": pair_id, "reason": "initial state or preprocessing mismatch"}); continue
        minimum_replans = min(conditions[c]["replans"] for c in CONDITIONS)
        if any(conditions[c]["noise_sha256_by_replan"][:minimum_replans] != reference["noise_sha256_by_replan"][:minimum_replans] for c in CONDITIONS):
            failures.append({"episode_id": pair_id, "reason": "per-replan noise mismatch"}); continue
        guided = conditions["coreact_top8"]["replan_traces"][0]
        masked = conditions["top8_mask_only"]["replan_traces"][0]
        if guided["selected_indices"] != masked["selected_indices"] or guided["prefix_sha256"] != masked["prefix_sha256"] or guided["negative_prefix_sha256"] != masked["masked_prefix_sha256"]:
            failures.append({"episode_id": pair_id, "reason": "initial top8 or prefix mismatch"}); continue
        row = {"pair_id": pair_id, "init_state_id": reference["init_state_id"]}
        for condition in CONDITIONS:
            record = conditions[condition]
            row.update({
                f"{condition}_success": int(record["success"]),
                f"{condition}_control_steps": record["control_steps"],
                f"{condition}_action_total_variation": record["action_total_variation"],
                f"{condition}_chunk_boundary_discontinuity": record["mean_chunk_boundary_discontinuity"],
                f"{condition}_latency_median_per_replan": record["policy_seconds_median_per_replan"],
                f"{condition}_nonfinite_count": record["nonfinite_action_count"],
                f"{condition}_bound_violation_count": record["normalized_bound_violation_count"],
            })
        pairs.append(row)
    if duplicate_specs: failures.extend({"episode_id": value, "reason": "duplicate manifest id"} for value in duplicate_specs)
    if len(pairs) != 50: failures.append({"episode_id": "aggregate", "reason": f"complete pairs {len(pairs)} != 50"})
    if failures:
        write_json(artifact / "analysis_integrity_failure.json", {"failures": failures, "episodes": len(records), "pairs": len(pairs)})
        raise SystemExit(1)
    rates = {condition: float(np.mean([row[f"{condition}_success"] for row in pairs])) for condition in CONDITIONS}
    comparisons = [paired_comparison(pairs, first, second) for first,second in COMPARISONS]
    lookup = {row["comparison"]: row for row in comparisons}
    patterns = Counter("".join(str(row[f"{condition}_success"]) for condition in CONDITIONS) for row in pairs)
    secondary = {
        condition: {
            metric: {"mean": float(np.mean([row[f"{condition}_{metric}"] for row in pairs])), "median": float(np.median([row[f"{condition}_{metric}"] for row in pairs]))}
            for metric in ("control_steps", "action_total_variation", "chunk_boundary_discontinuity", "latency_median_per_replan", "nonfinite_count", "bound_violation_count")
        } for condition in CONDITIONS
    }
    mask_delta = lookup["top8_mask_only_minus_vanilla"]
    guidance_delta = lookup["coreact_top8_minus_vanilla"]
    mask_clearly_worse = mask_delta["paired_bootstrap_95_ci"][1] < 0
    mask_not_worse_point = mask_delta["success_rate_difference"] >= 0
    guidance_worse_point = guidance_delta["success_rate_difference"] < 0
    answers = {
        "top8_mask_only_lowers_success": mask_delta["success_rate_difference"] < 0,
        "top8_mask_only_clearly_lowers_success": mask_clearly_worse,
        "coreact_guidance_worse_than_vanilla": guidance_worse_point,
        "unreliable_negative_branch_pattern_supported": bool(mask_not_worse_point and guidance_worse_point),
        "necessary_information_pattern_supported": bool(mask_clearly_worse),
    }
    status = "TASK4_MASK_TOP8_DAMAGING" if mask_clearly_worse else ("TASK4_GUIDANCE_WORSE_MASK_NOT_WORSE" if answers["unreliable_negative_branch_pattern_supported"] else "TASK4_MIXED_OR_NULL_DEVELOPMENT_RESULT")
    summary = {
        "stage": "development_replication_not_confirmation", "episodes": len(records), "complete_pairs": len(pairs),
        "missingness_rate": 0.0, "success_rates": rates, "comparisons": comparisons,
        "three_way_exclusive_success_counts": {"vanilla_only_100": patterns["100"], "coreact_only_010": patterns["010"], "mask_only_001": patterns["001"]},
        "all_outcome_patterns": dict(sorted(patterns.items())), "secondary_metrics": secondary,
        "direct_answers": answers, "decision": status, "integrity_failures": [],
    }
    write_json(artifact / "analysis_summary.json", summary); write_json(artifact / "decision.json", {"status": status, "direct_answers": answers})
    with (artifact / "paired_results.csv").open("w", newline="") as stream: writer = csv.DictWriter(stream, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    with (artifact / "missingness.csv").open("w", newline="") as stream: writer = csv.DictWriter(stream, fieldnames=("episode_id","reason")); writer.writeheader()
    report = f"""# LIBERO-Spatial Task 4 Top-8 Development Replication

## Integrity

- Episodes: 150/150
- Complete paired states: 50/50
- Missing, duplicate, nonfinite, or pair-identity failures: 0

## Success

- Vanilla: {rates['vanilla']:.1%}
- CoreAct top-8: {rates['coreact_top8']:.1%}
- Top-8 mask-only: {rates['top8_mask_only']:.1%}

## Paired Comparisons

```json
{json.dumps(comparisons, indent=2, sort_keys=True)}
```

## Discordant Outcomes

- Pairwise counts are included in each comparison above.
- Three-way vanilla-only/coreact-only/mask-only: {patterns['100']}/{patterns['010']}/{patterns['001']}

## Direct Answers

1. Top-8 mask-only lowers success (point estimate): {answers['top8_mask_only_lowers_success']}; clearly lower by paired CI: {answers['top8_mask_only_clearly_lowers_success']}.
2. CoreAct guidance is worse than vanilla (point estimate): {answers['coreact_guidance_worse_than_vanilla']}.
3. Mask not worse while guidance worse supports an unreliable masked negative branch: {answers['unreliable_negative_branch_pattern_supported']}.
4. Clearly damaging mask-only supports top-attention tokens containing necessary information: {answers['necessary_information_pattern_supported']}.

Decision: `{status}`. This is a development replication because init states overlap older pilot usage.
"""
    (artifact / "report.md").write_text(report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
