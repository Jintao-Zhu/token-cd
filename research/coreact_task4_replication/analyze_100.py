#!/usr/bin/env python3
"""Cluster-aware paired analysis for the 100-pair task-4 extension."""

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


def cluster_bootstrap_difference(
    rows: list[dict], first: str, second: str, *, replicates: int = 2000, seed: int = 44_004_104
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["init_state_id"])].append(
            float(row[f"{first}_success"]) - float(row[f"{second}_success"])
        )
    cluster_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        estimates.append(float(np.mean([value for cluster in sampled for value in grouped[int(cluster)]])))
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def cluster_sign_permutation_p_value(
    rows: list[dict], first: str, second: str, *, permutations: int = 200_000, seed: int = 44_004_105
) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["init_state_id"])].append(
            float(row[f"{first}_success"]) - float(row[f"{second}_success"])
        )
    cluster_deltas = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
    observed = abs(float(np.mean(cluster_deltas)))
    rng = np.random.default_rng(seed)
    exceed = 0
    batch_size = 10_000
    completed = 0
    while completed < permutations:
        count = min(batch_size, permutations - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(cluster_deltas)))
        exceed += int(np.sum(np.abs(np.mean(signs * cluster_deltas, axis=1)) >= observed - 1e-12))
        completed += count
    return float((exceed + 1) / (permutations + 1))


def paired_comparison(rows: list[dict], first: str, second: str) -> dict:
    a = np.asarray([row[f"{first}_success"] for row in rows], dtype=float)
    b = np.asarray([row[f"{second}_success"] for row in rows], dtype=float)
    first_only = int(np.sum((a == 1) & (b == 0)))
    second_only = int(np.sum((a == 0) & (b == 1)))
    discordant = first_only + second_only
    return {
        "comparison": f"{first}_minus_{second}",
        "success_rate_difference": float(np.mean(a - b)),
        "init_state_cluster_bootstrap_95_ci": cluster_bootstrap_difference(rows, first, second),
        "init_state_cluster_sign_permutation_p_value": cluster_sign_permutation_p_value(rows, first, second),
        "descriptive_exact_mcnemar_p_value": (
            float(binomtest(first_only, discordant, 0.5, alternative="two-sided").pvalue)
            if discordant
            else 1.0
        ),
        f"{first}_only_success": first_only,
        f"{second}_only_success": second_only,
        "discordant_total": discordant,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    specs = read_jsonl(artifact / "episode_manifest.jsonl")
    amendment_path = artifact / "episode_manifest.amendment.jsonl"
    if amendment_path.exists():
        specs.extend(read_jsonl(amendment_path))
    amendment = json.loads((artifact / "integrity_amendment.json").read_text()) if (artifact / "integrity_amendment.json").exists() else {}
    superseded_pair_ids = set(amendment.get("superseded_pair_ids", []))
    manifest_ids = [row["episode_id"] for row in specs]
    failures = []
    duplicate_specs = [key for key, count in Counter(manifest_ids).items() if count != 1]
    episode_root = artifact / "episodes"
    actual_dirs = {path.name for path in episode_root.iterdir() if path.is_dir()} if episode_root.exists() else set()
    if actual_dirs != set(manifest_ids):
        failures.append(
            {
                "episode_id": "aggregate",
                "reason": "episode directory set mismatch",
                "missing": sorted(set(manifest_ids) - actual_dirs),
                "extra": sorted(actual_dirs - set(manifest_ids)),
            }
        )

    records = {}
    for spec in specs:
        path = episode_root / spec["episode_id"] / "episode.json"
        if not path.exists():
            failures.append({"episode_id": spec["episode_id"], "reason": "missing episode"})
            continue
        record = json.loads(path.read_text())
        if record["episode_id"] != spec["episode_id"] or record["status"] != "complete":
            failures.append({"episode_id": spec["episode_id"], "reason": "record identity or status failure"})
        if not record["all_sampler_outputs_finite"] or record["nonfinite_action_count"] != 0:
            failures.append({"episode_id": spec["episode_id"], "reason": "finite failure"})
        if video_frames(artifact / record["video"]) is None:
            failures.append({"episode_id": spec["episode_id"], "reason": "unreadable video"})
        if len(read_jsonl(artifact / record["step_log"])) != record["control_steps"]:
            failures.append({"episode_id": spec["episode_id"], "reason": "step log mismatch"})
        records[spec["episode_id"]] = record

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records.values():
        if record["pair_id"] in superseded_pair_ids:
            continue
        grouped[record["pair_id"]][record["condition"]] = record
    pairs = []
    noise_by_init_replicate: dict[tuple[int, int], str] = {}
    for pair_id, conditions in sorted(grouped.items()):
        if set(conditions) != set(CONDITIONS):
            failures.append({"episode_id": pair_id, "reason": "incomplete three-condition pair"})
            continue
        reference = conditions["vanilla"]
        identity_keys = (
            "suite", "task_id", "init_state_id", "replicate_id", "reset_seed",
            "action_noise_seed", "selection_seed", "language",
        )
        if any(conditions[c][key] != reference[key] for c in CONDITIONS for key in identity_keys):
            failures.append({"episode_id": pair_id, "reason": "pair identity mismatch"})
            continue
        if len({conditions[c]["initial_sim_state_sha256"] for c in CONDITIONS}) != 1:
            failures.append({"episode_id": pair_id, "reason": "initial simulator state mismatch"})
            continue
        if len({conditions[c]["initial_prepared_input_sha256"] for c in CONDITIONS}) != 1:
            failures.append({"episode_id": pair_id, "reason": "initial preprocessing mismatch"})
            continue
        minimum_replans = min(conditions[c]["replans"] for c in CONDITIONS)
        if any(
            conditions[c]["noise_sha256_by_replan"][:minimum_replans]
            != reference["noise_sha256_by_replan"][:minimum_replans]
            for c in CONDITIONS
        ):
            failures.append({"episode_id": pair_id, "reason": "per-replan noise mismatch"})
            continue
        guided = conditions["coreact_top8"]["replan_traces"][0]
        masked = conditions["top8_mask_only"]["replan_traces"][0]
        if (
            guided["selected_indices"] != masked["selected_indices"]
            or guided["prefix_sha256"] != masked["prefix_sha256"]
            or guided["negative_prefix_sha256"] != masked["masked_prefix_sha256"]
        ):
            failures.append({"episode_id": pair_id, "reason": "initial top8 or prefix mismatch"})
            continue
        init_id = int(reference["init_state_id"])
        replicate_id = int(reference["replicate_id"])
        noise_by_init_replicate[(init_id, replicate_id)] = reference["noise_sha256_by_replan"][0]
        row = {"pair_id": pair_id, "init_state_id": init_id, "replicate_id": replicate_id}
        for condition in CONDITIONS:
            record = conditions[condition]
            row.update(
                {
                    f"{condition}_success": int(record["success"]),
                    f"{condition}_control_steps": record["control_steps"],
                    f"{condition}_action_total_variation": record["action_total_variation"],
                    f"{condition}_chunk_boundary_discontinuity": record["mean_chunk_boundary_discontinuity"],
                    f"{condition}_latency_median_per_replan": record["policy_seconds_median_per_replan"],
                    f"{condition}_nonfinite_count": record["nonfinite_action_count"],
                    f"{condition}_bound_violation_count": record["normalized_bound_violation_count"],
                }
            )
        pairs.append(row)

    if duplicate_specs:
        failures.extend({"episode_id": value, "reason": "duplicate manifest id"} for value in duplicate_specs)
    expected_units = {(init_id, replicate_id) for init_id in range(50) for replicate_id in range(2)}
    actual_units = {(int(row["init_state_id"]), int(row["replicate_id"])) for row in pairs}
    if len(records) != len(specs) or len(pairs) != 100 or actual_units != expected_units:
        failures.append({"episode_id": "aggregate", "reason": "expected all manifest records and 100 analysis pairs"})
    for init_id in range(50):
        if noise_by_init_replicate.get((init_id, 0)) == noise_by_init_replicate.get((init_id, 1)):
            failures.append({"episode_id": f"init{init_id:02d}", "reason": "replicate noise is not distinct"})
    if failures:
        write_json(
            artifact / "analysis_integrity_failure.json",
            {"failures": failures, "episodes": len(records), "pairs": len(pairs)},
        )
        raise SystemExit(1)

    rates = {
        condition: float(np.mean([row[f"{condition}_success"] for row in pairs]))
        for condition in CONDITIONS
    }
    comparisons = [paired_comparison(pairs, first, second) for first, second in COMPARISONS]
    lookup = {row["comparison"]: row for row in comparisons}
    patterns = Counter("".join(str(row[f"{condition}_success"]) for condition in CONDITIONS) for row in pairs)
    secondary = {
        condition: {
            metric: {
                "mean": float(np.mean([row[f"{condition}_{metric}"] for row in pairs])),
                "median": float(np.median([row[f"{condition}_{metric}"] for row in pairs])),
            }
            for metric in (
                "control_steps", "action_total_variation", "chunk_boundary_discontinuity",
                "latency_median_per_replan", "nonfinite_count", "bound_violation_count",
            )
        }
        for condition in CONDITIONS
    }
    mask_delta = lookup["top8_mask_only_minus_vanilla"]
    guidance_delta = lookup["coreact_top8_minus_vanilla"]
    mask_clearly_worse = mask_delta["init_state_cluster_bootstrap_95_ci"][1] < 0
    answers = {
        "top8_mask_only_lowers_success": mask_delta["success_rate_difference"] < 0,
        "top8_mask_only_clearly_lowers_success": mask_clearly_worse,
        "coreact_guidance_worse_than_vanilla": guidance_delta["success_rate_difference"] < 0,
        "unreliable_negative_branch_pattern_supported": bool(
            mask_delta["success_rate_difference"] >= 0
            and guidance_delta["success_rate_difference"] < 0
        ),
        "necessary_information_pattern_supported": bool(mask_clearly_worse),
    }
    status = (
        "TASK4_MASK_TOP8_DAMAGING"
        if mask_clearly_worse
        else (
            "TASK4_GUIDANCE_WORSE_MASK_NOT_WORSE"
            if answers["unreliable_negative_branch_pattern_supported"]
            else "TASK4_MIXED_OR_NULL_DEVELOPMENT_RESULT"
        )
    )
    summary = {
        "stage": "extended_development_replication_100_pairs_clustered_in_50_init_states",
        "episodes": len(records),
        "superseded_integrity_records": len(superseded_pair_ids) * len(CONDITIONS),
        "analysis_episodes": len(pairs) * len(CONDITIONS),
        "complete_pairs": len(pairs),
        "unique_init_states": 50,
        "noise_replicates_per_init_state": 2,
        "missingness_rate": 0.0,
        "success_rates": rates,
        "comparisons": comparisons,
        "three_way_exclusive_success_counts": {
            "vanilla_only_100": patterns["100"],
            "coreact_only_010": patterns["010"],
            "mask_only_001": patterns["001"],
        },
        "all_outcome_patterns": dict(sorted(patterns.items())),
        "secondary_metrics": secondary,
        "direct_answers": answers,
        "decision": status,
        "integrity_failures": [],
    }
    write_json(artifact / "analysis_summary.json", summary)
    write_json(
        artifact / "decision.json",
        {"status": status, "stage": summary["stage"], "direct_answers": answers},
    )
    with (artifact / "paired_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    with (artifact / "missingness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode_id", "reason"))
        writer.writeheader()
    report = f"""# LIBERO-Spatial Task 4 Top-8 Extended Development Replication

## Integrity

- Episodes: 300/300
- Complete paired rollout units: 100/100
- Unique fixed simulator init states: 50
- Noise replicates per init state: 2
- Missing, duplicate, nonfinite, or pair-identity failures: 0

## Success

- Vanilla: {rates['vanilla']:.1%}
- CoreAct top-8: {rates['coreact_top8']:.1%}
- Top-8 mask-only: {rates['top8_mask_only']:.1%}

## Paired Comparisons

```json
{json.dumps(comparisons, indent=2, sort_keys=True)}
```

Confidence intervals resample the 50 init-state clusters. Cluster sign-permutation p-values are the primary paired significance diagnostic. Exact McNemar values are descriptive only because the two replicates within each init state are dependent.

## Direct Answers

1. Top-8 mask-only lowers success (point estimate): {answers['top8_mask_only_lowers_success']}; clearly lower by clustered CI: {answers['top8_mask_only_clearly_lowers_success']}.
2. CoreAct guidance is worse than vanilla (point estimate): {answers['coreact_guidance_worse_than_vanilla']}.
3. Mask not worse while guidance worse supports an unreliable masked negative branch: {answers['unreliable_negative_branch_pattern_supported']}.
4. Clearly damaging mask-only supports top-attention tokens containing necessary information: {answers['necessary_information_pattern_supported']}.

Decision: `{status}`. This remains development evidence: the v1 outcomes were known and the 100 pairs contain only 50 unique simulator states.
"""
    (artifact / "report.md").write_text(report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
