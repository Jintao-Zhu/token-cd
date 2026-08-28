#!/usr/bin/env python3
"""Audit and analyze the locked paired closed-loop confirmation experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from research.coreact_closed_loop.audit_qualification import read_jsonl, video_frames


CONDITIONS = ("vanilla", "coreact_top8", "random8", "bottom8")
BOOTSTRAP_SEED = 8_675_309
BOOTSTRAP_REPLICATES = 2000


def cluster_bootstrap(pairs: list[dict], left: str, right: str) -> tuple[float, list[float]]:
    values = np.asarray([float(row[left]) - float(row[right]) for row in pairs])
    point = float(values.mean())
    by_task = defaultdict(list)
    for row in pairs:
        by_task[(row["suite"], row["task_id"])].append(row)
    tasks = sorted(by_task)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_tasks = rng.integers(0, len(tasks), len(tasks))
        sampled_values = []
        for task_index in sampled_tasks:
            task_rows = by_task[tasks[task_index]]
            sampled_pairs = rng.integers(0, len(task_rows), len(task_rows))
            sampled_values.extend(
                float(task_rows[index][left]) - float(task_rows[index][right])
                for index in sampled_pairs
            )
        estimates.append(float(np.mean(sampled_values)))
    return point, np.quantile(estimates, [0.025, 0.975]).tolist()


def mcnemar(left: np.ndarray, right: np.ndarray) -> dict:
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    p_value = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {"left_only": left_only, "right_only": right_only, "p_value": p_value}


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_value"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * rows[index]["p_value"]))
        rows[index]["holm_adjusted_p"] = running


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    expected_rows = [
        row for row in read_jsonl(artifact / "episode_manifest.jsonl") if row["split"] == "confirmation"
    ]
    expected = {row["episode_id"]: row for row in expected_rows}
    failures = []
    records = []
    for episode_id, spec in sorted(expected.items()):
        episode_dir = artifact / "episodes" / episode_id
        record_path = episode_dir / "episode.json"
        if not record_path.exists():
            failures.append({"episode_id": episode_id, "reason": "missing episode"})
            continue
        record = json.loads(record_path.read_text())
        video_path = artifact / record["video"]
        step_path = artifact / record["step_log"]
        if not video_path.exists() or video_frames(video_path) is None:
            failures.append({"episode_id": episode_id, "reason": "missing or undecodable video"})
        if not step_path.exists() or len(read_jsonl(step_path)) != record["control_steps"]:
            failures.append({"episode_id": episode_id, "reason": "step-log mismatch"})
        if record["nonfinite_action_count"] or not record["all_guidance_outputs_finite"]:
            failures.append({"episode_id": episode_id, "reason": "nonfinite output"})
        for key in ("pair_id", "suite", "task_id", "init_state_id", "condition", "language"):
            if record.get(key) != spec.get(key):
                failures.append({"episode_id": episode_id, "reason": f"identity mismatch: {key}"})
        records.append(record)

    pair_map = defaultdict(dict)
    for record in records:
        pair_map[record["pair_id"]][record["condition"]] = record
    complete_pairs = []
    for pair_id, conditions in sorted(pair_map.items()):
        if set(conditions) != set(CONDITIONS):
            failures.append({"pair_id": pair_id, "reason": "incomplete condition set"})
            continue
        reference = conditions["vanilla"]
        if any(
            conditions[condition][key] != reference[key]
            for condition in CONDITIONS
            for key in ("suite", "task_id", "init_state_id", "reset_seed", "action_noise_seed")
        ):
            failures.append({"pair_id": pair_id, "reason": "paired-control mismatch"})
            continue
        complete_pairs.append(
            {
                "pair_id": pair_id,
                "suite": reference["suite"],
                "task_id": reference["task_id"],
                "init_state_id": reference["init_state_id"],
                **{condition: int(conditions[condition]["success"]) for condition in CONDITIONS},
                **{
                    f"{condition}_steps": conditions[condition]["control_steps"]
                    for condition in CONDITIONS
                },
                **{
                    f"{condition}_tv": conditions[condition]["action_total_variation"]
                    for condition in CONDITIONS
                },
                **{
                    f"{condition}_latency": conditions[condition]["policy_seconds_median_per_replan"]
                    for condition in CONDITIONS
                },
            }
        )

    expected_count = len(expected_rows)
    missing_rate = (expected_count - len(records)) / expected_count
    comparisons = []
    for right in ("vanilla", "random8", "bottom8"):
        point, interval = cluster_bootstrap(complete_pairs, "coreact_top8", right)
        left_values = np.asarray([row["coreact_top8"] for row in complete_pairs], dtype=bool)
        right_values = np.asarray([row[right] for row in complete_pairs], dtype=bool)
        test = mcnemar(left_values, right_values)
        comparisons.append(
            {
                "comparison": f"coreact_top8_minus_{right}",
                "success_rate_difference": point,
                "cluster_bootstrap_95_ci": interval,
                **test,
            }
        )
    holm(comparisons)
    primary = comparisons[0]
    suite_results = []
    for suite in ("libero_spatial", "libero_object"):
        rows = [row for row in complete_pairs if row["suite"] == suite]
        suite_results.append(
            {
                "suite": suite,
                "pairs": len(rows),
                "vanilla_success_rate": float(np.mean([row["vanilla"] for row in rows])),
                "coreact_success_rate": float(np.mean([row["coreact_top8"] for row in rows])),
                "difference": float(np.mean([row["coreact_top8"] - row["vanilla"] for row in rows])),
            }
        )
    rates = {
        condition: float(np.mean([row[condition] for row in complete_pairs])) for condition in CONDITIONS
    }
    nonfinite_by_condition = {
        condition: sum(record["nonfinite_action_count"] for record in records if record["condition"] == condition)
        for condition in CONDITIONS
    }
    integrity_pass = (
        json.loads((artifact / "qualification_gate.json").read_text())["pass"]
        and json.loads((artifact / "integrity_report.json").read_text())["pass"]
        and json.loads((artifact / "development_gate.json").read_text())["pass"]
        and not failures
    )
    gates = {
        "mandatory_integrity": integrity_pass,
        "missingness_le_5_percent": missing_rate <= 0.05,
        "coreact_minus_vanilla_point_gt_zero": primary["success_rate_difference"] > 0,
        "coreact_minus_vanilla_ci_lower_gt_zero": primary["cluster_bootstrap_95_ci"][0] > 0,
        "coreact_minus_random_ci_lower_gt_zero": comparisons[1]["cluster_bootstrap_95_ci"][0] > 0,
        "both_suites_nonnegative": all(row["difference"] >= 0 for row in suite_results),
        "no_nonfinite_increase": nonfinite_by_condition["coreact_top8"]
        <= nonfinite_by_condition["vanilla"],
    }
    if not integrity_pass:
        status = "FAILED_INTEGRITY"
    elif missing_rate > 0.05:
        status = "INCONCLUSIVE_DATA_OR_RESOURCES"
    elif all(gates.values()):
        status = "PROCEED_TO_CONFIRMATORY_CLOSED_LOOP"
    elif primary["cluster_bootstrap_95_ci"][1] < 0:
        status = "STOP_CLOSED_LOOP_GUIDANCE_HYPOTHESIS"
    else:
        status = "REVISE_GUIDANCE_AND_REPEAT_PILOT"
    summary = {
        "status": status,
        "expected_episodes": expected_count,
        "actual_episodes": len(records),
        "expected_pairs": 200,
        "complete_pairs": len(complete_pairs),
        "missing_rate": missing_rate,
        "success_rates": rates,
        "comparisons": comparisons,
        "suite_results": suite_results,
        "nonfinite_by_condition": nonfinite_by_condition,
        "gates": gates,
        "integrity_failures": failures,
        "bootstrap_unit": "suite/task then paired initial state",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (artifact / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"status": status, "gates": gates}, indent=2) + "\n")
    with (artifact / "paired_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(complete_pairs[0]))
        writer.writeheader()
        writer.writerows(complete_pairs)
    with (artifact / "missingness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode_id", "reason"), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failures)

    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.bar(CONDITIONS, [rates[value] for value in CONDITIONS], color=["#555555", "#1976d2", "#d97706", "#6b7280"])
    plt.ylabel("Task success rate")
    plt.ylim(0, 1)
    plt.title(f"Confirmation success (n={len(complete_pairs)} paired states)")
    plt.tight_layout()
    plt.savefig(figures / "success_rates.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 4))
    labels = [row["comparison"].removeprefix("coreact_top8_minus_") for row in comparisons]
    points = [row["success_rate_difference"] for row in comparisons]
    lower = [point - row["cluster_bootstrap_95_ci"][0] for point, row in zip(points, comparisons, strict=True)]
    upper = [row["cluster_bootstrap_95_ci"][1] - point for point, row in zip(points, comparisons, strict=True)]
    plt.errorbar(labels, points, yerr=[lower, upper], fmt="o", capsize=4, color="#1976d2")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Paired success-rate difference")
    plt.title("95% task/state cluster bootstrap CI (2,000 replicates)")
    plt.tight_layout()
    plt.savefig(figures / "paired_success_differences.png", dpi=160)
    plt.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
