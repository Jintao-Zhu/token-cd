#!/usr/bin/env python3
"""Audit and analyze the paired direct-mask development diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from research.coreact_closed_loop.audit_qualification import read_jsonl, video_frames


CONDITIONS = ("vanilla", "top8_mask_only", "random8_mask_only", "bottom8_mask_only")


def bootstrap(rows: list[dict], masked: str) -> tuple[float, list[float]]:
    values = np.asarray([row["vanilla"] - row[masked] for row in rows], dtype=float)
    rng = np.random.default_rng(71_071_071)
    by_task = defaultdict(list)
    for row in rows:
        by_task[(row["suite"], row["task_id"])].append(row)
    tasks = sorted(by_task)
    estimates = []
    for _ in range(2000):
        sampled = []
        for task_index in rng.integers(0, len(tasks), len(tasks)):
            task_rows = by_task[tasks[task_index]]
            for pair_index in rng.integers(0, len(task_rows), len(task_rows)):
                row = task_rows[pair_index]
                sampled.append(row["vanilla"] - row[masked])
        estimates.append(float(np.mean(sampled)))
    return float(values.mean()), np.quantile(estimates, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    expected_rows = read_jsonl(artifact / "episode_manifest.jsonl")
    expected = {row["episode_id"]: row for row in expected_rows}
    failures, records = [], []
    for episode_id, spec in sorted(expected.items()):
        path = artifact / "episodes" / episode_id / "episode.json"
        if not path.exists():
            failures.append({"episode_id": episode_id, "reason": "missing episode"})
            continue
        record = json.loads(path.read_text())
        if any(record.get(key) != spec.get(key) for key in ("pair_id", "suite", "task_id", "init_state_id", "condition", "language")):
            failures.append({"episode_id": episode_id, "reason": "identity mismatch"})
        video = artifact / record["video"]
        steps = artifact / record["step_log"]
        if not video.exists() or video_frames(video) is None:
            failures.append({"episode_id": episode_id, "reason": "video missing or undecodable"})
        if not steps.exists() or len(read_jsonl(steps)) != record["control_steps"]:
            failures.append({"episode_id": episode_id, "reason": "step-log mismatch"})
        if record["nonfinite_action_count"] or not record["all_mask_outputs_finite"]:
            failures.append({"episode_id": episode_id, "reason": "nonfinite output"})
        records.append(record)

    grouped = defaultdict(dict)
    for record in records:
        grouped[record["pair_id"]][record["condition"]] = record
    pairs = []
    for pair_id, conditions in sorted(grouped.items()):
        if set(conditions) != set(CONDITIONS):
            failures.append({"episode_id": pair_id, "reason": "incomplete condition set"})
            continue
        reference = conditions["vanilla"]
        if any(conditions[c][key] != reference[key] for c in CONDITIONS for key in ("suite", "task_id", "init_state_id", "reset_seed", "action_noise_seed")):
            failures.append({"episode_id": pair_id, "reason": "paired-control mismatch"})
            continue
        pairs.append({
            "pair_id": pair_id, "suite": reference["suite"], "task_id": reference["task_id"],
            "init_state_id": reference["init_state_id"],
            **{c: int(conditions[c]["success"]) for c in CONDITIONS},
            **{f"{c}_steps": conditions[c]["control_steps"] for c in CONDITIONS},
            **{f"{c}_tv": conditions[c]["action_total_variation"] for c in CONDITIONS},
        })
    comparisons = []
    for condition in CONDITIONS[1:]:
        point, interval = bootstrap(pairs, condition)
        vanilla = np.asarray([row["vanilla"] for row in pairs], dtype=bool)
        masked = np.asarray([row[condition] for row in pairs], dtype=bool)
        vanilla_only = int(np.sum(vanilla & ~masked))
        masked_only = int(np.sum(~vanilla & masked))
        discordant = vanilla_only + masked_only
        comparisons.append({
            "comparison": f"vanilla_minus_{condition}", "success_damage": point,
            "cluster_bootstrap_95_ci": interval, "vanilla_only": vanilla_only,
            "masked_only": masked_only,
            "mcnemar_exact_p": float(binomtest(vanilla_only, discordant, 0.5).pvalue) if discordant else 1.0,
        })
    rates = {condition: float(np.mean([row[condition] for row in pairs])) for condition in CONDITIONS}
    summary = {
        "stage": "revision_development_only_not_confirmation",
        "expected_episodes": len(expected_rows), "actual_episodes": len(records),
        "complete_pairs": len(pairs), "missing_rate": (len(expected_rows) - len(records)) / len(expected_rows),
        "integrity_failures": failures, "success_rates": rates, "mask_damage_comparisons": comparisons,
        "directional_ordering": {
            "top_damage_gt_random": comparisons[0]["success_damage"] > comparisons[1]["success_damage"],
            "top_damage_gt_bottom": comparisons[0]["success_damage"] > comparisons[2]["success_damage"],
        },
        "interpretation_limit": "Only two task clusters; use for revision development, never as confirmation evidence.",
    }
    (artifact / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (artifact / "paired_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    with (artifact / "missingness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode_id", "reason"))
        writer.writeheader(); writer.writerows(failures)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
