#!/usr/bin/env python3
"""Audit and analyze segmentation-grounded clean versus first-chunk mask pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.coreact_closed_loop.audit_qualification import read_jsonl, video_frames
from research.coreact_region.run_region_diagnostic import CONDITIONS


MASKS = CONDITIONS[1:]
COMPONENTS = ("composite_progress", "reach_progress", "transport_progress", "grasp_ever", "predicate_ever")


def cluster_bootstrap(rows: list[dict], condition: str, component: str) -> tuple[float, list[float]]:
    values = np.asarray([row[f"clean_{component}"] - row[f"{condition}_{component}"] for row in rows])
    tasks = sorted({(row["suite"], row["task_id"]) for row in rows})
    by_task = {task: [row for row in rows if (row["suite"], row["task_id"]) == task] for task in tasks}
    rng = np.random.default_rng(83_083_083)
    estimates = []
    for _ in range(2000):
        sample = []
        for task_index in rng.integers(0, len(tasks), len(tasks)):
            task_rows = by_task[tasks[task_index]]
            for pair_index in rng.integers(0, len(task_rows), len(task_rows)):
                row = task_rows[pair_index]
                sample.append(row[f"clean_{component}"] - row[f"{condition}_{component}"])
        estimates.append(float(np.mean(sample)))
    return float(np.mean(values)), np.quantile(estimates, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    specs = read_jsonl(artifact / "episode_manifest.jsonl")
    expected = {row["episode_id"]: row for row in specs}
    records, failures = [], []
    for episode_id, spec in sorted(expected.items()):
        path = artifact / "episodes" / episode_id / "episode.json"
        if not path.exists():
            failures.append({"episode_id": episode_id, "reason": "missing episode"}); continue
        record = json.loads(path.read_text())
        if any(record.get(key) != spec.get(key) for key in ("pair_id", "suite", "task_id", "init_state_id", "condition", "language")):
            failures.append({"episode_id": episode_id, "reason": "identity mismatch"})
        if record["control_steps"] != 30 or not record["all_outputs_finite"]:
            failures.append({"episode_id": episode_id, "reason": "step or finite mismatch"})
        if record["masked_replans"] != ([] if record["condition"] == "clean" else [0]):
            failures.append({"episode_id": episode_id, "reason": "mask schedule mismatch"})
        if video_frames(artifact / record["video"]) is None:
            failures.append({"episode_id": episode_id, "reason": "video decode failure"})
        if len(read_jsonl(artifact / record["step_log"])) != 30:
            failures.append({"episode_id": episode_id, "reason": "step log mismatch"})
        records.append(record)

    grouped = defaultdict(dict)
    for record in records:
        grouped[record["pair_id"]][record["condition"]] = record
    pairs = []
    for pair_id, conditions in sorted(grouped.items()):
        if set(conditions) != set(CONDITIONS):
            failures.append({"episode_id": pair_id, "reason": "incomplete conditions"}); continue
        reference = conditions["clean"]
        if any(conditions[c]["initial_sim_state_sha256"] != reference["initial_sim_state_sha256"] or conditions[c]["noise_sha256_by_replan"] != reference["noise_sha256_by_replan"] or conditions[c]["region_mapping_sha256"] != reference["region_mapping_sha256"] or conditions[c]["all_candidate_groups"] != reference["all_candidate_groups"] for c in CONDITIONS):
            failures.append({"episode_id": pair_id, "reason": "paired state/noise/region mismatch"}); continue
        row = {"pair_id": pair_id, "suite": reference["suite"], "task_id": reference["task_id"], "init_state_id": reference["init_state_id"], "matched_group_count": reference["matched_group_count"]}
        for condition in CONDITIONS:
            for component in COMPONENTS:
                row[f"{condition}_{component}"] = float(conditions[condition]["progress"][component])
        pairs.append(row)

    effects = []
    for condition in MASKS:
        component_results = {}
        for component in COMPONENTS:
            point, interval = cluster_bootstrap(pairs, condition, component)
            component_results[component] = {"mean_D_clean_minus_mask": point, "cluster_bootstrap_95_ci": interval}
        by_suite = {}
        for suite in sorted({row["suite"] for row in pairs}):
            subset = [row for row in pairs if row["suite"] == suite]
            by_suite[suite] = float(np.median([row["clean_composite_progress"] - row[f"{condition}_composite_progress"] for row in subset]))
        effects.append({"condition": condition, "components": component_results, "median_composite_D_by_suite": by_suite})
    lookup = {row["condition"]: row for row in effects}
    bg = lookup["background_high"]
    rel = lookup["relevant_high"]
    background_supports_dtp = bg["components"]["composite_progress"]["mean_D_clean_minus_mask"] < -0.05 and all(value < 0 for value in bg["median_composite_D_by_suite"].values())
    relevant_supports_coreact = rel["components"]["composite_progress"]["mean_D_clean_minus_mask"] > 0.05 and all(value > 0 for value in rel["median_composite_D_by_suite"].values())
    if background_supports_dtp:
        decision = "DEVELOP_DTP_NUISANCE_MASK_ONLY"
    elif relevant_supports_coreact:
        decision = "DEVELOP_REGION_GROUNDED_COREACT_NEGATIVE_BRANCH"
    else:
        decision = "ATTENTION_SIGN_UNSTABLE_COMPARE_STRONGER_ATTRIBUTION"
    summary = {
        "stage": "region_mechanism_development_not_confirmation",
        "expected_episodes": len(specs), "actual_episodes": len(records),
        "complete_pairs": len(pairs), "missing_rate": (len(specs) - len(records)) / len(specs),
        "integrity_failures": failures, "effects": effects,
        "decision": decision,
        "decision_gates": {"background_supports_dtp": background_supports_dtp, "relevant_supports_coreact": relevant_supports_coreact},
        "D_definition": "clean progress minus mask progress; positive means deletion hurt progress",
        "interpretation_limit": "Two task clusters and five states per suite; development evidence only.",
    }
    (artifact / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"status": decision, "gates": summary["decision_gates"]}, indent=2, sort_keys=True) + "\n")
    with (artifact / "paired_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    with (artifact / "missingness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode_id", "reason")); writer.writeheader(); writer.writerows(failures)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
