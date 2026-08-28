from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import TASKS, read_jsonl, write_json


def finite_median(values):
    values = np.asarray(values, dtype=float)
    return float(np.median(values[np.isfinite(values)]))


def bootstrap_difference(rows, repetitions=10000, seed=20260814):
    differences = np.asarray([row["object_pixel_cosine"] - row["random_pixel_cosine"] for row in rows])
    rng = np.random.default_rng(seed)
    draws = np.median(differences[rng.integers(0, len(differences), size=(repetitions, len(differences)))], axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.artifact / "results.jsonl")
    if len(rows) != 270 or len({row["state_id"] for row in rows}) != 270:
        raise RuntimeError(f"Expected 270 unique results, got {len(rows)}")
    object_cos = finite_median([row["object_pixel_cosine"] for row in rows])
    random_cos = finite_median([row["random_pixel_cosine"] for row in rows])
    difference = finite_median([row["object_pixel_cosine"] - row["random_pixel_cosine"] for row in rows])
    norm_ratio = finite_median([row["object_pixel_norm_ratio"] for row in rows])
    ci = bootstrap_difference(rows)
    task_results = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        task_results[task] = {"n": len(selected),
                              "median_object_pixel_cosine": finite_median([row["object_pixel_cosine"] for row in selected]),
                              "median_random_pixel_cosine": finite_median([row["random_pixel_cosine"] for row in selected])}
    tasks_better = sum(value["median_object_pixel_cosine"] > value["median_random_pixel_cosine"] for value in task_results.values())
    checks = {"median_object_pixel_cosine_ge_0_50": object_cos >= 0.50,
              "median_object_minus_random_ge_0_20": difference >= 0.20,
              "bootstrap_ci_lower_gt_zero": ci[0] > 0,
              "at_least_7_tasks_object_gt_random": tasks_better >= 7,
              "effect_norm_ratio_in_range": 0.5 <= norm_ratio <= 1.5}
    no_go = object_cos < 0.2 or abs(difference) < 0.05
    status = "STRONG_GO" if all(checks.values()) else ("NO_GO" if no_go else "INCONCLUSIVE")
    report = {"status": status, "n": len(rows), "primary_threshold": 0.25,
              "median_object_pixel_cosine": object_cos, "median_random_pixel_cosine": random_cos,
              "median_object_minus_random_cosine": difference, "paired_bootstrap_difference_ci95": ci,
              "median_object_pixel_effect_norm_ratio": norm_ratio, "tasks_object_gt_random": tasks_better,
              "pixel_token_pcd_top1_position_agreement": float(np.mean([row["pcd_position_agreement"] for row in rows])),
              "pixel_token_pcd_full_sequence_agreement": float(np.mean([row["pcd_full_sequence_match"] for row in rows])),
              "sensitivity_50": {
                  "median_object_pixel_cosine": finite_median([row["object_pixel_cosine_50"] for row in rows]),
                  "median_random_pixel_cosine": finite_median([row["random_pixel_cosine_50"] for row in rows]),
                  "median_object_minus_random_cosine": finite_median([row["object_pixel_cosine_50"] - row["random_pixel_cosine_50"] for row in rows]),
                  "median_object_pixel_effect_norm_ratio": finite_median([row["object_pixel_norm_ratio_50"] for row in rows])},
              "checks": checks, "task_results": task_results}
    write_json(args.artifact / "stage_a_decision.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
