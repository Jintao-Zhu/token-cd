from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

from research.token_pcd_stage_a.core import TASKS, write_json


ARMS = ("vanilla", "pixel_pcd", "object_token_pcd", "random_token_pcd")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100); parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args(); seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    rows = []
    for task in TASKS:
        for arm in ARMS:
            for seed in seeds:
                path = args.artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                if not path.is_file(): raise FileNotFoundError(path)
                rows.append(json.loads(path.read_text()))
        pairing = json.loads((args.artifact / "episodes" / task / "pairing_manifest.json").read_text())
        if not pairing["all_four_arm_exact_pairing"]: raise RuntimeError(f"Pairing failure {task}")
    expected = len(TASKS) * len(seeds) * len(ARMS)
    if len(rows) != expected: raise RuntimeError(len(rows))
    pooled = {arm: sum(row["success"] for row in rows if row["arm"] == arm) for arm in ARMS}
    denominator = len(TASKS) * len(seeds)
    rates = {arm: pooled[arm] / denominator for arm in ARMS}
    def paired_ci(task, left, right, draws=20000):
        left_rows = {row["seed"]: int(row["success"]) for row in rows if row["task"] == task and row["arm"] == left}
        right_rows = {row["seed"]: int(row["success"]) for row in rows if row["task"] == task and row["arm"] == right}
        differences = np.asarray([left_rows[seed] - right_rows[seed] for seed in seeds], dtype=float)
        rng = np.random.default_rng(20260814)
        boot = differences[rng.integers(0, len(differences), size=(draws, len(differences)))].mean(axis=1) * 100
        return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    tasks = {}
    for task in TASKS:
        values = {arm: sum(row["success"] for row in rows if row["task"] == task and row["arm"] == arm) for arm in ARMS}
        tasks[task] = {**values, "object_minus_vanilla_pp": 100.0 * (values["object_token_pcd"] - values["vanilla"]) / len(seeds),
                       "object_minus_random_pp": 100.0 * (values["object_token_pcd"] - values["random_token_pcd"]) / len(seeds),
                       "object_minus_vanilla_paired_ci_pp": paired_ci(task, "object_token_pcd", "vanilla"),
                       "object_minus_random_paired_ci_pp": paired_ci(task, "object_token_pcd", "random_token_pcd")}
    object_vanilla_pp = 100 * (rates["object_token_pcd"] - rates["vanilla"])
    object_random_pp = 100 * (rates["object_token_pcd"] - rates["random_token_pcd"])
    pixel_vanilla_pp = 100 * (rates["pixel_pcd"] - rates["vanilla"])
    pooled_diffs = np.asarray([int(row["success"]) for row in rows if row["arm"] == "object_token_pcd"]) - np.asarray([int(row["success"]) for row in rows if row["arm"] == "vanilla"])
    rng = np.random.default_rng(20260814)
    pooled_boot = pooled_diffs[rng.integers(0, len(pooled_diffs), size=(20000, len(pooled_diffs)))].mean(axis=1) * 100
    object_vanilla_ci = [float(np.percentile(pooled_boot, 2.5)), float(np.percentile(pooled_boot, 97.5))]
    nonnegative = sum(value["object_minus_vanilla_pp"] >= 0 for value in tasks.values())
    catastrophic = [task for task, value in tasks.items() if value["object_minus_vanilla_pp"] <= -10 and value["object_minus_vanilla_paired_ci_pp"][1] < 0]
    checks = {"object_minus_vanilla_ge_5pp": object_vanilla_pp >= 5,
              "object_minus_random_ge_3pp": object_random_pp >= 3,
              "object_minus_vanilla_paired_ci_lower_gt_zero": object_vanilla_ci[0] > 0,
              "at_least_7_tasks_nonnegative": nonnegative >= 7,
              "no_task_catastrophic_harm_gt_10pp": not catastrophic,
              "pixel_positive_control_direction_normal": pixel_vanilla_pp > 0}
    if not checks["pixel_positive_control_direction_normal"]:
        status = "REFERENCE_PCD_NOT_DIRECTIONALLY_POSITIVE_INCONCLUSIVE"
    elif all(checks.values()): status = "STRONG_GO"
    elif object_vanilla_pp < 3 or object_random_pp < 3: status = "NO_GO"
    else: status = "INCONCLUSIVE"
    report = {"status": status, "episodes": expected, "successes": pooled, "rates": rates,
              "object_minus_vanilla_pp": object_vanilla_pp, "object_minus_random_pp": object_random_pp,
              "object_minus_vanilla_paired_ci_pp": object_vanilla_ci,
              "pixel_minus_vanilla_pp": pixel_vanilla_pp, "tasks_object_nonnegative": nonnegative,
              "catastrophic_tasks": catastrophic, "checks": checks, "task_results": tasks}
    write_json(args.artifact / "bridge_decision.json", report); print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
