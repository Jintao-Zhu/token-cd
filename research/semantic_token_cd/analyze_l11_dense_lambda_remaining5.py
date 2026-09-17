#!/usr/bin/env python3
"""Analyze the remaining-five-task 13-point L11 lambda sweep."""
from __future__ import annotations

import csv
import json
import math
import pickle

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.l11_dense_lambda_remaining5_protocol import (
    ALL_LAMBDAS,
    ARM_LAMBDAS,
    ARTIFACT,
    CANONICAL,
    SEEDS,
    TASKS,
    atomic_json,
    existing_summary,
    lambda_arm,
)


def exact_p(rescue: int, harm: int) -> float:
    count = rescue + harm
    if not count:
        return 1.0
    tail = sum(math.comb(count, index) for index in range(min(rescue, harm) + 1)) / 2**count
    return min(1.0, 2 * tail)


def summary_path(task: str, seed: int, value: float):
    if value in (0.0, 0.5):
        return existing_summary(task, seed, value)
    return ARTIFACT / "episodes" / task / lambda_arm(value) / f"episode_{seed:03d}_summary.json"


def main() -> None:
    outcomes = {task: {value: [] for value in ALL_LAMBDAS} for task in TASKS}
    paired_rows = []
    for task in TASKS:
        for seed in SEEDS:
            with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                expected = snapshot_sha(pickle.load(handle))
            paired = {"task": task, "seed": seed}
            for value in ALL_LAMBDAS:
                path = summary_path(task, seed, value)
                item = json.loads(path.read_text())
                if item.get("canonical_snapshot_sha256") != expected:
                    raise RuntimeError(f"canonical mismatch: {path}")
                if value not in (0.0, 0.5) and item.get("technical_pass") is not True:
                    raise RuntimeError(f"technical audit failure: {path}")
                success = bool(item["success"])
                outcomes[task][value].append(success)
                paired[f"lambda_{value:g}"] = int(success)
            paired_rows.append(paired)

    result = {"complete": True, "n_per_task_lambda": 100, "tasks": {}, "overall": {}}
    for task in TASKS:
        baseline = outcomes[task][0.5]
        values = {}
        for value in ALL_LAMBDAS:
            current = outcomes[task][value]
            rescue = sum(now and not base for now, base in zip(current, baseline, strict=True))
            harm = sum(base and not now for now, base in zip(current, baseline, strict=True))
            values[f"{value:g}"] = {
                "success": sum(current), "rescue_vs_0.5": rescue, "harm_vs_0.5": harm,
                "net_vs_0.5": rescue - harm, "p_exact_vs_0.5": exact_p(rescue, harm),
            }
        best = max(item["success"] for item in values.values())
        result["tasks"][task] = {
            "lambdas": values, "best_success": best,
            "best_lambdas": [float(key) for key, item in values.items() if item["success"] == best],
        }

    for value in ALL_LAMBDAS:
        current = [success for task in TASKS for success in outcomes[task][value]]
        baseline = [success for task in TASKS for success in outcomes[task][0.5]]
        rescue = sum(now and not base for now, base in zip(current, baseline, strict=True))
        harm = sum(base and not now for now, base in zip(current, baseline, strict=True))
        result["overall"][f"{value:g}"] = {
            "success": sum(current), "rescue_vs_0.5": rescue, "harm_vs_0.5": harm,
            "net_vs_0.5": rescue - harm, "p_exact_vs_0.5": exact_p(rescue, harm),
        }
    global_best = max(item["success"] for item in result["overall"].values())
    result["global_best"] = {
        "success": global_best,
        "lambdas": [float(key) for key, item in result["overall"].items() if item["success"] == global_best],
    }
    selected = {
        task: result["tasks"][task]["best_lambdas"][0] for task in TASKS
    }
    result["taskwise_posthoc_best"] = {
        "success": sum(result["tasks"][task]["best_success"] for task in TASKS),
        "selected_lambdas": selected,
    }
    atomic_json(ARTIFACT / "FINAL_RESULTS.json", result)
    with (ARTIFACT / "PAIRED_OUTCOMES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader(); writer.writerows(paired_rows)

    labels = [f"{value:g}" for value in ALL_LAMBDAS]
    lines = [
        "# Remaining Five Tasks: Pure L11-Matched Lambda Sweep", "",
        "Same canonical seeds 0--99, 100 paired episodes per task and lambda.", "",
        "| Task | " + " | ".join(f"λ={label}" for label in labels) + " | Best |",
        "|---|" + "---:|" * (len(labels) + 1),
    ]
    for task in TASKS:
        task_result = result["tasks"][task]
        cells = [f"{task_result['lambdas'][label]['success']}/100" for label in labels]
        best_labels = ", ".join(f"{value:g}" for value in task_result["best_lambdas"])
        lines.append(
            f"| {task.removeprefix('google_robot_').removeprefix('widowx_')} | "
            + " | ".join(cells) + f" | {best_labels}: {task_result['best_success']}/100 |"
        )
    cells = [f"{result['overall'][label]['success']}/500" for label in labels]
    lines.append(
        "| **Overall** | " + " | ".join(cells)
        + f" | {', '.join(f'{value:g}' for value in result['global_best']['lambdas'])}: {global_best}/500 |"
    )
    lines.extend(["", "## Summary", "", f"- Global best fixed λ: {result['global_best']['lambdas']} with {global_best}/500.", f"- Per-task post-hoc best: {result['taskwise_posthoc_best']['success']}/500.", f"- Per-task selected λ: {selected}."])
    (ARTIFACT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(ARTIFACT / "COMPLETE.json", {"complete": True, "new_episodes": 5500})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
