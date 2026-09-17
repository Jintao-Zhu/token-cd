"""Analyze the complete pure L11-Matched fixed-lambda response curve."""
from __future__ import annotations

import csv
import json
import math
import pickle

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.l11_dense_lambda_protocol import (
    ARM_LAMBDAS,
    ARTIFACT,
    CANONICAL,
    EXISTING_LAMBDA_ROOTS,
    SEEDS,
    TASKS,
    atomic_json,
    lambda_arm,
)


def exact_p(rescue: int, harm: int) -> float:
    count = rescue + harm
    if not count:
        return 1.0
    tail = sum(math.comb(count, index) for index in range(min(rescue, harm) + 1)) / 2**count
    return min(1.0, 2 * tail)


def summary_path(task: str, seed: int, lambd: float):
    if lambd in EXISTING_LAMBDA_ROOTS:
        root = EXISTING_LAMBDA_ROOTS[lambd][task]
    else:
        root = ARTIFACT / "episodes" / task / lambda_arm(lambd)
    return root / f"episode_{seed:03d}_summary.json"


def main() -> None:
    lambdas = tuple(sorted((*EXISTING_LAMBDA_ROOTS.keys(), *ARM_LAMBDAS.values())))
    outcomes: dict[str, dict[float, list[bool]]] = {task: {lambd: [] for lambd in lambdas} for task in TASKS}
    paired_rows = []
    for task in TASKS:
        for seed in SEEDS:
            with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                expected = snapshot_sha(pickle.load(handle))
            row = {"task": task, "seed": seed}
            for lambd in lambdas:
                path = summary_path(task, seed, lambd)
                item = json.loads(path.read_text())
                if item.get("canonical_snapshot_sha256") != expected:
                    raise RuntimeError(f"canonical mismatch: {path}")
                if lambd not in EXISTING_LAMBDA_ROOTS and item.get("technical_pass") is not True:
                    raise RuntimeError(f"technical audit failure: {path}")
                success = bool(item["success"])
                outcomes[task][lambd].append(success)
                row[f"lambda_{lambd:g}"] = int(success)
            paired_rows.append(row)

    result = {"complete": True, "n_per_task_lambda": 100, "tasks": {}, "overall": {}}
    for task in TASKS:
        task_result = {}
        baseline = outcomes[task][0.5]
        for lambd in lambdas:
            current = outcomes[task][lambd]
            rescue = sum(value and not reference for value, reference in zip(current, baseline, strict=True))
            harm = sum(reference and not value for value, reference in zip(current, baseline, strict=True))
            task_result[f"{lambd:g}"] = {
                "success": sum(current),
                "rescue_vs_0.5": rescue,
                "harm_vs_0.5": harm,
                "net_vs_0.5": rescue - harm,
                "p_exact_vs_0.5": exact_p(rescue, harm),
            }
        best_success = max(value["success"] for value in task_result.values())
        result["tasks"][task] = {
            "lambdas": task_result,
            "best_success": best_success,
            "best_lambdas": [float(key) for key, value in task_result.items() if value["success"] == best_success],
        }

    for lambd in lambdas:
        current = [value for task in TASKS for value in outcomes[task][lambd]]
        baseline = [value for task in TASKS for value in outcomes[task][0.5]]
        rescue = sum(value and not reference for value, reference in zip(current, baseline, strict=True))
        harm = sum(reference and not value for value, reference in zip(current, baseline, strict=True))
        result["overall"][f"{lambd:g}"] = {
            "success": sum(current),
            "n": 400,
            "rescue_vs_0.5": rescue,
            "harm_vs_0.5": harm,
            "net_vs_0.5": rescue - harm,
            "p_exact_vs_0.5": exact_p(rescue, harm),
        }
    result["taskwise_posthoc_best"] = {
        "success": sum(value["best_success"] for value in result["tasks"].values()),
        "n": 400,
        "selected_lambdas": {task: value["best_lambdas"] for task, value in result["tasks"].items()},
    }
    global_best = max(value["success"] for value in result["overall"].values())
    result["global_best"] = {
        "success": global_best,
        "n": 400,
        "lambdas": [float(key) for key, value in result["overall"].items() if value["success"] == global_best],
    }
    atomic_json(ARTIFACT / "FINAL_RESULTS.json", result)

    fieldnames = list(paired_rows[0])
    with (ARTIFACT / "PAIRED_OUTCOMES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)

    labels = [f"{lambd:g}" for lambd in lambdas]
    lines = [
        "# Pure L11-Matched Dense Lambda Sweep",
        "",
        "Same four tasks, canonical seeds 0--99, 100 paired episodes per task and lambda.",
        "",
        "| Task | " + " | ".join(f"λ={label}" for label in labels) + " | Best |",
        "|---|" + "---:|" * (len(labels) + 1),
    ]
    for task in TASKS:
        task_result = result["tasks"][task]
        values = [f"{task_result['lambdas'][label]['success']}/100" for label in labels]
        best_labels = ", ".join(f"{value:g}" for value in task_result["best_lambdas"])
        lines.append(f"| {task.removeprefix('google_robot_')} | " + " | ".join(values) + f" | {best_labels}: {task_result['best_success']}/100 |")
    overall_values = [f"{result['overall'][label]['success']}/400" for label in labels]
    lines.append("| **Overall** | " + " | ".join(overall_values) + f" | {', '.join(f'{value:g}' for value in result['global_best']['lambdas'])}: {global_best}/400 |")
    lines += [
        "",
        "## New lambdas paired against λ=.5",
        "",
        "| λ | Success | Rescue | Harm | Net | exact p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for lambd in ARM_LAMBDAS.values():
        value = result["overall"][f"{lambd:g}"]
        lines.append(
            f"| {lambd:g} | {value['success']}/400 | {value['rescue_vs_0.5']} | "
            f"{value['harm_vs_0.5']} | {value['net_vs_0.5']:+d} | {value['p_exact_vs_0.5']:.5g} |"
        )
    taskwise = result["taskwise_posthoc_best"]
    lines += [
        "",
        "## Selection summaries",
        "",
        f"- Global best fixed lambda: {result['global_best']['lambdas']} with {global_best}/400.",
        f"- Per-task post-hoc best: {taskwise['success']}/400.",
        f"- Per-task selected lambdas: {taskwise['selected_lambdas']}.",
    ]
    (ARTIFACT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(ARTIFACT / "COMPLETE.json", {"complete": True, "new_episodes": 3200})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

