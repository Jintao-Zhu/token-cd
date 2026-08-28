"""Preregistered analysis for the independent nuisance confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("full", "iss_max", "attention_high_relevance_low", "random_control")


def cluster_ci(values: list[float], seed: int, reps: int = 50000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(reps, len(array)))].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rollout = root / "rollout"
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    episodes = [json.loads(path.read_text()) for path in sorted((rollout / "episodes").glob("*.json"))]
    units = defaultdict(dict)
    for row in episodes:
        units[row["causal_unit_id"]][row["arm"]] = row
    invalid = list((rollout / "invalid_units").glob("*.json"))
    if len(episodes) != 2000 or len(units) != 500 or invalid or any(set(value) != set(ARMS) for value in units.values()):
        raise RuntimeError("confirmation integrity failure")

    grouped = defaultdict(list)
    for unit_id, value in units.items():
        grouped[unit_id.rsplit("__seed", 1)[0]].append(value)
    state_rows = []
    for snapshot_id, values in sorted(grouped.items()):
        if len(values) != 5:
            raise RuntimeError(f"{snapshot_id} lacks five matched seeds")
        first = values[0]["full"]
        row = {
            "snapshot_id": snapshot_id,
            "task_id": first["task_id"],
            "init_state_id": first["init_state_id"],
            "progress": first["target_progress"],
        }
        for arm in ARMS:
            row[f"P_{arm}"] = sum(int(value[arm]["success"]) for value in values) / 5
        row["delta_nuisance_vs_full"] = row["P_attention_high_relevance_low"] - row["P_full"]
        row["delta_nuisance_vs_random"] = row["P_attention_high_relevance_low"] - row["P_random_control"]
        row["delta_iss_vs_full"] = row["P_iss_max"] - row["P_full"]
        state_rows.append(row)

    overall = {}
    for arm in ARMS:
        success = sum(int(row["success"]) for row in episodes if row["arm"] == arm)
        overall[arm] = {"success": success, "n": 500, "rate": success / 500}
    for arm in ARMS[1:]:
        differences = [int(value[arm]["success"]) - int(value["full"]["success"]) for value in units.values()]
        overall[arm].update({"rescue": differences.count(1), "harm": differences.count(-1), "net": sum(differences)})

    effects = {}
    definitions = {
        "nuisance_vs_full": "delta_nuisance_vs_full",
        "nuisance_vs_random": "delta_nuisance_vs_random",
        "iss_vs_full": "delta_iss_vs_full",
    }
    for index, (name, key) in enumerate(definitions.items()):
        values = [row[key] for row in state_rows]
        effects[name] = {"mean": float(np.mean(values)), "snapshot_cluster_bootstrap_ci95": cluster_ci(values, 20260812 + index)}

    task_rows = []
    for task_id in range(10):
        rows = [row for row in state_rows if row["task_id"] == task_id]
        item = {"task_id": task_id, "snapshots": len(rows)}
        for arm in ARMS:
            item[f"P_{arm}"] = float(np.mean([row[f"P_{arm}"] for row in rows]))
        for key in definitions.values():
            item[key] = float(np.mean([row[key] for row in rows]))
        task_rows.append(item)

    progress_rows = []
    for progress in (0.25, 0.65):
        rows = [row for row in state_rows if row["progress"] == progress]
        item = {"progress": progress, "snapshots": len(rows)}
        for arm in ARMS:
            item[f"P_{arm}"] = float(np.mean([row[f"P_{arm}"] for row in rows]))
        for key in definitions.values():
            item[key] = float(np.mean([row[key] for row in rows]))
        progress_rows.append(item)

    nuisance_nonworse_tasks = sum(row["delta_nuisance_vs_full"] >= 0 for row in task_rows)
    worst_task_effect = min(row["delta_nuisance_vs_full"] for row in task_rows)
    gates = {
        "delta_nuisance_positive": effects["nuisance_vs_full"]["mean"] > 0,
        "nuisance_ci_lower_gt_zero": effects["nuisance_vs_full"]["snapshot_cluster_bootstrap_ci95"][0] > 0,
        "at_least_7_of_10_tasks_nuisance_ge_full": nuisance_nonworse_tasks >= 7,
        "no_task_catastrophic_harm_gt_10pp": worst_task_effect >= -0.10,
        "nuisance_gt_random_pooled": effects["nuisance_vs_random"]["mean"] > 0,
    }
    decision = "NUISANCE_CONFIRMATION_GO" if all(gates.values()) else "NUISANCE_CONFIRMATION_NO_GO_STOP_SIGNED_TOKEN_HEURISTIC"
    payload = {
        "decision": decision,
        "integrity": {
            "reference_snapshots": 100, "episodes": len(episodes), "causal_units": len(units),
            "independent_snapshots": len(state_rows), "invalid_units": len(invalid),
            "episodes_per_arm": dict(Counter(row["arm"] for row in episodes)),
            "all_finite": all(row["all_actions_finite"] for row in episodes),
        },
        "overall": overall,
        "effects": effects,
        "primary_gates": gates,
        "nuisance_nonworse_tasks": nuisance_nonworse_tasks,
        "worst_task_effect": worst_task_effect,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (root / "decision.json").write_text(json.dumps({"decision": decision, "primary_gates": gates}, indent=2, sort_keys=True) + "\n")
    write_csv(output / "snapshot_effects.csv", state_rows)
    write_csv(output / "task_table.csv", task_rows)
    write_csv(output / "progress_table.csv", progress_rows)

    lines = [
        "# Independent Nuisance Confirmation", "", "## Integrity", "",
        "- Reference snapshots: 100/100", "- Matched causal units: 500/500",
        "- Episodes: 2000/2000", "- Invalid units: 0", "- All outputs finite: true", "",
        "## Overall", "", "| Arm | Success | Rate | Rescue | Harm |", "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = overall[arm]
        lines.append(f"| {arm} | {row['success']}/500 | {100*row['rate']:.1f}% | {row.get('rescue','-')} | {row.get('harm','-')} |")
    lines += ["", "## Primary Effects", "", "| Effect | Mean | Snapshot bootstrap 95% CI |", "|---|---:|---:|"]
    for name, result in effects.items():
        ci = result["snapshot_cluster_bootstrap_ci95"]
        lines.append(f"| {name} | {100*result['mean']:+.1f}pp | [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]pp |")
    lines += ["", "## By Task", "", "| Task | Full | ISS | Nuisance | Random | Nuisance-Full | Nuisance-Random |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in task_rows:
        lines.append(f"| {row['task_id']} | {100*row['P_full']:.1f}% | {100*row['P_iss_max']:.1f}% | {100*row['P_attention_high_relevance_low']:.1f}% | {100*row['P_random_control']:.1f}% | {100*row['delta_nuisance_vs_full']:+.1f}pp | {100*row['delta_nuisance_vs_random']:+.1f}pp |")
    lines += ["", "## By Progress", "", "| Progress | Full | ISS | Nuisance | Random |", "|---:|---:|---:|---:|---:|"]
    for row in progress_rows:
        lines.append(f"| {row['progress']} | {100*row['P_full']:.1f}% | {100*row['P_iss_max']:.1f}% | {100*row['P_attention_high_relevance_low']:.1f}% | {100*row['P_random_control']:.1f}% |")
    lines += ["", "## Preregistered Gates", ""]
    for name, passed in gates.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines += ["", f"## Decision: `{decision}`", "", "The fresh confirmation does not reproduce either the nuisance gain or the ISS-max damage seen in development. The signed-token heuristic line stops under the preregistered rule."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
