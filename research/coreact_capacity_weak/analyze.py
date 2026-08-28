#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(artifact, split):
    records = [json.loads(path.read_text()) for path in sorted((artifact / split).glob("task*.json"))]
    if len(records) != 250 or any(not record["integrity"]["finite"] for record in records):
        raise RuntimeError(f"{split} incomplete or invalid: {len(records)} states")
    return records


def state_values(records, candidate, key, flow_step=None):
    output = []
    for record in records:
        rows = [row for row in record["rows"] if row["candidate"] == candidate and (flow_step is None or row["flow_step"] == flow_step)]
        output.append((record["state"]["task_id"], record["state"]["state_id"], float(np.mean([row[key] for row in rows]))))
    return output


def bootstrap(values, seed=20260816, count=10000):
    by_task = defaultdict(list)
    for task, _, value in values: by_task[task].append(value)
    rng = np.random.default_rng(seed); draws = np.empty(count)
    for index in range(count): draws[index] = np.mean([np.mean(rng.choice(by_task[task], len(by_task[task]), replace=True)) for task in range(10)])
    estimate = float(np.mean([value for _, _, value in values]))
    return estimate, [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(records, candidate):
    g, ci = bootstrap(state_values(records, candidate, "g_positive"))
    per_task = {task: float(np.mean([value for row_task, _, value in state_values(records, candidate, "g_positive") if row_task == task])) for task in range(10)}
    point_rows = [row for record in records for row in record["rows"] if row["candidate"] == candidate]
    return {
        "candidate": candidate, "p_g_positive": g, "g_ci95": ci, "per_task_g_positive": per_task, "tasks_above_half": sum(value > 0.5 for value in per_task.values()),
        "weak_mse": float(np.mean([row["weak_mse"] for row in point_rows])), "strong_mse": float(np.mean([row["strong_mse"] for row in point_rows])), "weak_inferior": bool(np.mean([row["weak_mse"] for row in point_rows]) > np.mean([row["strong_mse"] for row in point_rows])),
        "mean_weak_to_strong_mse_ratio": float(np.mean([row["weak_to_strong_mse_ratio"] for row in point_rows])), "p_r_gt_1": float(np.mean([row["r_gt_1"] for row in point_rows])), "p_error_cosine_positive": float(np.mean([row["error_cosine_positive"] for row in point_rows])),
        "mean_cosine": float(np.mean([row["cosine"] for row in point_rows])), "median_cosine": float(np.median([row["cosine"] for row in point_rows])), "p_delta_mse_raw_positive": float(np.mean([row["delta_mse_raw_positive"] for row in point_rows])), "p_delta_mse_applied_positive": float(np.mean([row["delta_mse_applied_positive"] for row in point_rows])),
        "mean_direction_relative_norm": float(np.mean([row["direction_relative_norm"] for row in point_rows])), "clip_rate": float(np.mean([row["clipped"] for row in point_rows])),
    }


def diagnostics(artifact, split, records, candidates):
    timestep, tasks, groups = [], [], []
    for candidate in candidates:
        for step in range(10):
            rows = [row for record in records for row in record["rows"] if row["candidate"] == candidate and row["flow_step"] == step]
            timestep.append({"candidate": candidate, "flow_step": step, "timestep": 1.0-step*0.1, "p_g_positive": np.mean([row["g_positive"] for row in rows]), "mean_cosine": np.mean([row["cosine"] for row in rows]), "p_delta_mse_applied_positive": np.mean([row["delta_mse_applied_positive"] for row in rows])})
        for task in range(10):
            rows = [row for record in records for row in record["rows"] if row["candidate"] == candidate and row["task_id"] == task]
            tasks.append({"candidate": candidate, "task_id": task, "p_g_positive": np.mean([row["g_positive"] for row in rows]), "p_r_gt_1": np.mean([row["r_gt_1"] for row in rows]), "p_delta_mse_applied_positive": np.mean([row["delta_mse_applied_positive"] for row in rows])})
        for group in ("translation", "rotation", "gripper"):
            rows = [row for record in records for row in record["rows"] if row["candidate"] == candidate]
            groups.append({"candidate": candidate, "group": group, "p_g_positive": np.mean([row[f"{group}_g_positive"] for row in rows]), "mean_cosine": np.mean([row[f"{group}_cosine"] for row in rows]), "p_delta_mse_applied_positive": np.mean([row[f"{group}_delta_mse_applied_positive"] for row in rows])})
    for name, rows in (("timestep", timestep), ("per_task", tasks), ("action_groups", groups)):
        with (artifact / f"{split}_{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True); parser.add_argument("--split", choices=("selection", "confirmation"), required=True); args = parser.parse_args()
    artifact = args.artifact.resolve(); records = load(artifact, args.split)
    candidates = ["weak_a_8l", "weak_b_4l"] if args.split == "selection" else [json.loads((artifact / "selected_weak.lock.json").read_text())["selected_weak"]]
    summaries = {candidate: summarize(records, candidate) for candidate in candidates}; diagnostics(artifact, args.split, records, candidates)
    for summary in summaries.values(): summary["passes"] = bool(summary["weak_inferior"] and summary["p_g_positive"] >= .60 and summary["g_ci95"][0] > .50 and summary["tasks_above_half"] >= 7 and summary["p_r_gt_1"] >= .55 and summary["p_delta_mse_applied_positive"] >= .55)
    if args.split == "selection":
        legal = [value for value in summaries.values() if value["passes"]]
        if not legal:
            decision = "CAPACITY_WEAK_COMPATIBILITY_NO_GO"; (artifact / "decision.json").write_text(json.dumps({"decision": decision}, indent=2) + "\n")
        else:
            a, b = summaries["weak_a_8l"], summaries["weak_b_4l"]
            if a["passes"] and b["passes"]: selected = "weak_a_8l" if b["p_g_positive"] - a["p_g_positive"] < .03 else "weak_b_4l"
            else: selected = legal[0]["candidate"]
            decision = "CAPACITY_WEAK_SELECTION_PASS_CONFIRMATION_SEALED"; (artifact / "selected_weak.lock.json").write_text(json.dumps({"selected_weak": selected, "selection_p_g_positive": summaries[selected]["p_g_positive"], "lambda": .5, "trust_region_kappa": .25}, indent=2) + "\n")
        output = {"decision": decision, "candidates": summaries}
        (artifact / "selection_gate.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    else:
        summary = next(iter(summaries.values())); decision = "CAPACITY_WEAK_COMPATIBILITY_CONFIRMED_READY_FOR_ROLLOUT" if summary["passes"] else "CAPACITY_WEAK_SELECTION_PASS_CONFIRMATION_FAIL"
        output = {"decision": decision, "candidate": summary}; (artifact / "confirmation_gate.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n"); (artifact / "decision.json").write_text(json.dumps({"decision": decision}, indent=2) + "\n")
    print(json.dumps({"split": args.split, "decision": decision}, sort_keys=True))


if __name__ == "__main__": main()
