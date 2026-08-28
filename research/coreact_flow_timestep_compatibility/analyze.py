#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TIMES = tuple(round(1.0 - 0.1 * i, 1) for i in range(10))


def load_rows(artifact: Path, split: str):
    files = sorted((artifact / split).glob("task*.json"))
    records = [json.loads(path.read_text()) for path in files]
    if len(records) != 250:
        raise RuntimeError(f"{split}: expected 250 completed states, got {len(records)}")
    if any(len(record["rows"]) != 30 for record in records):
        raise RuntimeError(f"{split}: each state must contain 3 noise x 10 timestep rows")
    if any(not record["integrity"]["finite"] for record in records):
        raise RuntimeError(f"{split}: nonfinite record")
    if any(record["integrity"]["manual_vs_helper_x_t_max_abs"] != 0.0 or record["integrity"]["manual_vs_helper_target_max_abs"] != 0.0 for record in records):
        raise RuntimeError(f"{split}: flow target reconstruction mismatch")
    return records


def state_values(records, steps, key):
    output = []
    for record in records:
        selected = [row for row in record["rows"] if row["flow_step"] in steps]
        output.append((record["state"]["task_id"], record["state"]["state_id"], float(np.mean([row[key] for row in selected]))))
    return output


def cluster_ci(values, *, seed=20260815, replicates=10000):
    by_task = defaultdict(list)
    for task, _, value in values:
        by_task[task].append(value)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for index in range(replicates):
        task_means = []
        for task in range(10):
            data = np.asarray(by_task[task], dtype=float)
            task_means.append(float(np.mean(rng.choice(data, size=len(data), replace=True))))
        samples[index] = np.mean(task_means)
    estimate = float(np.mean([value for _, _, value in values]))
    return estimate, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_window(records, steps):
    values = state_values(records, steps, "g_positive")
    estimate, low, high = cluster_ci(values)
    per_task = {
        task: float(np.mean([value for row_task, _, value in values if row_task == task]))
        for task in range(10)
    }
    applied = float(np.mean([value for _, _, value in state_values(records, steps, "delta_mse_applied_positive")]))
    return {"steps": list(steps), "p_g_positive": estimate, "ci95": [low, high], "per_task": per_task, "tasks_at_or_above_half": sum(value >= 0.5 for value in per_task.values()), "tasks_above_half": sum(value > 0.5 for value in per_task.values()), "p_applied_delta_mse_positive": applied}


def timestep_table(records):
    rows = []
    for step, timestep in enumerate(TIMES):
        selected = [row for record in records for row in record["rows"] if row["flow_step"] == step]
        state_g = state_values(records, (step,), "g_positive")
        p_g, ci_low, ci_high = cluster_ci(state_g)
        rows.append({
            "flow_step": step,
            "timestep": timestep,
            "noise_regime": "highest" if step == 0 else ("lowest" if step == 9 else "intermediate"),
            "p_g_positive": p_g,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "mean_g": float(np.mean([row["g_dot"] for row in selected])),
            "median_g": float(np.median([row["g_dot"] for row in selected])),
            "mean_cosine": float(np.mean([row["cosine"] for row in selected])),
            "median_cosine": float(np.median([row["cosine"] for row in selected])),
            "p_cosine_positive": float(np.mean([row["cosine_positive"] for row in selected])),
            "p_delta_mse_raw_positive": float(np.mean([row["delta_mse_raw_positive"] for row in selected])),
            "p_delta_mse_applied_positive": float(np.mean([row["delta_mse_applied_positive"] for row in selected])),
            "mean_delta_mse_raw": float(np.mean([row["delta_mse_raw"] for row in selected])),
            "mean_delta_mse_applied": float(np.mean([row["delta_mse_applied"] for row in selected])),
            "median_direction_relative_norm": float(np.median([row["direction_relative_norm"] for row in selected])),
            "mean_direction_relative_norm": float(np.mean([row["direction_relative_norm"] for row in selected])),
            "clip_rate": float(np.mean([row["clipped"] for row in selected])),
        })
    return rows


def per_task_table(records):
    rows = []
    for task in range(10):
        task_records = [record for record in records if record["state"]["task_id"] == task]
        for step, timestep in enumerate(TIMES):
            points = [row for record in task_records for row in record["rows"] if row["flow_step"] == step]
            rows.append({"task_id": task, "flow_step": step, "timestep": timestep, "p_g_positive": float(np.mean([row["g_positive"] for row in points])), "median_cosine": float(np.median([row["cosine"] for row in points])), "p_delta_mse_applied_positive": float(np.mean([row["delta_mse_applied_positive"] for row in points]))})
    return rows


def group_table(records):
    rows = []
    for group in ("translation", "rotation", "gripper"):
        for step, timestep in enumerate(TIMES):
            points = [row for record in records for row in record["rows"] if row["flow_step"] == step]
            rows.append({"group": group, "flow_step": step, "timestep": timestep, "p_g_positive": float(np.mean([row[f"{group}_g_positive"] for row in points])), "mean_cosine": float(np.mean([row[f"{group}_cosine"] for row in points])), "median_cosine": float(np.median([row[f"{group}_cosine"] for row in points]))})
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(artifact, split, overall, per_task):
    x = [row["timestep"] for row in overall]
    y = [100 * row["p_g_positive"] for row in overall]
    lo = [100 * row["ci95_low"] for row in overall]
    hi = [100 * row["ci95_high"] for row in overall]
    figure, axis = plt.subplots(figsize=(8, 5))
    for task in range(10):
        task_rows = [row for row in per_task if row["task_id"] == task]
        axis.plot(x, [100 * row["p_g_positive"] for row in task_rows], alpha=0.25, linewidth=1)
    axis.plot(x, y, color="black", marker="o", linewidth=2, label="Overall")
    axis.fill_between(x, lo, hi, color="black", alpha=0.12, label="95% state-cluster CI")
    axis.axhline(50, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("Flow time t (1.0 = highest noise, 0.1 = lowest noise)")
    axis.set_ylabel("P(G > 0), %")
    axis.invert_xaxis()
    axis.legend()
    figure.tight_layout()
    figure.savefig(artifact / "figures" / f"{split}_g_positive_curve.png", dpi=180)
    plt.close(figure)


def analyze_selection(artifact, records):
    full = summarize_window(records, tuple(range(10)))
    candidates = []
    for length in range(2, 10):
        low_steps = tuple(range(10 - length, 10))
        high_steps = tuple(range(length))
        low = summarize_window(records, low_steps)
        high = summarize_window(records, high_steps)
        low.update({"length": length, "matched_high_noise": high, "gain_over_full_pp": 100 * (low["p_g_positive"] - full["p_g_positive"]), "gain_over_high_pp": 100 * (low["p_g_positive"] - high["p_g_positive"])})
        low["passes"] = bool(low["p_g_positive"] >= 0.60 and low["ci95"][0] > 0.50 and low["tasks_at_or_above_half"] >= 7 and low["gain_over_full_pp"] >= 10.0 and low["gain_over_high_pp"] >= 10.0)
        candidates.append(low)
    legal = [row for row in candidates if row["passes"]]
    result = {"full": full, "candidates": candidates}
    if not legal:
        result["decision"] = "NO_FLOW_TIMESTEP_COMPATIBILITY_REGIME"
        (artifact / "decision.json").write_text(json.dumps({"decision": result["decision"]}, indent=2) + "\n")
    else:
        selected = sorted(legal, key=lambda row: (-row["p_g_positive"], row["length"], row["steps"][0]))[0]
        result["decision"] = "SELECTION_PASS_CONFIRMATION_SEALED"
        result["selected"] = selected
        lock = {"selection_decision": "PASS", "low_noise_steps": selected["steps"], "low_noise_times": [TIMES[i] for i in selected["steps"]], "high_noise_placebo_steps": selected["matched_high_noise"]["steps"], "lambda": 0.5, "trust_region_kappa": 0.25}
        (artifact / "selected_segment.lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        (artifact / "decision.json").write_text(json.dumps({"decision": result["decision"]}, indent=2) + "\n")
    (artifact / "selection_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def analyze_confirmation(artifact, records):
    lock = json.loads((artifact / "selected_segment.lock.json").read_text())
    low = summarize_window(records, tuple(lock["low_noise_steps"]))
    high = summarize_window(records, tuple(lock["high_noise_placebo_steps"]))
    passes = bool(low["p_g_positive"] >= 0.60 and low["ci95"][0] > 0.50 and low["tasks_above_half"] >= 7 and low["p_g_positive"] > high["p_g_positive"] and low["p_applied_delta_mse_positive"] >= 0.55)
    decision = "FLOW_TIMESTEP_COMPATIBILITY_CONFIRMED_READY_FOR_CLOSED_LOOP" if passes else "FLOW_TIMESTEP_REGIME_SELECTION_PASS_CONFIRMATION_FAIL"
    result = {"decision": decision, "low_noise": low, "matched_high_noise": high}
    (artifact / "confirmation_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"decision": decision}, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "confirmation"), required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    records = load_rows(artifact, args.split)
    overall, tasks, groups = timestep_table(records), per_task_table(records), group_table(records)
    write_csv(artifact / f"{args.split}_timestep_compatibility.csv", overall)
    write_csv(artifact / f"{args.split}_per_task_timestep.csv", tasks)
    write_csv(artifact / f"{args.split}_action_groups.csv", groups)
    plot_curves(artifact, args.split, overall, tasks)
    result = analyze_selection(artifact, records) if args.split == "selection" else analyze_confirmation(artifact, records)
    print(json.dumps({"split": args.split, "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
