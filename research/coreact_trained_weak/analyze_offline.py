from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


LAMBDAS = (0.1, 0.25, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    files = sorted((artifact / "trained_weak_offline_raw").glob("*.json"))
    if len(files) != 200:
        raise RuntimeError(f"expected 200 state files, got {len(files)}")
    payloads = [json.loads(path.read_text()) for path in files]
    if any(len(value["rows"]) != 30 or not value["integrity"]["finite"] for value in payloads):
        raise RuntimeError("offline raw integrity failure")
    points = pd.DataFrame([row for value in payloads for row in value["rows"]])
    if len(points) != 6000 or points.point_id.nunique() != 6000:
        raise RuntimeError("point count/identity mismatch")
    points.to_parquet(artifact / "trained_weak_offline_points.parquet", index=False)

    metric_columns = [
        "strong_error", "weak_error", "delta_quality", "extrapolation_validity",
        "ev_cosine", "strong_weak_cosine", "relative_correction_norm", "direction_norm",
        *[f"applied_ratio_lambda_{value:g}" for value in LAMBDAS],
        *[f"clip_scale_lambda_{value:g}" for value in LAMBDAS],
    ]
    states = points.groupby(["task_id", "state_id"], as_index=False)[metric_columns].mean()
    states["quality_ordered"] = states.delta_quality > 0
    states["ev_positive"] = states.extrapolation_validity > 0
    states.to_csv(artifact / "trained_weak_offline_states.csv", index=False)

    tasks = states.groupby("task_id", as_index=False).agg(
        states=("state_id", "count"),
        median_delta_quality=("delta_quality", "median"),
        mean_delta_quality=("delta_quality", "mean"),
        quality_ordered_states=("quality_ordered", "sum"),
        ev_positive_states=("ev_positive", "sum"),
        median_ev_cosine=("ev_cosine", "median"),
        median_relative_correction_norm=("relative_correction_norm", "median"),
    )
    tasks["quality_ordered_rate"] = tasks.quality_ordered_states / tasks.states
    tasks.to_csv(artifact / "trained_weak_offline_tasks.csv", index=False)

    calibration = []
    for value in LAMBDAS:
        column = f"applied_ratio_lambda_{value:g}"
        ratios = points[column].to_numpy()
        calibration.append({
            "lambda": value,
            "median_applied_ratio": float(np.median(ratios)),
            "mean_applied_ratio": float(np.mean(ratios)),
            "p95_applied_ratio": float(np.quantile(ratios, 0.95)),
            "clipped_fraction": float(np.mean(points[f"clip_scale_lambda_{value:g}"] < 1 - 1e-12)),
            "distance_to_target_0.10": abs(float(np.median(ratios)) - 0.10),
        })
    calibration.sort(key=lambda row: (row["distance_to_target_0.10"], row["lambda"]))
    selected = calibration[0]["lambda"]
    with (artifact / "guidance_strength_calibration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration[0]))
        writer.writeheader()
        writer.writerows(sorted(calibration, key=lambda row: row["lambda"]))
    lock = {
        "selected_lambda": selected,
        "candidates": list(LAMBDAS),
        "selection_target_median_applied_ratio": 0.10,
        "selection_rule": "closest median applied correction ratio to 0.10; ties choose smaller lambda",
        "trust_region_kappa": 0.25,
        "calibration_points": 6000,
        "closed_loop_outcomes_used": False,
        "strong_step": 15000,
        "weak_step": 10000,
    }
    with (artifact / "guidance_strength.lock.yaml").open("w") as stream:
        yaml.safe_dump(lock, stream, sort_keys=False)

    state_rate = float(states.quality_ordered.mean())
    positive_tasks = int((tasks.median_delta_quality > 0).sum())
    gate_pass = state_rate > 0.60 and positive_tasks >= 7
    analysis = {
        "status": "TRAINED_WEAK_OFFLINE_PASS_LAMBDA_LOCKED" if gate_pass else "TRAINED_CHECKPOINTS_NOT_FLOW_QUALITY_ORDERED_STOP",
        "points": 6000,
        "states": 200,
        "state_quality_ordered_rate": state_rate,
        "tasks_positive_median_delta_quality": positive_tasks,
        "ev_positive_state_rate": float(states.ev_positive.mean()),
        "selected_lambda": selected,
        "calibration": sorted(calibration, key=lambda row: row["lambda"]),
        "gate_pass": gate_pass,
        "deterministic_repeat_max_abs": max(value["integrity"]["deterministic_repeat_max_abs"] for value in payloads),
    }
    (artifact / "trained_weak_offline_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    report = f"""# Trained-Weak Offline Diagnosis

- Points: 6000 (10 tasks x 20 states x 3 noise x 10 flow steps)
- State-level quality ordered: {state_rate:.1%}
- Tasks with positive median delta Q: {positive_tasks}/10
- State-level EV positive: {analysis['ev_positive_state_rate']:.1%}
- Selected lambda: {selected}
- Gate: {'PASS' if gate_pass else 'FAIL'}

Offline EV is diagnostic only. Lambda was selected without closed-loop outcomes.
"""
    (artifact / "trained_weak_offline_report.md").write_text(report)
    print(json.dumps(analysis))


if __name__ == "__main__":
    main()
