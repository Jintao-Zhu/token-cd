"""Aggregate the 240-state projector feature/reconstruction diagnostic."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from research.semantic_token_cd.prompt_action_complement_protocol import ARMS, ARTIFACT, TASKS, atomic_json


def mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def main():
    files = sorted((ARTIFACT / "mechanism_analysis/features").glob("**/step_*.json"))
    if len(files) != 240: raise RuntimeError(f"feature diagnostic incomplete: {len(files)}/240")
    rows = []
    for path in files:
        data = json.loads(path.read_text())
        for arm, metrics in data["arms"].items():
            rows.append({"task": data["task"], "seed": data["seed"], "control_step": data["control_step"],
                         "arm": arm, **metrics})
    out = ARTIFACT / "mechanism_analysis/feature_reconstruction_per_state.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for task in (*TASKS, "overall"):
        task_rows = rows if task == "overall" else [row for row in rows if row["task"] == task]
        key = task.removeprefix("google_robot_"); summary[key] = {}
        for arm in ARMS:
            subset = [row for row in task_rows if row["arm"] == arm]
            summary[key][arm] = {field: mean(row[field] for row in subset) for field in (
                "supplement_core_feature_cosine", "supplement_internal_feature_cosine",
                "core_internal_feature_cosine", "selected_reconstruction_error_mean",
                "selected_reconstruction_error_norm", "core_reconstruction_error_mean",
                "supplement_reconstruction_error_mean")}
    atomic_json(ARTIFACT / "mechanism_analysis/feature_reconstruction_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
