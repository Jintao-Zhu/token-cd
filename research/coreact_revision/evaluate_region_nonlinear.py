#!/usr/bin/env python3
"""Task-heldout nonlinear sign classifiers with locked hyperparameters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score, recall_score

from research.coreact_revision.evaluate_region_sign_probe_v3 import aggregate


FEATURES = ["late_half_action_to_context_attention", "late_half_value_weighted_attention", "target_object_fraction", "goal_container_fraction", "background_fraction", "ambiguous_fraction", "row", "column", "camera_numeric"]


def matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[row[name] if name != "camera_numeric" else (0 if row["camera_id"] == "camera1" else 1) for name in FEATURES] for row in rows], dtype=np.float32)


def run_model(rows: list[dict], model_factory) -> dict:
    labels = np.asarray([r["label_nuisance"] for r in rows]); predictions = np.zeros(len(rows), dtype=int); probabilities = np.zeros(len(rows)); folds = []
    tasks = sorted({r["task_id"] for r in rows})
    for task in tasks:
        train = [r for r in rows if r["task_id"] != task]; test = [r for r in rows if r["task_id"] == task]
        model = model_factory(); model.fit(matrix(train), [r["label_nuisance"] for r in train]); p = model.predict_proba(matrix(test))[:, 1]; idx = [i for i, r in enumerate(rows) if r["task_id"] == task]
        probabilities[idx] = p; predictions[idx] = p >= 0.5
        folds.append({"heldout_task_id": task, "groups": len(test), "nuisances": int(sum(r["label_nuisance"] for r in test)), "predicted_nuisance": int((p >= .5).sum())})
    return {"groups": len(rows), "anchors": int((labels == 0).sum()), "nuisances": int(labels.sum()), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)), "nuisance_recall": float(recall_score(labels, predictions, zero_division=0)), "nuisance_precision": float(precision_score(labels, predictions, zero_division=0)), "nuisance_average_precision": float(average_precision_score(labels, probabilities)), "folds": folds}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--raw-effects", type=Path, required=True); p.add_argument("--value-scores", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    rows = aggregate(args.raw_effects); values = {}
    for line in args.value_scores.read_text().splitlines():
        row = json.loads(line); values[(row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])] = row
    for row in rows:
        row.update({k: values[(row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])][k] for k in ("late_half_value_weighted_attention",)})
    methods = {
        "random_forest": lambda: RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=10, class_weight="balanced_subsample", random_state=20260808, n_jobs=1),
        "histogram_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1.0, class_weight="balanced", random_state=20260808),
    }
    result = {"stage": "multitask development; whole-task heldout", "features": FEATURES, "methods": {name: run_model(rows, factory) for name, factory in methods.items()}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
