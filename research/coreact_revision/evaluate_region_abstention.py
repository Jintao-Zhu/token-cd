#!/usr/bin/env python3
"""Nested task-heldout evaluation of high-confidence nuisance abstention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research.coreact_revision.evaluate_region_sign_probe_v3 import ATTENTION, MAGNITUDE, REGION, aggregate


TARGET_PRECISION = 0.60


def matrix(rows: list[dict], numeric: list[str]) -> np.ndarray:
    return np.asarray([[row[name] for name in numeric] + [row["camera_id"]] for row in rows], dtype=object)


def fit_predict(train: list[dict], test: list[dict], numeric: list[str]) -> np.ndarray:
    transform = ColumnTransformer([
        ("numeric", StandardScaler(), list(range(len(numeric)))),
        ("camera", OneHotEncoder(handle_unknown="ignore"), [len(numeric)]),
    ])
    model = make_pipeline(transform, LogisticRegression(class_weight="balanced", max_iter=3000, random_state=20260808))
    model.fit(matrix(train, numeric), [row["label_nuisance"] for row in train])
    return model.predict_proba(matrix(test, numeric))[:, 1]


def inner_probabilities(rows: list[dict], numeric: list[str]) -> tuple[np.ndarray, np.ndarray]:
    probabilities, labels = [], []
    for task in sorted({row["task_id"] for row in rows}):
        train = [row for row in rows if row["task_id"] != task]
        test = [row for row in rows if row["task_id"] == task]
        probabilities.extend(fit_predict(train, test, numeric)); labels.extend(row["label_nuisance"] for row in test)
    return np.asarray(probabilities), np.asarray(labels)


def locked_threshold(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    best = None
    for threshold in sorted(set(probabilities.tolist()), reverse=True):
        selected = probabilities >= threshold
        if not np.any(selected):
            continue
        precision = float(np.mean(labels[selected] == 1))
        recall = float(np.sum(labels[selected] == 1) / max(1, np.sum(labels == 1)))
        if precision < TARGET_PRECISION:
            continue
        candidate = (recall, precision, threshold, int(selected.sum()))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return {"feasible": False, "threshold": 1.0 + 1e-9, "precision": None, "recall": 0.0, "selected": 0}
    recall, precision, threshold, selected = best
    return {"feasible": True, "threshold": threshold, "precision": precision, "recall": recall, "selected": selected}


def evaluate(rows: list[dict], numeric: list[str]) -> tuple[dict, list[dict]]:
    all_labels, all_probabilities, all_predictions = [], [], []
    folds, decisions = [], []
    for heldout_task in sorted({row["task_id"] for row in rows}):
        outer_train = [row for row in rows if row["task_id"] != heldout_task]
        outer_test = [row for row in rows if row["task_id"] == heldout_task]
        inner_p, inner_y = inner_probabilities(outer_train, numeric)
        threshold = locked_threshold(inner_p, inner_y)
        probabilities = fit_predict(outer_train, outer_test, numeric)
        labels = np.asarray([row["label_nuisance"] for row in outer_test])
        predictions = probabilities >= threshold["threshold"]
        selected = int(predictions.sum()); true_selected = int(np.sum(labels[predictions])) if selected else 0
        fold = {
            "heldout_task_id": heldout_task, "groups": len(outer_test), "nuisances": int(labels.sum()),
            "inner_threshold": threshold, "outer_selected": selected, "outer_true_nuisance": true_selected,
            "outer_false_anchor": selected - true_selected,
            "outer_precision": (true_selected / selected) if selected else None,
            "outer_recall": true_selected / max(1, int(labels.sum())),
        }
        folds.append(fold)
        for row, probability, prediction in zip(outer_test, probabilities, predictions, strict=True):
            decisions.append({
                "task_id": row["task_id"], "demo_id": row["demo_id"], "frame_id": row["frame_id"],
                "group_id": row["group_id"], "camera_id": row["camera_id"],
                "label_nuisance": row["label_nuisance"], "probability": float(probability),
                "threshold": threshold["threshold"], "selected": bool(prediction),
            })
        all_labels.extend(labels); all_probabilities.extend(probabilities); all_predictions.extend(predictions)
    labels = np.asarray(all_labels); probabilities = np.asarray(all_probabilities); predictions = np.asarray(all_predictions, dtype=bool)
    selected = int(predictions.sum())
    result = {
        "groups": len(rows), "nuisances": int(labels.sum()), "selected": selected,
        "selection_rate": selected / len(rows),
        "nuisance_precision": float(precision_score(labels, predictions, zero_division=0)),
        "nuisance_recall": float(recall_score(labels, predictions, zero_division=0)),
        "nuisance_average_precision_unthresholded": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "all_tasks_selected_at_least_one": all(fold["outer_selected"] > 0 for fold in folds),
        "folds": folds,
    }
    result["gate_pass"] = result["nuisance_precision"] >= .60 and result["nuisance_recall"] >= .20 and result["all_tasks_selected_at_least_one"]
    return result, decisions


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--raw-effects", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--decisions", type=Path, required=True); args = p.parse_args()
    rows = aggregate(args.raw_effects)
    methods = {"attention_only": ATTENTION, "attention_plus_region": ATTENTION + REGION, "attention_region_plus_unsigned_magnitude": ATTENTION + REGION + MAGNITUDE}
    result = {"stage": "nested task-heldout development", "target_inner_precision": TARGET_PRECISION, "methods": {}}
    decision_rows = []
    for name, features in methods.items():
        metrics, decisions = evaluate(rows, features); result["methods"][name] = metrics
        decision_rows.extend({"method": name, **row} for row in decisions)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.decisions.open("w") as stream:
        for row in decision_rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
