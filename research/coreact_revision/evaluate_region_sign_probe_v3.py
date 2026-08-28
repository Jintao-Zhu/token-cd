#!/usr/bin/env python3
"""Evaluate same-state region features for anchor/nuisance sign."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EFFECT_THRESHOLD = 0.0015907290887758336
ATTENTION = ["late_half_action_to_context_attention"]
REGION = ["target_object_fraction", "goal_container_fraction", "background_fraction", "ambiguous_fraction", "row", "column"]
MAGNITUDE = ["i_median", "i_iqr", "i_max"]


def aggregate(path: Path) -> list[dict]:
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups[(row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])].append(row)
    output = []
    for (task, demo, frame, group), rows in groups.items():
        if len(rows) != 6:
            raise RuntimeError(f"{(demo, frame, group)} has {len(rows)} flow rows, expected 6")
        q = np.asarray([r["Q_rel"] for r in rows]); effect = np.asarray([r["I_G"] for r in rows])
        q_med, i_med = float(np.median(q)), float(np.median(effect))
        if i_med <= EFFECT_THRESHOLD or abs(q_med) <= 0.05:
            continue
        first = rows[0]
        output.append({
            "task_id": task, "demo_id": demo, "frame_id": frame, "group_id": group,
            "camera_id": first["camera_id"], "late_half_action_to_context_attention": float(first["late_half_action_to_context_attention"]),
            **{name: float(first[name]) for name in REGION},
            "i_median": i_med, "i_iqr": float(np.quantile(effect, .75) - np.quantile(effect, .25)), "i_max": float(np.max(effect)),
            "label_nuisance": int(q_med < -0.05), "q_median": q_med,
        })
    return output


def evaluate(rows: list[dict], numeric: list[str], split_field: str) -> dict:
    labels = np.asarray([r["label_nuisance"] for r in rows]); predictions = np.zeros(len(rows), dtype=int); probabilities = np.zeros(len(rows))
    folds = []
    folds_values = sorted({r[split_field] for r in rows})
    for heldout in folds_values:
        test = [i for i, r in enumerate(rows) if r[split_field] == heldout]; train = [i for i, r in enumerate(rows) if r[split_field] != heldout]
        x_train = np.asarray([[rows[i][k] for k in numeric] + [rows[i]["camera_id"]] for i in train], dtype=object)
        x_test = np.asarray([[rows[i][k] for k in numeric] + [rows[i]["camera_id"]] for i in test], dtype=object)
        transform = ColumnTransformer([("numeric", StandardScaler(), list(range(len(numeric)))), ("camera", OneHotEncoder(handle_unknown="ignore"), [len(numeric)])])
        model = make_pipeline(transform, LogisticRegression(class_weight="balanced", max_iter=3000, random_state=20260808))
        model.fit(x_train, labels[train]); predictions[test] = model.predict(x_test); probabilities[test] = model.predict_proba(x_test)[:, 1]
        fold_labels = labels[test]
        fold_predictions = predictions[test]
        folds.append({
            f"heldout_{split_field}": heldout,
            "groups": len(test),
            "nuisances": int(fold_labels.sum()),
            "predicted_nuisance": int(fold_predictions.sum()),
            "nuisance_precision": float(precision_score(fold_labels, fold_predictions, zero_division=0)),
            "nuisance_recall": float(recall_score(fold_labels, fold_predictions, zero_division=0)),
            "nuisance_average_precision": float(average_precision_score(fold_labels, probabilities[test])),
        })
    return {
        "groups": len(rows), "anchors": int((labels == 0).sum()), "nuisances": int(labels.sum()),
        "nuisance_prevalence": float(labels.mean()), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "nuisance_recall": float(recall_score(labels, predictions, zero_division=0)),
        "nuisance_precision": float(precision_score(labels, predictions, zero_division=0)),
        "nuisance_average_precision": float(average_precision_score(labels, probabilities)), "folds": folds,
    }


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--raw-effects", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    rows = aggregate(args.raw_effects)
    tasks = sorted({row["task_id"] for row in rows})
    result = {
        "stage": "multitask development; leave-one-whole-task-out, not independent confirmation",
        "task_ids": tasks,
        "effect_threshold_reused_from_v8": EFFECT_THRESHOLD, "label_q_rel_threshold": 0.05,
        "attention_only": evaluate(rows, ATTENTION, "task_id"),
        "attention_plus_region": evaluate(rows, ATTENTION + REGION, "task_id"),
        "attention_region_plus_unsigned_magnitude": evaluate(rows, ATTENTION + REGION + MAGNITUDE, "task_id"),
        "gate": {"nuisance_precision_min": .2, "nuisance_average_precision_min": .2},
        "limits": ["five tasks from one LIBERO suite", "simulator transition replay mismatch; stored states are authoritative", "I_G features require one masked velocity probe"],
    }
    primary = result["attention_plus_region"]
    result["cheap_region_gate_pass"] = primary["nuisance_precision"] >= .2 and primary["nuisance_average_precision"] >= .2
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
