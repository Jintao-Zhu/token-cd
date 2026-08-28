#!/usr/bin/env python3
"""Evaluate whether runtime-available features predict anchor versus nuisance sign."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EFFECT_THRESHOLD = 0.0015907290887758336
NUMERIC_FULL = [
    "late_half_action_to_context_attention",
    "raw_last_expert_layer_attention",
    "embedding_norm",
    "relative_magnitude_I",
    "velocity_delta_cosine",
    "v_norm_ratio",
    "visual_token_position",
]


def aggregate(path: Path) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["modality"] != "visual" or row["replacement_type"] != "position_conditioned_modality_mean":
                continue
            key = (row["suite"], row["task_id"], row["state_index"], row["group_id"])
            groups[key].append(row)
    output = []
    median_fields = [
        "Q_rel", "relative_magnitude_I", "late_half_action_to_context_attention",
        "raw_last_expert_layer_attention", "embedding_norm", "velocity_delta_cosine",
        "v_pos_norm", "v_neg_norm", "visual_token_index",
    ]
    for (suite, task_id, state_index, group_id), rows in groups.items():
        values = {name: float(np.median([row[name] for row in rows])) for name in median_fields}
        if values["relative_magnitude_I"] <= EFFECT_THRESHOLD or abs(values["Q_rel"]) <= 0.05:
            continue
        output.append({
            "suite": suite, "task_id": task_id, "state_index": state_index, "group_id": group_id,
            "camera_id": rows[0]["camera_id"], "label_nuisance": int(values["Q_rel"] < -0.05),
            "v_norm_ratio": values["v_neg_norm"] / (values["v_pos_norm"] + 1e-12),
            "visual_token_position": values["visual_token_index"] / 63.0, **values,
        })
    return output


def matrix(rows: list[dict], numeric: list[str]) -> np.ndarray:
    return np.asarray([[row[name] for name in numeric] + [row["camera_id"]] for row in rows], dtype=object)


def evaluate(rows: list[dict], numeric: list[str]) -> dict:
    tasks = sorted({(row["suite"], row["task_id"]) for row in rows})
    predictions = np.zeros(len(rows), dtype=int)
    probabilities = np.zeros(len(rows), dtype=float)
    fold_rows = []
    for task in tasks:
        train_indices = [i for i, row in enumerate(rows) if (row["suite"], row["task_id"]) != task]
        test_indices = [i for i, row in enumerate(rows) if (row["suite"], row["task_id"]) == task]
        train, test = [rows[i] for i in train_indices], [rows[i] for i in test_indices]
        transformer = ColumnTransformer(
            [
                ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(range(len(numeric)))),
                ("camera", OneHotEncoder(handle_unknown="ignore"), [len(numeric)]),
            ]
        )
        model = make_pipeline(
            transformer,
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260807),
        )
        y_train = np.asarray([row["label_nuisance"] for row in train])
        y_test = np.asarray([row["label_nuisance"] for row in test])
        model.fit(matrix(train, numeric), y_train)
        predictions[test_indices] = model.predict(matrix(test, numeric))
        probabilities[test_indices] = model.predict_proba(matrix(test, numeric))[:, 1]
        fold_rows.append({
            "suite": task[0], "task_id": task[1], "groups": len(test),
            "nuisance_groups": int(y_test.sum()),
            "predicted_nuisance": int(predictions[test_indices].sum()),
        })
    labels = np.asarray([row["label_nuisance"] for row in rows])
    return {
        "groups": len(rows), "anchors": int((labels == 0).sum()), "nuisances": int(labels.sum()),
        "nuisance_prevalence": float(labels.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "nuisance_recall": float(recall_score(labels, predictions, zero_division=0)),
        "nuisance_precision": float(precision_score(labels, predictions, zero_division=0)),
        "nuisance_average_precision": float(average_precision_score(labels, probabilities)),
        "predicted_nuisance": int(predictions.sum()), "task_folds": fold_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-effects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = aggregate(args.raw_effects)
    results = {
        "stage": "post_mask_diagnostic_revision_development",
        "split": "leave_one_suite_task_out_over_offline_v8_heldout_effects",
        "labels": f"I>{EFFECT_THRESHOLD} and Q_rel outside [-0.05,0.05]",
        "attention_only": evaluate(rows, ["late_half_action_to_context_attention"]),
        "full_runtime_features": evaluate(rows, NUMERIC_FULL),
        "preregistered_gate": {
            "balanced_accuracy_gt": 0.6, "nuisance_recall_gte": 0.5,
            "nuisance_precision_gte": 0.2, "nuisance_average_precision_gte": 0.2,
        },
    }
    full = results["full_runtime_features"]
    results["gate_pass"] = (
        full["balanced_accuracy"] > 0.6 and full["nuisance_recall"] >= 0.5
        and full["nuisance_precision"] >= 0.2 and full["nuisance_average_precision"] >= 0.2
    )
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
