#!/usr/bin/env python3
"""Predict frozen-model anchor/nuisance sign with repeated-flow stability features."""

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
BASE_NUMERIC = [
    "late_half_action_to_context_attention", "raw_last_expert_layer_attention", "embedding_norm",
    "relative_magnitude_I", "velocity_delta_cosine", "v_norm_ratio", "visual_token_position",
]
STABILITY_NUMERIC = [
    "q_median", "q_iqr", "q_negative_fraction", "q_positive_fraction", "q_sign_stability",
    "i_median", "i_iqr", "i_max", "attention_std", "camera_q_disagreement",
]
NO_SIGNED_PROBE_NUMERIC = [
    "i_median", "i_iqr", "i_max", "attention_std", "camera_attention_disagreement",
]


def aggregate(path: Path) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with path.open() as stream:
        for row in map(json.loads, stream):
            if row["modality"] != "visual" or row["replacement_type"] != "position_conditioned_modality_mean":
                continue
            groups[(row["suite"], row["task_id"], row["state_index"], row["group_id"])].append(row)
    raw = []
    for (suite, task_id, state_index, group_id), rows in groups.items():
        q = np.asarray([float(r["Q_rel"]) for r in rows])
        effect = np.asarray([float(r["relative_magnitude_I"]) for r in rows])
        attention = np.asarray([float(r["late_half_action_to_context_attention"]) for r in rows])
        median = lambda key: float(np.median([float(r[key]) for r in rows]))
        q_sign = np.sign(q[np.abs(q) > 0])
        raw.append({
            "suite": suite, "task_id": task_id, "state_index": state_index, "group_id": group_id,
            "camera_id": rows[0]["camera_id"], "visual_token_index": int(median("visual_token_index")),
            "late_half_action_to_context_attention": median("late_half_action_to_context_attention"),
            "raw_last_expert_layer_attention": median("raw_last_expert_layer_attention"),
            "embedding_norm": median("embedding_norm"), "relative_magnitude_I": median("relative_magnitude_I"),
            "velocity_delta_cosine": median("velocity_delta_cosine"),
            "v_norm_ratio": median("v_neg_norm") / (median("v_pos_norm") + 1e-12),
            "visual_token_position": median("visual_token_index") / 63.0,
            "q_median": float(np.median(q)), "q_iqr": float(np.quantile(q, .75) - np.quantile(q, .25)),
            "q_negative_fraction": float(np.mean(q < -0.05)), "q_positive_fraction": float(np.mean(q > 0.05)),
            "q_sign_stability": float(max(np.mean(q_sign < 0), np.mean(q_sign > 0))) if len(q_sign) else 0.0,
            "i_median": float(np.median(effect)), "i_iqr": float(np.quantile(effect, .75) - np.quantile(effect, .25)),
            "i_max": float(np.max(effect)), "attention_std": float(np.std(attention)),
            "label_nuisance": int(np.median(q) < -0.05) if np.median(effect) > EFFECT_THRESHOLD and abs(np.median(q)) > 0.05 else -1,
        })
    by_position: dict[tuple, list[dict]] = defaultdict(list)
    for row in raw:
        by_position[(row["suite"], row["task_id"], row["state_index"], row["visual_token_index"])].append(row)
    for row in raw:
        peers = [x["q_median"] for x in by_position[(row["suite"], row["task_id"], row["state_index"], row["visual_token_index"])] if x["camera_id"] != row["camera_id"]]
        row["camera_q_disagreement"] = abs(row["q_median"] - float(np.median(peers))) if peers else 0.0
        attention_peers = [x["late_half_action_to_context_attention"] for x in by_position[(row["suite"], row["task_id"], row["state_index"], row["visual_token_index"])] if x["camera_id"] != row["camera_id"]]
        row["camera_attention_disagreement"] = abs(row["late_half_action_to_context_attention"] - float(np.median(attention_peers))) if attention_peers else 0.0
    return [row for row in raw if row["label_nuisance"] >= 0]


def matrix(rows: list[dict], numeric: list[str]) -> np.ndarray:
    return np.asarray([[row[name] for name in numeric] + [row["camera_id"]] for row in rows], dtype=object)


def evaluate(rows: list[dict], numeric: list[str]) -> dict:
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows}); predictions = np.zeros(len(rows), dtype=int); probabilities = np.zeros(len(rows))
    folds = []
    for task in tasks:
        test = [i for i, r in enumerate(rows) if (r["suite"], r["task_id"]) == task]; train = [i for i in range(len(rows)) if i not in test]
        transformer = ColumnTransformer([("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(range(len(numeric)))), ("camera", OneHotEncoder(handle_unknown="ignore"), [len(numeric)])])
        model = make_pipeline(transformer, LogisticRegression(class_weight="balanced", max_iter=3000, random_state=20260808))
        model.fit(matrix([rows[i] for i in train], numeric), [rows[i]["label_nuisance"] for i in train])
        predictions[test] = model.predict(matrix([rows[i] for i in test], numeric)); probabilities[test] = model.predict_proba(matrix([rows[i] for i in test], numeric))[:, 1]
        folds.append({"task": task, "groups": len(test), "nuisances": int(sum(rows[i]["label_nuisance"] for i in test)), "predicted_nuisance": int(predictions[test].sum())})
    labels = np.asarray([r["label_nuisance"] for r in rows])
    return {"groups": len(rows), "anchors": int((labels == 0).sum()), "nuisances": int(labels.sum()), "nuisance_prevalence": float(labels.mean()), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)), "nuisance_recall": float(recall_score(labels, predictions, zero_division=0)), "nuisance_precision": float(precision_score(labels, predictions, zero_division=0)), "nuisance_average_precision": float(average_precision_score(labels, probabilities)), "predicted_nuisance": int(predictions.sum()), "task_folds": folds}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--raw-effects", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    rows = aggregate(args.raw_effects); results = {"stage": "offline_sign_classifier_v2", "repetitions_required": "3 tau x 2 noise", "groups": len(rows), "full_stability_features": evaluate(rows, BASE_NUMERIC + STABILITY_NUMERIC), "no_signed_probe_features": evaluate(rows, BASE_NUMERIC + NO_SIGNED_PROBE_NUMERIC), "attention_only": evaluate(rows, ["late_half_action_to_context_attention"]), "gate": {"balanced_accuracy_gt": .6, "nuisance_recall_gte": .5, "nuisance_precision_gte": .2, "nuisance_average_precision_gte": .2}}
    full = results["full_stability_features"]; results["gate_pass"] = full["balanced_accuracy"] > .6 and full["nuisance_recall"] >= .5 and full["nuisance_precision"] >= .2 and full["nuisance_average_precision"] >= .2
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n"); print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__": main()
