from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "lerobot/src"))

from research.coreact_selective_cfg.features import (
    GEOMETRY_LOCAL_FEATURES,
    assert_deployable_feature_names,
)


PRIMARY_BRANCH = "W4_half_last_2"
LOCAL_FEATURES = tuple(GEOMETRY_LOCAL_FEATURES)
MODEL_SPEC = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "random_state": 20260811,
}


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**MODEL_SPEC)


def platt_fit(raw_probability: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=20260811)
    model.fit(logits, labels.astype(int))
    return model


def platt_apply(model: LogisticRegression, raw_probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def choose_threshold(inner: pd.DataFrame) -> dict:
    candidates = []
    for quantile in (0.50, 0.55, 0.60, 0.65, 0.70):
        threshold = float(inner.probability.quantile(quantile))
        task_rows = []
        for task_id, group in inner.groupby("task_id"):
            selected = group.probability.to_numpy() >= threshold
            coverage = float(selected.mean())
            precision = float(group.ev_positive.to_numpy()[selected].mean()) if selected.any() else 0.0
            task_rows.append({"task_id": int(task_id), "coverage": coverage, "precision": precision})
        if min(row["coverage"] for row in task_rows) < 0.10:
            continue
        candidates.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "coverage": float((inner.probability >= threshold).mean()),
                "macro_precision": float(np.mean([row["precision"] for row in task_rows])),
                "minimum_task_precision": float(min(row["precision"] for row in task_rows)),
                "task_rows": task_rows,
            }
        )
    if not candidates:
        raise RuntimeError("no frozen threshold candidate retains 10% coverage on every Spatial task")
    chosen = max(
        candidates,
        key=lambda row: (
            row["macro_precision"],
            row["minimum_task_precision"],
            -abs(row["coverage"] - 0.40),
        ),
    )
    return {"chosen": chosen, "candidates": candidates}


def freeze_from_spatial(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    assert_deployable_feature_names(LOCAL_FEATURES)
    branch = frame[frame.branch == PRIMARY_BRANCH].copy()
    if set(branch.task_id.unique()) != set(range(10)) or branch.state_id.nunique() != 500:
        raise RuntimeError("Spatial development source is incomplete")
    out_of_task = []
    for heldout_task in range(10):
        train = branch[branch.task_id != heldout_task]
        test = branch[branch.task_id == heldout_task].copy()
        model = make_model()
        model.fit(train[list(LOCAL_FEATURES)].to_numpy(), train.ev_positive.astype(int).to_numpy())
        test["raw_probability"] = model.predict_proba(test[list(LOCAL_FEATURES)].to_numpy())[:, 1]
        out_of_task.append(test)
    predictions = pd.concat(out_of_task, ignore_index=True)
    calibrator = platt_fit(predictions.raw_probability.to_numpy(), predictions.ev_positive.to_numpy())
    predictions["probability"] = platt_apply(calibrator, predictions.raw_probability.to_numpy())
    threshold = choose_threshold(predictions)
    selected_threshold = threshold["chosen"]["threshold"]
    predictions["selected"] = predictions.probability >= selected_threshold

    final_model = make_model()
    final_model.fit(
        branch[list(LOCAL_FEATURES)].to_numpy(), branch.ev_positive.astype(int).to_numpy()
    )
    task_rows = []
    for task_id, group in predictions.groupby("task_id"):
        selected = group.selected.to_numpy()
        prevalence = float(group.ev_positive.mean())
        precision = float(group.ev_positive.to_numpy()[selected].mean())
        task_rows.append(
            {
                "task_id": int(task_id),
                "coverage": float(selected.mean()),
                "precision": precision,
                "prevalence": prevalence,
                "lift": precision - prevalence,
            }
        )
    metadata = {
        "source": "coreact_selective_cfg_ev_validity_phase1_v1_20260811_203336",
        "source_suite": "libero_spatial",
        "source_tasks": list(range(10)),
        "branch": PRIMARY_BRANCH,
        "features": list(LOCAL_FEATURES),
        "model_spec": MODEL_SPEC,
        "calibration": "Platt logistic on 10-task out-of-task predictions",
        "threshold_selection": threshold,
        "frozen_threshold": selected_threshold,
        "development_out_of_task": {
            "mean_coverage": float(predictions.selected.mean()),
            "macro_precision": float(np.mean([row["precision"] for row in task_rows])),
            "tasks_positive_lift": int(sum(row["lift"] > 0 for row in task_rows)),
            "task_rows": task_rows,
        },
        "object_data_used": False,
    }
    bundle = {
        "model": final_model,
        "calibrator": calibrator,
        "threshold": selected_threshold,
        "features": LOCAL_FEATURES,
        "branch": PRIMARY_BRANCH,
        "model_spec": MODEL_SPEC,
    }
    return bundle, predictions, metadata
