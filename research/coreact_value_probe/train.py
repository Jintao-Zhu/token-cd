#!/usr/bin/env python3
"""Fit pre-registered frozen-latent value probes after capture completeness is proven."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_TASKS = {0, 1, 2, 5, 6, 7, 9}
VALIDATION_TASKS = {3}
TEST_TASKS = {4}
BOOTSTRAP_REPLICATES = 2000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if mask.any():
            value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return value


def safe_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    result = {"n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()) if len(y) else None}
    if len(np.unique(y)) < 2:
        result.update({"auroc": None, "ap": None, "brier": brier_score_loss(y, p) if len(y) else None, "ece10": ece(y, p) if len(y) else None})
    else:
        result.update({"auroc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "ece10": ece(y, p)})
    return result


def bootstrap_auc(rows: pd.DataFrame, replicates: int = BOOTSTRAP_REPLICATES, seed: int = 918273) -> dict:
    episodes = rows["capture_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(episodes, size=len(episodes), replace=True)
        sample = pd.concat([rows[rows.capture_id == episode] for episode in sampled], ignore_index=True)
        y, p = sample.label.to_numpy(dtype=int), sample.prob.to_numpy(dtype=float)
        if len(np.unique(y)) == 2:
            values.append(roc_auc_score(y, p))
    return {"replicates": replicates, "valid_replicates": len(values), "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else [None, None]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    manifest = read_jsonl(artifact / "capture_manifest.jsonl")
    audits = [json.loads((artifact / "captures" / f"{row['capture_id']}.json").read_text()) for row in manifest if (artifact / "captures" / f"{row['capture_id']}.json").exists()]
    if len(audits) != len(manifest):
        raise RuntimeError(f"capture completeness gate failed: {len(audits)}/{len(manifest)}")
    records = []
    for spec, audit in zip(manifest, audits, strict=False):
        if spec["capture_id"] != audit["capture_id"] or audit["status"] != "complete":
            raise RuntimeError(f"capture audit mismatch for {spec['capture_id']}")
        data = np.load(artifact / "captures" / f"{spec['capture_id']}.npz")
        reps, phases, labels = data["representation"], data["phase"], data["label"].astype(int)
        if len(reps) != audit["states"] or not np.isfinite(reps).all() or not np.all(labels == spec["expected_success"]):
            raise RuntimeError(f"capture numeric/label gate failed for {spec['capture_id']}")
        for index in range(len(reps)):
            records.append({"capture_id": spec["capture_id"], "task_id": spec["task_id"], "split": spec["split"], "phase": str(phases[index]), "label": int(labels[index]), "X": reps[index].astype(np.float64)})
    frame = pd.DataFrame(records)
    frame.to_pickle(artifact / "probe_rows.pkl")
    x_train = np.stack(frame.loc[frame.task_id.isin(TRAIN_TASKS), "X"])
    y_train = frame.loc[frame.task_id.isin(TRAIN_TASKS), "label"].to_numpy()
    train_weights = frame.loc[frame.task_id.isin(TRAIN_TASKS)].groupby("capture_id")["capture_id"].transform(lambda s: 1.0 / len(s)).to_numpy()
    linear = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=817263))
    scaler = StandardScaler().fit(x_train)
    x_validation = np.stack(frame.loc[frame.task_id.isin(VALIDATION_TASKS), "X"])
    y_validation = frame.loc[frame.task_id.isin(VALIDATION_TASKS), "label"].to_numpy()
    mlp = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=1e-4, max_iter=1, warm_start=True, random_state=817263, early_stopping=False)
    best_mlp, best_loss, stale = None, math.inf, 0
    for _epoch in range(100):
        mlp.fit(scaler.transform(x_train), y_train, sample_weight=train_weights)
        validation_loss = log_loss(y_validation, mlp.predict_proba(scaler.transform(x_validation))[:, 1], labels=[0, 1])
        if validation_loss < best_loss - 1e-6:
            best_mlp, best_loss, stale = copy.deepcopy(mlp), float(validation_loss), 0
        else:
            stale += 1
        if stale >= 15:
            break
    if best_mlp is None:
        raise RuntimeError("MLP validation selection failed")
    mlp_pipeline = make_pipeline(scaler, best_mlp)
    models = {"linear": linear, "mlp": mlp_pipeline}
    summary = {"capture_count": len(audits), "state_count": len(frame), "models": {}, "phase_support": {}}
    for phase in sorted(frame.phase.unique()):
        subset = frame[frame.phase == phase]
        summary["phase_support"][phase] = {"episodes": int(subset.capture_id.nunique()), "positive_episodes": int(subset.groupby("capture_id").label.first().sum()), "negative_episodes": int((1 - subset.groupby("capture_id").label.first()).sum())}
    for name, model in models.items():
        if name == "linear":
            model.fit(x_train, y_train, logisticregression__sample_weight=train_weights)
        frame[f"{name}_probability"] = model.predict_proba(np.stack(frame.X))[:, 1]
        model_summary = {}
        for split, tasks in (("train", TRAIN_TASKS), ("validation", VALIDATION_TASKS), ("heldout_test", TEST_TASKS)):
            subset = frame[frame.task_id.isin(tasks)].copy()
            probs = model.predict_proba(np.stack(subset.X))[:, 1]
            model_summary[split] = safe_metrics(subset.label.to_numpy(), probs)
            subset["prob"] = probs
            model_summary[split]["task_metrics"] = {str(task): safe_metrics(g.label.to_numpy(), g.prob.to_numpy()) for task, g in subset.groupby("task_id")}
            model_summary[split]["phase_metrics"] = {str(phase): safe_metrics(g.label.to_numpy(), g.prob.to_numpy()) for phase, g in subset.groupby("phase")}
            if split == "heldout_test":
                model_summary[split]["bootstrap"] = bootstrap_auc(subset)
        # The four pools have the same hidden width. Fit a static instruction-only control.
        hidden_width = x_train.shape[1] // 4
        language_slice = slice(2 * hidden_width, 3 * hidden_width)
        language_control = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=817263))
        language_control.fit(x_train[:, language_slice], y_train, logisticregression__sample_weight=train_weights)
        test = frame[frame.task_id.isin(TEST_TASKS)]
        contextual_language_probs = language_control.predict_proba(np.stack(test.X)[:, language_slice])[:, 1]
        model_summary["heldout_test"]["contextual_language_position_pool_diagnostic"] = safe_metrics(test.label.to_numpy(), contextual_language_probs)
        # All states of held-out task 4 share one fixed instruction. A true static
        # instruction/task-identity control is therefore constant within this test.
        static_probability = float(frame.loc[frame.task_id.isin(TRAIN_TASKS)].groupby("capture_id").label.first().mean())
        static_probs = np.full(len(test), static_probability, dtype=np.float64)
        model_summary["heldout_test"]["language_only_control"] = safe_metrics(test.label.to_numpy(), static_probs)
        summary["models"][name] = model_summary
        import joblib
        joblib.dump(model, artifact / f"{name}_probe.joblib")
    frame.drop(columns=["X"]).to_csv(artifact / "probe_predictions.csv", index=False)
    primary = summary["models"]["linear"]["heldout_test"]
    control = primary["language_only_control"]
    ci_low = primary["bootstrap"]["ci95"][0]
    checks = {
        "auroc_at_least_0_65": primary["auroc"] is not None and primary["auroc"] >= 0.65,
        "auroc_ci_lower_above_0_50": ci_low is not None and ci_low > 0.50,
        "ap_lift_at_least_0_10": primary["ap"] is not None and primary["ap"] - primary["prevalence"] >= 0.10,
        "stateful_minus_language_auroc_at_least_0_05": primary["auroc"] is not None and control["auroc"] is not None and primary["auroc"] - control["auroc"] >= 0.05,
    }
    summary["qualification"] = {"primary_model": "linear", "checks": checks, "pass": all(checks.values())}
    decision = {
        "decision": "QUALIFIED_FOR_TOKEN_CAUSAL_CALIBRATION" if all(checks.values()) else "STOP_SINGLE_STATE_VALUE_PROBE_ROUTE",
        "reason": "All pre-registered linear-probe gates passed." if all(checks.values()) else "At least one pre-registered linear-probe gate failed; MLP cannot override the primary gate.",
        "guidance_permitted": False,
        "next_stage": "lock a separate new-state signed token intervention protocol" if all(checks.values()) else "consider a separately pre-registered trajectory-conditioned critic",
    }
    (artifact / "probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
