#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from research.coreact_selective_cfg.features import assert_deployable_feature_names


RANDOM_SEED = 20260811
BOOTSTRAP_REPLICATES = 5000
PRIMARY_BRANCH = "W4_half_last_2"


def fit_feature_transform(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    language_embeddings: dict[int, np.ndarray],
    include_language: bool,
) -> tuple[np.ndarray, np.ndarray]:
    assert_deployable_feature_names(feature_names)
    train_x = train[feature_names].to_numpy(dtype=np.float64)
    test_x = test[feature_names].to_numpy(dtype=np.float64)
    if not include_language:
        return train_x, test_x
    train_tasks = sorted(int(value) for value in train.task_id.unique())
    language_matrix = np.stack([language_embeddings[task] for task in train_tasks])
    components = min(8, len(train_tasks) - 1, language_matrix.shape[1])
    pca = PCA(n_components=components, random_state=RANDOM_SEED)
    pca.fit(language_matrix)
    train_language = pca.transform(
        np.stack([language_embeddings[int(task)] for task in train.task_id])
    )
    test_language = pca.transform(
        np.stack([language_embeddings[int(task)] for task in test.task_id])
    )
    return np.column_stack([train_x, train_language]), np.column_stack([test_x, test_language])


def make_model(spec: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=spec["learning_rate"],
        max_iter=spec["max_iter"],
        max_leaf_nodes=spec["max_leaf_nodes"],
        min_samples_leaf=spec["min_samples_leaf"],
        l2_regularization=spec["l2_regularization"],
        random_state=spec["random_state"],
    )


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    language_embeddings: dict[int, np.ndarray],
    include_language: bool,
    model_spec: dict,
) -> np.ndarray:
    train_x, test_x = fit_feature_transform(
        train, test, feature_names, language_embeddings, include_language
    )
    model = make_model(model_spec)
    model.fit(train_x, train.ev_positive.astype(int).to_numpy())
    return model.predict_proba(test_x)[:, 1]


def platt_fit(raw_probability: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
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
        raise RuntimeError("no preregistered threshold retains 10% coverage on every inner task")
    chosen = max(
        candidates,
        key=lambda row: (
            row["macro_precision"],
            row["minimum_task_precision"],
            -abs(row["coverage"] - 0.40),
        ),
    )
    return {"chosen": chosen, "candidates": candidates}


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= lower) & (
            (probability <= upper) if upper == 1.0 else (probability < upper)
        )
        if mask.any():
            value += mask.sum() / total * abs(probability[mask].mean() - labels[mask].mean())
    return float(value)


def task_metrics(frame: pd.DataFrame, threshold: float, train_prevalence: float) -> dict:
    labels = frame.ev_positive.astype(int).to_numpy()
    probability = frame.probability.to_numpy()
    selected = probability >= threshold
    precision = float(labels[selected].mean()) if selected.any() else 0.0
    prevalence = float(labels.mean())
    return {
        "points": len(frame),
        "states": int(frame.state_id.nunique()),
        "ev_positive_prevalence": prevalence,
        "coverage": float(selected.mean()),
        "precision": precision,
        "precision_lift": precision - prevalence,
        "recall": float(labels[selected].sum() / max(labels.sum(), 1)),
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "constant_train_prevalence_brier": float(np.mean((labels - train_prevalence) ** 2)),
        "ece10": expected_calibration_error(labels, probability),
    }


def state_bootstrap(frame: pd.DataFrame, threshold: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    states = np.asarray(sorted(frame.state_id.unique()))
    grouped = {state: frame[frame.state_id == state] for state in states}
    precision_values = []
    coverage_values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(states, size=len(states), replace=True)
        replica = pd.concat([grouped[state] for state in sampled], ignore_index=True)
        selected = replica.probability.to_numpy() >= threshold
        coverage_values.append(float(selected.mean()))
        if selected.any():
            precision_values.append(float(replica.ev_positive.to_numpy()[selected].mean()))
        else:
            precision_values.append(0.0)
    return {
        "precision_ci_low": float(np.quantile(precision_values, 0.025)),
        "precision_ci_high": float(np.quantile(precision_values, 0.975)),
        "coverage_ci_low": float(np.quantile(coverage_values, 0.025)),
        "coverage_ci_high": float(np.quantile(coverage_values, 0.975)),
    }


def nested_outer_predictions(
    branch_frame: pd.DataFrame,
    feature_names: list[str],
    language_embeddings: dict[int, np.ndarray],
    include_language: bool,
    model_spec: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    predictions = []
    fold_metadata = []
    tasks = sorted(int(value) for value in branch_frame.task_id.unique())
    for outer_task in tasks:
        outer_train = branch_frame[branch_frame.task_id != outer_task].copy()
        outer_test = branch_frame[branch_frame.task_id == outer_task].copy()
        inner_predictions = []
        for inner_task in sorted(int(value) for value in outer_train.task_id.unique()):
            inner_train = outer_train[outer_train.task_id != inner_task]
            inner_test = outer_train[outer_train.task_id == inner_task].copy()
            inner_test["raw_probability"] = fit_predict(
                inner_train,
                inner_test,
                feature_names,
                language_embeddings,
                include_language,
                model_spec,
            )
            inner_predictions.append(inner_test[["task_id", "ev_positive", "raw_probability"]])
        inner = pd.concat(inner_predictions, ignore_index=True)
        calibrator = platt_fit(inner.raw_probability.to_numpy(), inner.ev_positive.to_numpy())
        inner["probability"] = platt_apply(calibrator, inner.raw_probability.to_numpy())
        threshold_metadata = choose_threshold(inner)
        raw_outer = fit_predict(
            outer_train,
            outer_test,
            feature_names,
            language_embeddings,
            include_language,
            model_spec,
        )
        outer_test["raw_probability"] = raw_outer
        outer_test["probability"] = platt_apply(calibrator, raw_outer)
        outer_test["threshold"] = threshold_metadata["chosen"]["threshold"]
        outer_test["selected"] = outer_test.probability >= outer_test.threshold
        predictions.append(outer_test)
        fold_metadata.append(
            {
                "outer_task": outer_task,
                "train_tasks": sorted(int(value) for value in outer_train.task_id.unique()),
                "test_states": int(outer_test.state_id.nunique()),
                "train_prevalence": float(outer_train.ev_positive.mean()),
                "threshold": threshold_metadata,
                "platt_coefficient": float(calibrator.coef_[0, 0]),
                "platt_intercept": float(calibrator.intercept_[0]),
            }
        )
    return pd.concat(predictions, ignore_index=True), fold_metadata


def pooled_task_bootstrap(task_rows: pd.DataFrame, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    records = task_rows.to_dict(orient="records")
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [records[index] for index in rng.integers(0, len(records), size=len(records))]
        values.append(float(np.mean([row["precision"] for row in sampled])))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def make_plot(primary: pd.DataFrame, artifact: Path) -> None:
    ordered = primary.sort_values("task_id")
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(ordered.task_id - 0.18, ordered.precision, width=0.36, label="gated precision")
    axes[0].bar(
        ordered.task_id + 0.18,
        ordered.ev_positive_prevalence,
        width=0.36,
        label="ungated EV prevalence",
    )
    axes[0].axhline(0.70, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("EV-positive fraction")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].bar(ordered.task_id, ordered.coverage, width=0.6, color="#4C78A8")
    axes[1].axhspan(0.30, 0.50, color="#59A14F", alpha=0.15)
    axes[1].set_ylabel("Guidance coverage")
    axes[1].set_xlabel("Held-out LIBERO-Spatial task")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(ordered.task_id)
    figure.tight_layout()
    figure.savefig(artifact / "plots/primary_loto_precision_coverage.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    if not (artifact / "status/capture.complete").exists():
        raise RuntimeError("capture must pass before analysis")
    frame = pd.read_parquet(artifact / "point_level_metrics.parquet")
    language_embeddings = {
        int(task): np.asarray(value, dtype=np.float64)
        for task, value in json.loads((artifact / "language_embeddings.json").read_text()).items()
    }

    variants = protocol["features"]["variants"]
    configs = [
        (PRIMARY_BRANCH, "geometry_local", False),
        (PRIMARY_BRANCH, "geometry_consensus", False),
        (PRIMARY_BRANCH, "geometry_consensus_language", True),
        ("W1_skip_last_1", "geometry_consensus", False),
        ("W2_skip_last_2", "geometry_consensus", False),
        ("W3_half_last_1", "geometry_consensus", False),
    ]
    prediction_frames = []
    task_rows = []
    fold_records = []
    for config_index, (branch, variant, include_language) in enumerate(configs):
        branch_frame = frame[frame.branch == branch].copy()
        feature_names = variants[variant]
        predictions, folds = nested_outer_predictions(
            branch_frame,
            feature_names,
            language_embeddings,
            include_language,
            protocol["validation"]["model"],
        )
        predictions["variant"] = variant
        prediction_frames.append(predictions)
        for fold in folds:
            fold_records.append({"branch": branch, "variant": variant, **fold})
            task_id = fold["outer_task"]
            group = predictions[predictions.task_id == task_id]
            threshold = float(group.threshold.iloc[0])
            metrics = task_metrics(group, threshold, fold["train_prevalence"])
            metrics.update(state_bootstrap(group, threshold, RANDOM_SEED + config_index * 100 + task_id))
            task_rows.append(
                {
                    "branch": branch,
                    "variant": variant,
                    "task_id": task_id,
                    "threshold": threshold,
                    **metrics,
                    "brier_better_than_constant": metrics["brier"]
                    < metrics["constant_train_prevalence_brier"],
                }
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(artifact / "loto_point_predictions.parquet", index=False)
    task = pd.DataFrame(task_rows).sort_values(["branch", "variant", "task_id"])
    task.to_csv(artifact / "loto_task_summary.csv", index=False)
    (artifact / "fold_metadata.json").write_text(
        json.dumps(fold_records, indent=2, sort_keys=True) + "\n"
    )

    aggregate_rows = []
    for (branch, variant), group in task.groupby(["branch", "variant"]):
        ci_low, ci_high = pooled_task_bootstrap(
            group, RANDOM_SEED + len(aggregate_rows) * 1000
        )
        aggregate_rows.append(
            {
                "branch": branch,
                "variant": variant,
                "mean_coverage": float(group.coverage.mean()),
                "minimum_task_coverage": float(group.coverage.min()),
                "macro_precision": float(group.precision.mean()),
                "macro_precision_task_bootstrap_ci_low": ci_low,
                "macro_precision_task_bootstrap_ci_high": ci_high,
                "tasks_precision_at_least_0_70": int((group.precision >= 0.70).sum()),
                "tasks_precision_above_prevalence": int((group.precision_lift > 0).sum()),
                "macro_precision_lift": float(group.precision_lift.mean()),
                "macro_ece10": float(group.ece10.mean()),
                "tasks_brier_better_than_constant": int(group.brier_better_than_constant.sum()),
                "macro_auroc": float(group.auroc.mean()),
                "macro_average_precision": float(group.average_precision.mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["branch", "variant"])
    aggregate.to_csv(artifact / "aggregate_summary.csv", index=False)

    primary_task = task[
        (task.branch == protocol["go_gate"]["decision_branch"])
        & (task.variant == protocol["go_gate"]["decision_variant"])
    ].copy()
    primary = aggregate[
        (aggregate.branch == protocol["go_gate"]["decision_branch"])
        & (aggregate.variant == protocol["go_gate"]["decision_variant"])
    ].iloc[0]
    gate = protocol["go_gate"]
    checks = {
        "mean_coverage": gate["mean_heldout_coverage_range_inclusive"][0]
        <= primary.mean_coverage
        <= gate["mean_heldout_coverage_range_inclusive"][1],
        "minimum_task_coverage": primary.minimum_task_coverage
        >= gate["minimum_each_task_coverage"],
        "macro_precision": primary.macro_precision >= gate["macro_precision_min"],
        "tasks_precision": primary.tasks_precision_at_least_0_70
        >= gate["tasks_precision_at_least_0_70_min"],
        "tasks_lift": primary.tasks_precision_above_prevalence
        >= gate["tasks_precision_above_ungated_prevalence_min"],
        "precision_ci": primary.macro_precision_task_bootstrap_ci_low
        >= gate["task_bootstrap_pooled_precision_ci_lower_min"],
        "calibration_ece": primary.macro_ece10 <= gate["macro_ece_max"],
        "calibration_brier": primary.tasks_brier_better_than_constant
        >= gate["tasks_brier_better_than_constant_prevalence_min"],
    }
    decision_name = (
        protocol["decision_rules"]["pass"]
        if all(checks.values())
        else protocol["decision_rules"]["fail"]
    )

    expected_rows = protocol["scope"]["point_rows"]
    integrity_checks = {
        "point_rows_complete": len(frame) == expected_rows,
        "unique_point_ids": frame.point_id.nunique() == expected_rows,
        "matched_points_complete": frame.matched_point_id.nunique()
        == protocol["scope"]["matched_points"],
        "ten_tasks": set(frame.task_id.unique()) == set(range(10)),
        "four_branches": frame.branch.nunique() == 4,
        "fifty_states_each_task": bool((frame.groupby("task_id").state_id.nunique() == 50).all()),
        "same_condition_hashes": bool(
            (
                frame.groupby("matched_point_id")[[
                    "prefix_sha256",
                    "action_sha256",
                    "noise_sha256",
                    "x_t_sha256",
                    "target_sha256",
                ]].nunique()
                == 1
            ).all().all()
        ),
        "all_numeric_finite": bool(
            frame.select_dtypes(include=[np.number]).apply(np.isfinite).all().all()
        ),
        "feature_contract_target_free": True,
        "outer_predictions_complete": len(predictions) == 90000,
        "every_outer_task_once_per_config": bool(
            (predictions.groupby(["branch", "variant", "task_id"]).state_id.nunique() == 50).all()
        ),
        "language_embeddings_complete": set(language_embeddings) == set(range(10)),
    }
    integrity = {
        "pass": all(integrity_checks.values()),
        "checks": integrity_checks,
        "capture": json.loads((artifact / "capture_summary.json").read_text()),
    }
    (artifact / "integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    if not integrity["pass"]:
        raise RuntimeError(f"integrity failed: {integrity_checks}")

    result = {
        "decision": decision_name,
        "primary_branch": PRIMARY_BRANCH,
        "primary_variant": "geometry_consensus",
        "gate_checks": checks,
        "primary_aggregate": primary.to_dict(),
        "closed_loop_authorized": False,
        "secondary_variants_can_trigger_go": False,
    }
    (artifact / "decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (artifact / "phase1_summary.json").write_text(
        json.dumps(
            {
                **result,
                "aggregate": aggregate.to_dict(orient="records"),
                "primary_tasks": primary_task.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    make_plot(primary_task, artifact)

    report = [
        "# Selective CFG for Flow-VLA: Offline EV-Validity Phase-1",
        "",
        "## Decision",
        "",
        f"`{decision_name}`",
        "",
        "Closed-loop rollout is not authorized by this experiment.",
        "",
        "## Integrity",
        "",
        f"PASS: {protocol['scope']['states']}/500 states, {protocol['scope']['state_noise_units']}/1,500 state-noise units, {protocol['scope']['matched_points']:,} matched points, and {expected_rows:,} branch rows. All features are target-free by contract; EV is used only as the label.",
        "",
        "## Primary leave-one-task-out result",
        "",
        "Primary method: W4, geometry plus W1-W4 multi-branch consensus, no language embedding.",
        "",
        "| Held-out task | EV prevalence | Coverage | Gated precision (95% state CI) | Lift | AUROC | ECE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_task.sort_values("task_id").to_dict(orient="records"):
        report.append(
            f"| {row['task_id']} | {row['ev_positive_prevalence']:.3f} | {row['coverage']:.3f} | "
            f"{row['precision']:.3f} [{row['precision_ci_low']:.3f},{row['precision_ci_high']:.3f}] | "
            f"{row['precision_lift']:+.3f} | {row['auroc']:.3f} | {row['ece10']:.3f} |"
        )
    report += [
        "",
        "## Aggregate and ablations",
        "",
        "| Branch | Variant | Mean coverage | Macro precision | Precision lift | Tasks >=70% | Tasks lift>0 | Macro AUROC | Macro ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        report.append(
            f"| {row['branch']} | {row['variant']} | {row['mean_coverage']:.3f} | "
            f"{row['macro_precision']:.3f} | {row['macro_precision_lift']:+.3f} | "
            f"{row['tasks_precision_at_least_0_70']}/10 | {row['tasks_precision_above_prevalence']}/10 | "
            f"{row['macro_auroc']:.3f} | {row['macro_ece10']:.3f} |"
        )
    report += [
        "",
        "## Preregistered gate",
        "",
        *[f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()],
        "",
        "The language-augmented model, local-only ablation, and W1-W3 audits are descriptive and cannot trigger GO. No branch, threshold, model family, feature, or coverage range may be changed post hoc to rescue this result.",
        "",
    ]
    (artifact / "report.md").write_text("\n".join(report))
    (artifact / "status/integrity.pass").write_text("all integrity checks passed\n")
    (artifact / "status/analysis.complete").write_text(decision_name + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
