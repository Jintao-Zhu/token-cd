#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from research.coreact_local_ev_confirmation.modeling import (
    LOCAL_FEATURES,
    PRIMARY_BRANCH,
    platt_apply,
)
from research.coreact_selective_cfg.features import assert_deployable_feature_names


BOOTSTRAP_REPLICATES = 10000
RANDOM_SEED = 20260811


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (probability >= lower) & (
            (probability <= upper) if upper == 1 else (probability < upper)
        )
        if selected.any():
            value += selected.mean() * abs(labels[selected].mean() - probability[selected].mean())
    return float(value)


def metrics(frame: pd.DataFrame, threshold: float, frozen_prevalence: float) -> dict:
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
        "frozen_prevalence_brier": float(np.mean((labels - frozen_prevalence) ** 2)),
        "ece10": ece(labels, probability),
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
        precision_values.append(
            float(replica.ev_positive.to_numpy()[selected].mean()) if selected.any() else 0.0
        )
    return {
        "precision_ci_low": float(np.quantile(precision_values, 0.025)),
        "precision_ci_high": float(np.quantile(precision_values, 0.975)),
        "coverage_ci_low": float(np.quantile(coverage_values, 0.025)),
        "coverage_ci_high": float(np.quantile(coverage_values, 0.975)),
    }


def task_bootstrap(task: pd.DataFrame) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    precision = task.precision.to_numpy()
    values = [
        float(rng.choice(precision, size=len(precision), replace=True).mean())
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def make_plot(task: pd.DataFrame, artifact: Path) -> None:
    ordered = task.sort_values("task_id")
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(ordered.task_id - 0.18, ordered.precision, 0.36, label="frozen gate")
    axes[0].bar(
        ordered.task_id + 0.18,
        ordered.ev_positive_prevalence,
        0.36,
        label="ungated prevalence",
    )
    axes[0].axhline(0.70, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("EV-positive fraction")
    axes[0].legend()
    axes[1].bar(ordered.task_id, ordered.coverage, 0.6)
    axes[1].axhspan(0.20, 0.35, color="#59A14F", alpha=0.15)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Coverage")
    axes[1].set_xlabel("LIBERO-Object task")
    axes[1].set_xticks(ordered.task_id)
    figure.tight_layout()
    figure.savefig(artifact / "plots/object_precision_coverage.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    if not (artifact / "status/capture.complete").exists():
        raise RuntimeError("capture must complete before confirmation analysis")
    model_path = artifact / "frozen/spatial_local_ev_gate.joblib"
    if sha256(model_path) != protocol["frozen_gate"]["model_sha256"]:
        raise RuntimeError("frozen model hash mismatch")
    bundle = joblib.load(model_path)
    if (
        tuple(bundle["features"]) != LOCAL_FEATURES
        or bundle["branch"] != PRIMARY_BRANCH
        or float(bundle["threshold"]) != float(protocol["frozen_gate"]["threshold"])
    ):
        raise RuntimeError("frozen gate contract mismatch")
    assert_deployable_feature_names(LOCAL_FEATURES)

    frame = pd.read_parquet(artifact / "object_point_metrics.parquet")
    raw_probability = bundle["model"].predict_proba(frame[list(LOCAL_FEATURES)].to_numpy())[:, 1]
    frame["raw_probability"] = raw_probability
    frame["probability"] = platt_apply(bundle["calibrator"], raw_probability)
    threshold = float(bundle["threshold"])
    frame["selected"] = frame.probability >= threshold
    frame.to_parquet(artifact / "object_predictions.parquet", index=False)

    frozen_meta = json.loads((artifact / "frozen/spatial_freeze.json").read_text())
    frozen_prevalence = float(
        np.mean([row["prevalence"] for row in frozen_meta["development_out_of_task"]["task_rows"]])
    )
    task_rows = []
    for task_id, group in frame.groupby("task_id"):
        row = metrics(group, threshold, frozen_prevalence)
        row.update(state_bootstrap(group, threshold, RANDOM_SEED + int(task_id)))
        task_rows.append({"task_id": int(task_id), **row})
    task = pd.DataFrame(task_rows).sort_values("task_id")
    task.to_csv(artifact / "object_task_summary.csv", index=False)
    ci_low, ci_high = task_bootstrap(task)
    aggregate = {
        "mean_coverage": float(task.coverage.mean()),
        "minimum_task_coverage": float(task.coverage.min()),
        "macro_precision": float(task.precision.mean()),
        "macro_precision_task_bootstrap_ci_low": ci_low,
        "macro_precision_task_bootstrap_ci_high": ci_high,
        "macro_prevalence": float(task.ev_positive_prevalence.mean()),
        "macro_precision_lift": float(task.precision_lift.mean()),
        "tasks_positive_lift": int((task.precision_lift > 0).sum()),
        "tasks_precision_at_least_0_70": int((task.precision >= 0.70).sum()),
        "macro_auroc": float(task.auroc.mean()),
        "macro_average_precision": float(task.average_precision.mean()),
        "macro_ece10": float(task.ece10.mean()),
    }
    gate = protocol["go_gate"]
    checks = {
        "mean_coverage": gate["mean_coverage_range_inclusive"][0]
        <= aggregate["mean_coverage"]
        <= gate["mean_coverage_range_inclusive"][1],
        "minimum_task_coverage": aggregate["minimum_task_coverage"]
        >= gate["minimum_each_task_coverage"],
        "macro_precision": aggregate["macro_precision"] >= gate["macro_precision_min"],
        "macro_precision_lift": aggregate["macro_precision_lift"]
        >= gate["macro_precision_lift_min"],
        "tasks_positive_lift": aggregate["tasks_positive_lift"]
        >= gate["tasks_positive_lift_min"],
        "precision_ci": aggregate["macro_precision_task_bootstrap_ci_low"]
        >= gate["task_bootstrap_precision_ci_lower_min"],
    }
    decision_name = (
        protocol["decision_rules"]["pass"]
        if all(checks.values())
        else protocol["decision_rules"]["fail"]
    )
    integrity_checks = {
        "point_rows_complete": len(frame) == protocol["scope"]["point_rows"],
        "unique_points": frame.point_id.nunique() == protocol["scope"]["point_rows"],
        "ten_tasks": set(frame.task_id.unique()) == set(range(10)),
        "forty_states_each_task": bool((frame.groupby("task_id").state_id.nunique() == 40).all()),
        "all_numeric_finite": bool(
            frame.select_dtypes(include=[np.number]).apply(np.isfinite).all().all()
        ),
        "branch_frozen": frame.branch.nunique() == 1 and frame.branch.iloc[0] == PRIMARY_BRANCH,
        "feature_contract_target_free": True,
        "model_hash_match": True,
        "threshold_exact_match": True,
        "no_fit_on_object": True,
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
    decision = {
        "decision": decision_name,
        "integrity": "PASS",
        "frozen_branch": PRIMARY_BRANCH,
        "frozen_features": list(LOCAL_FEATURES),
        "frozen_threshold": threshold,
        "aggregate": aggregate,
        "gate_checks": checks,
        "closed_loop_authorized": False,
        "next_step": "submit result; a separate locked closed-loop protocol is required even if confirmation passes",
    }
    (artifact / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    (artifact / "confirmation_summary.json").write_text(
        json.dumps(
            {**decision, "tasks": task.to_dict(orient="records")}, indent=2, sort_keys=True
        )
        + "\n"
    )
    make_plot(task, artifact)
    report = [
        "# Local Flow Geometry EV-Validity: Independent LIBERO-Object Confirmation",
        "",
        "## Decision",
        "",
        f"`{decision_name}`",
        "",
        "This offline confirmation never authorizes automatic closed-loop rollout.",
        "",
        "## Frozen method",
        "",
        f"- Branch: `{PRIMARY_BRANCH}` only.",
        f"- Features: {', '.join(f'`{name}`' for name in LOCAL_FEATURES)}.",
        f"- Threshold: `{threshold:.12g}`, frozen from Spatial out-of-task predictions before Object capture.",
        "- No consensus, language embedding, new branch, feature change, classifier change, calibration change, or Object-task tuning.",
        "",
        "## Independent task results",
        "",
        "| Object task | EV prevalence | Coverage | Precision (95% state CI) | Lift | AUROC | ECE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in task.to_dict(orient="records"):
        report.append(
            f"| {row['task_id']} | {row['ev_positive_prevalence']:.3f} | {row['coverage']:.3f} | "
            f"{row['precision']:.3f} [{row['precision_ci_low']:.3f},{row['precision_ci_high']:.3f}] | "
            f"{row['precision_lift']:+.3f} | {row['auroc']:.3f} | {row['ece10']:.3f} |"
        )
    report += [
        "",
        "## Aggregate",
        "",
        f"- Mean coverage: {aggregate['mean_coverage']:.3f}; minimum task coverage: {aggregate['minimum_task_coverage']:.3f}.",
        f"- Macro precision: {aggregate['macro_precision']:.3f}, task-bootstrap 95% CI [{ci_low:.3f},{ci_high:.3f}].",
        f"- Macro prevalence: {aggregate['macro_prevalence']:.3f}; precision lift: {aggregate['macro_precision_lift']:+.3f}.",
        f"- Tasks with positive lift: {aggregate['tasks_positive_lift']}/10; tasks with precision >=0.70: {aggregate['tasks_precision_at_least_0_70']}/10.",
        "",
        "## Preregistered gate",
        "",
        *[f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()],
        "",
        "No threshold, feature, branch, classifier, task rule, or coverage criterion was changed after Object capture.",
        "",
    ]
    (artifact / "report.md").write_text("\n".join(report))
    (artifact / "status/integrity.pass").write_text("all confirmation integrity checks passed\n")
    (artifact / "status/analysis.complete").write_text(decision_name + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
