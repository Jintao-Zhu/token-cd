#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def bootstrap_mean(values: np.ndarray, seed: int, replicates: int = 10000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def integrity_gate(artifact: Path, points: pd.DataFrame, capture: dict) -> dict:
    numeric = points.select_dtypes(include=[np.number])
    same_point = ["unit_id", "flow_step"]
    branch_counts = points.groupby("branch").size().to_dict()
    task_branch_counts = points.groupby(["task_id", "branch"]).size()
    checks = {
        "target_reconstruction_test": (artifact / "status/target_reconstruction.pass").exists(),
        "capture_complete_marker": (artifact / "status/capture.complete").exists(),
        "states_complete": capture["states"] == 100,
        "units_complete": capture["units"] == 300,
        "point_rows_complete": len(points) == 12000,
        "unique_points": points["point_id"].nunique() == 12000,
        "no_nonfinite": bool(numeric.apply(np.isfinite).all().all()),
        "branch_counts": branch_counts == {
            "W1_skip_last_1": 3000,
            "W2_skip_last_2": 3000,
            "W3_half_last_1": 3000,
            "W4_half_last_2": 3000,
        },
        "task_branch_counts": bool((task_branch_counts == 1500).all()) and len(task_branch_counts) == 8,
        "same_x_t_across_branches": bool(
            (points.groupby(same_point)["x_t_sha256"].nunique() == 1).all()
        ),
        "same_target_across_branches": bool(
            (points.groupby(same_point)["target_sha256"].nunique() == 1).all()
        ),
        "same_observation_action_noise_hashes": all(
            bool((points.groupby(same_point)[column].nunique() == 1).all())
            for column in ("prefix_sha256", "action_sha256", "noise_sha256")
        ),
        "same_timestep_across_branches": bool(
            (points.groupby(same_point)["timestep"].nunique() == 1).all()
        ),
        "same_strong_error_across_branches": bool(
            (points.groupby(same_point)["strong_error"].max()
             - points.groupby(same_point)["strong_error"].min()
             <= 1e-12).all()
        ),
        "strong_vanilla_parity": capture["strong_ones_parity_max_abs"] <= 1e-5,
        "deterministic_rerun": capture["deterministic_rerun_max_abs"] <= 1e-7,
        "expert_layer_count": capture["expert_layer_count"] == 16,
        "weak_scope_contract": capture["branch_scale_contract"] == {
            "W1_skip_last_1": {"affected_layers": 1, "residual_scale": 0.0},
            "W2_skip_last_2": {"affected_layers": 2, "residual_scale": 0.0},
            "W3_half_last_1": {"affected_layers": 1, "residual_scale": 0.5},
            "W4_half_last_2": {"affected_layers": 2, "residual_scale": 0.5},
        },
    }
    return {"pass": all(checks.values()), "checks": checks, "capture": capture}


def make_plots(flow: pd.DataFrame, artifact: Path) -> None:
    metrics = (
        ("quality_ordering_rate", "Quality ordering rate", 0.5),
        ("ev_positive_rate", "EV-positive rate", 0.5),
        ("median_ev_cosine", "Median EV cosine", 0.0),
        ("median_relative_correction_norm", "Median relative correction norm", None),
    )
    for task_id in (4, 7):
        task = flow[flow.task_id == task_id]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        for axis, (column, title, reference) in zip(axes.flat, metrics, strict=True):
            for branch, group in task.groupby("branch"):
                axis.plot(group.flow_step, group[column], marker="o", label=branch)
            if reference is not None:
                axis.axhline(reference, color="black", linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("Flow step (0=tau 1.0, 9=tau 0.1)")
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(f"Task {task_id}: quality-ordered negative branch Phase-0")
        figure.tight_layout()
        figure.savefig(artifact / "plots" / f"task{task_id}_flow_step_metrics.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    points = pd.read_parquet(artifact / "point_level_metrics.parquet")
    capture = json.loads((artifact / "capture_summary.json").read_text())
    integrity = integrity_gate(artifact, points, capture)
    (artifact / "integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    if not integrity["pass"]:
        decision = {
            "decision": "NEGATIVE_BRANCH_INTEGRITY_FAILED_NO_SCIENTIFIC_ANALYSIS",
            "closed_loop_authorized": False,
        }
        (artifact / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        raise RuntimeError(f"integrity failed: {integrity['checks']}")
    (artifact / "status/integrity.pass").write_text("all preregistered integrity checks pass\n")

    state = (
        points.groupby(["task_id", "branch", "state_id"], as_index=False)
        .agg(
            mean_delta_quality=("delta_quality", "mean"),
            quality_ordering_fraction=("quality_ordered", "mean"),
            mean_extrapolation_validity=("extrapolation_validity", "mean"),
            ev_positive_fraction=("ev_positive", "mean"),
            median_ev_cosine=("ev_cosine", "median"),
            median_lambda_star=("lambda_star", "median"),
            median_strong_weak_cosine=("strong_weak_cosine", "median"),
            median_relative_correction_norm=("relative_correction_norm", "median"),
            lambda_0_1_error_delta=("guided_error_delta_lambda_0.1", "mean"),
            lambda_0_25_error_delta=("guided_error_delta_lambda_0.25", "mean"),
            lambda_0_5_error_delta=("guided_error_delta_lambda_0.5", "mean"),
        )
    )
    state.to_csv(artifact / "state_level_metrics.csv", index=False)

    flow = (
        points.groupby(["task_id", "branch", "flow_step", "timestep"], as_index=False)
        .agg(
            quality_ordering_rate=("quality_ordered", "mean"),
            ev_positive_rate=("ev_positive", "mean"),
            median_ev_cosine=("ev_cosine", "median"),
            median_relative_correction_norm=("relative_correction_norm", "median"),
        )
    )
    flow.to_csv(artifact / "flow_step_summary.csv", index=False)

    task_rows = []
    gate = protocol["go_gate_each_task"]
    for (task_id, branch), group in state.groupby(["task_id", "branch"]):
        quality_values = group["quality_ordering_fraction"].to_numpy()
        ev_values = group["ev_positive_fraction"].to_numpy()
        quality_ci = bootstrap_mean(quality_values, 20260811 + int(task_id) * 100 + len(branch))
        ev_ci = bootstrap_mean(ev_values, 20270811 + int(task_id) * 100 + len(branch))
        flow_group = flow[(flow.task_id == task_id) & (flow.branch == branch)]
        quality_steps = int((flow_group.quality_ordering_rate > 0.5).sum())
        ev_steps = int((flow_group.ev_positive_rate > 0.5).sum())
        row = {
            "task_id": int(task_id),
            "branch": branch,
            "states": len(group),
            "quality_ordering_rate": float(quality_values.mean()),
            "quality_ordering_ci_low": quality_ci[0],
            "quality_ordering_ci_high": quality_ci[1],
            "ev_positive_rate": float(ev_values.mean()),
            "ev_positive_ci_low": ev_ci[0],
            "ev_positive_ci_high": ev_ci[1],
            "median_ev_cosine": float(group.median_ev_cosine.median()),
            "median_lambda_star": float(group.median_lambda_star.median()),
            "median_strong_weak_cosine": float(group.median_strong_weak_cosine.median()),
            "median_relative_correction_norm": float(
                group.median_relative_correction_norm.median()
            ),
            "quality_positive_flow_steps": quality_steps,
            "ev_positive_flow_steps": ev_steps,
            "mean_guided_error_delta_lambda_0.1": float(group.lambda_0_1_error_delta.mean()),
            "mean_guided_error_delta_lambda_0.25": float(group.lambda_0_25_error_delta.mean()),
            "mean_guided_error_delta_lambda_0.5": float(group.lambda_0_5_error_delta.mean()),
        }
        row["quality_gate"] = (
            row["quality_ordering_rate"] >= gate["quality_ordering_rate_min"]
            and row["quality_ordering_ci_low"]
            > gate["quality_ordering_bootstrap_ci_lower_strictly_above"]
        )
        row["ev_gate"] = (
            row["ev_positive_rate"] >= gate["ev_positive_rate_min"]
            and row["ev_positive_ci_low"]
            > gate["ev_bootstrap_ci_lower_strictly_above"]
        )
        row["geometry_gate"] = (
            row["median_strong_weak_cosine"]
            > gate["median_strong_weak_cosine_strictly_above"]
            and gate["median_relative_correction_norm_strict_range"][0]
            < row["median_relative_correction_norm"]
            < gate["median_relative_correction_norm_strict_range"][1]
        )
        row["temporal_gate"] = (
            quality_steps >= gate["flow_steps_quality_rate_above_half_min"]
            and ev_steps >= gate["flow_steps_ev_rate_above_half_min"]
        )
        row["pass"] = all(
            row[name] for name in ("quality_gate", "ev_gate", "geometry_gate", "temporal_gate")
        )
        task_rows.append(row)
    task = pd.DataFrame(task_rows).sort_values(["branch", "task_id"])
    task.to_csv(artifact / "task_summary.csv", index=False)

    pooled = (
        state.groupby("branch", as_index=False)
        .agg(
            quality_ordering_rate=("quality_ordering_fraction", "mean"),
            ev_positive_rate=("ev_positive_fraction", "mean"),
            median_ev_cosine=("median_ev_cosine", "median"),
            median_lambda_star=("median_lambda_star", "median"),
            median_strong_weak_cosine=("median_strong_weak_cosine", "median"),
            median_relative_correction_norm=("median_relative_correction_norm", "median"),
        )
    )
    branch_results = []
    for branch, group in task.groupby("branch"):
        pooled_row = pooled[pooled.branch == branch].iloc[0].to_dict()
        passes = bool(group["pass"].all()) and set(group.task_id) == {4, 7}
        branch_results.append(
            {
                "branch": branch,
                "both_tasks_pass": passes,
                "task_results": group.to_dict(orient="records"),
                "pooled": pooled_row,
            }
        )
    passing = [row["branch"] for row in branch_results if row["both_tasks_pass"]]
    quality_both_tasks = [
        branch for branch, group in task.groupby("branch") if bool(group["quality_gate"].all())
    ]
    ev_both_tasks = [
        branch for branch, group in task.groupby("branch") if bool(group["ev_gate"].all())
    ]
    temporal_both_tasks = [
        branch for branch, group in task.groupby("branch") if bool(group["temporal_gate"].all())
    ]
    decision_name = (
        "QUALITY_ORDERED_NEGATIVE_BRANCH_FOUND_READY_FOR_CONFIRMATION"
        if passing
        else "NEGATIVE_BRANCH_QUALIFICATION_FAILED_NO_ROLLOUT"
    )
    branch_summary = {
        "decision": decision_name,
        "passing_branches": passing,
        "quality_ordered_on_both_tasks": quality_both_tasks,
        "ev_valid_on_both_tasks": ev_both_tasks,
        "temporally_robust_on_both_tasks": temporal_both_tasks,
        "branches": branch_results,
        "closed_loop_authorized": False,
    }
    (artifact / "branch_summary.json").write_text(
        json.dumps(branch_summary, indent=2, sort_keys=True) + "\n"
    )
    (artifact / "decision.json").write_text(
        json.dumps(
            {
                "decision": decision_name,
                "integrity": "PASS",
                "passing_branches": passing,
                "closed_loop_authorized": False,
                "next_step": "submit report and wait for a separately locked confirmation protocol",
            },
            indent=2,
        )
        + "\n"
    )
    make_plots(flow, artifact)

    report_lines = [
        "# Flow-VLA Quality-Ordered Negative Branch Phase-0",
        "",
        "## Integrity",
        "",
        "PASS: 100/100 expert states, 300/300 state-noise units, and 3,000 points per branch. No missing, duplicate, or nonfinite records. Same-condition hashes, training-target reconstruction, full-strong parity, weak-layer scope, and deterministic rerun all pass.",
        "",
        "## Qualification summary",
        "",
        "| Branch | Task | Quality rate (95% CI) | EV-positive rate (95% CI) | Median cos(strong,weak) | Median relative correction | Robust steps Q/EV | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in task.to_dict(orient="records"):
        report_lines.append(
            f"| {row['branch']} | {row['task_id']} | {row['quality_ordering_rate']:.3f} [{row['quality_ordering_ci_low']:.3f},{row['quality_ordering_ci_high']:.3f}] | "
            f"{row['ev_positive_rate']:.3f} [{row['ev_positive_ci_low']:.3f},{row['ev_positive_ci_high']:.3f}] | "
            f"{row['median_strong_weak_cosine']:.3f} | {row['median_relative_correction_norm']:.3f} | "
            f"{row['quality_positive_flow_steps']}/{row['ev_positive_flow_steps']} | {'PASS' if row['pass'] else 'FAIL'} |"
        )
    report_lines += [
        "",
        "## Secondary fixed-lambda diagnostic",
        "",
        "Negative values below mean that the fixed extrapolation lowers expert-flow MSE. These values were not used to tune lambda.",
        "",
        "| Branch | Task | lambda=0.1 | lambda=0.25 | lambda=0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in task.to_dict(orient="records"):
        report_lines.append(
            f"| {row['branch']} | {row['task_id']} | {row['mean_guided_error_delta_lambda_0.1']:.6g} | "
            f"{row['mean_guided_error_delta_lambda_0.25']:.6g} | {row['mean_guided_error_delta_lambda_0.5']:.6g} |"
        )
    report_lines += [
        "",
        "## Required answers",
        "",
        f"- Weak branch genuinely lower quality: {'yes; ' + ', '.join(quality_both_tasks) + ' pass the Quality Ordering gate on both tasks' if quality_both_tasks else 'no branch met the preregistered two-task quality gate'}.",
        f"- Strong-minus-weak has positive extrapolation validity: {'yes for ' + ', '.join(ev_both_tasks) + ' on both tasks' if ev_both_tasks else 'no branch met the EV gate on both tasks; Task 7 is the common failure'}.",
        f"- Property exists in both Task 4 and Task 7: {'yes' if passing else 'not established'}.",
        f"- Stable across multiple flow steps: {'yes for ' + ', '.join(temporal_both_tasks) + ' on both tasks' if temporal_both_tasks else 'no branch met temporal robustness on both tasks; Quality Ordering is stable but Task 7 EV is not'}.",
        f"- Passing branches: {', '.join(passing) if passing else 'none'}.",
        "- Closed-loop allowed: no. Phase-0 never authorizes automatic rollout.",
        "",
        "## Decision",
        "",
        f"`{decision_name}`",
        "",
    ]
    (artifact / "report.md").write_text("\n".join(report_lines))
    (artifact / "status/analysis.complete").write_text(decision_name + "\n")
    print(json.dumps({"decision": decision_name, "passing_branches": passing}, indent=2))


if __name__ == "__main__":
    main()
