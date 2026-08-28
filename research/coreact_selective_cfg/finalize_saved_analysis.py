#!/usr/bin/env python3
"""Finalize an already-computed Phase-1 analysis after a JSON scalar serialization failure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.coreact_selective_cfg.analyze import make_plot


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    task = pd.read_csv(artifact / "loto_task_summary.csv")
    aggregate = pd.read_csv(artifact / "aggregate_summary.csv")
    integrity = json.loads((artifact / "integrity.json").read_text())
    if not integrity["pass"]:
        raise RuntimeError("cannot finalize failed integrity")
    if len(task) != 60 or len(aggregate) != 6:
        raise RuntimeError("saved analysis tables are incomplete")

    gate = protocol["go_gate"]
    primary_task = task[
        (task.branch == gate["decision_branch"])
        & (task.variant == gate["decision_variant"])
    ].copy()
    primary = aggregate[
        (aggregate.branch == gate["decision_branch"])
        & (aggregate.variant == gate["decision_variant"])
    ].iloc[0]
    if len(primary_task) != 10:
        raise RuntimeError("primary outer-task rows are incomplete")
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
    result = {
        "decision": decision_name,
        "primary_branch": gate["decision_branch"],
        "primary_variant": gate["decision_variant"],
        "gate_checks": checks,
        "primary_aggregate": primary.to_dict(),
        "closed_loop_authorized": False,
        "secondary_variants_can_trigger_go": False,
        "finalization_amendment": "protocol.amendment.serialization.yaml",
    }
    (artifact / "decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    (artifact / "phase1_summary.json").write_text(
        json.dumps(
            {
                **result,
                "aggregate": aggregate.to_dict(orient="records"),
                "primary_tasks": primary_task.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
            default=json_default,
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
        "PASS: 500/500 states, 1,500/1,500 state-noise units, 15,000 matched points, 60,000 branch rows, and 90,000 nested outer predictions. Same-condition hashes, finite values, feature-contract leakage audit, task isolation, and completeness all pass.",
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
        "| Branch | Variant | Mean coverage | Min task coverage | Macro precision | Precision lift | Tasks >=70% | Tasks lift>0 | Macro AUROC | Macro ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        report.append(
            f"| {row['branch']} | {row['variant']} | {row['mean_coverage']:.3f} | "
            f"{row['minimum_task_coverage']:.3f} | {row['macro_precision']:.3f} | "
            f"{row['macro_precision_lift']:+.3f} | {int(row['tasks_precision_at_least_0_70'])}/10 | "
            f"{int(row['tasks_precision_above_prevalence'])}/10 | {row['macro_auroc']:.3f} | "
            f"{row['macro_ece10']:.3f} |"
        )
    report += [
        "",
        "## Preregistered gate",
        "",
        *[f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()],
        "",
        "## Interpretation",
        "",
        "The primary gate enriches EV-positive points on all 10 held-out tasks and reaches 75.6% macro precision, but misses the preregistered mean-coverage gate (29.42% vs 30%) and the task-count precision gate (6/10 vs 7/10). Therefore it is a near miss, not a GO.",
        "",
        "Local-only geometry is descriptively stronger than multi-branch consensus (77.3% vs 75.6% macro precision), so this experiment does not support a claim that consensus adds predictive value. The language variant is also descriptive only and cannot rescue the primary decision.",
        "",
        "No branch, threshold, model family, feature, coverage range, or gate may be changed post hoc to rescue this result. A new protocol would be required for any confirmation, and no rollout is authorized here.",
        "",
        "## Amendment",
        "",
        "All scientific tables and integrity results were saved before final JSON serialization failed on a `numpy.bool_`. `protocol.amendment.serialization.yaml` records the implementation-only finalization; no fit, prediction, calibration, threshold, metric, or decision rule was recomputed or changed.",
        "",
    ]
    (artifact / "report.md").write_text("\n".join(report))
    (artifact / "status/integrity.pass").write_text("all integrity checks passed\n")
    (artifact / "status/analysis.complete").write_text(decision_name + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
