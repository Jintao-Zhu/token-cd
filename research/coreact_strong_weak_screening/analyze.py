#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

from research.coreact_strong_weak_screening.sampler import ARMS


BOOTSTRAP_REPLICATES = 10000
SEED = 20260811


def paired_comparison(frame: pd.DataFrame, arm: str, seed: int) -> dict:
    pivot = frame.pivot(index="pair_id", columns="arm", values="success").astype(int)
    differences = pivot[arm].to_numpy() - pivot["Vanilla"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(BOOTSTRAP_REPLICATES, len(differences)), replace=True)
    arm_only = int(((pivot[arm] == 1) & (pivot["Vanilla"] == 0)).sum())
    vanilla_only = int(((pivot[arm] == 0) & (pivot["Vanilla"] == 1)).sum())
    discordant = arm_only + vanilla_only
    return {
        "comparison": f"{arm}_minus_Vanilla",
        "point_estimate": float(differences.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(samples.mean(axis=1), 0.025)),
            float(np.quantile(samples.mean(axis=1), 0.975)),
        ],
        "arm_only_success": arm_only,
        "vanilla_only_success": vanilla_only,
        "discordant": discordant,
        "mcnemar_exact_p": float(binomtest(arm_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    manifest = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines()]
    records, missing = [], []
    for spec in manifest:
        path = artifact / "episodes" / f"{spec['episode_id']}.json"
        if not path.exists():
            missing.append(spec["episode_id"])
        else:
            records.append(json.loads(path.read_text()))
    if missing:
        raise RuntimeError(f"missing {len(missing)} episodes; first={missing[0]}")
    frame = pd.DataFrame(records)
    failures = []
    if len(frame) != 400 or frame.episode_id.nunique() != 400:
        failures.append("episode_count")
    if set(frame.arm) != set(ARMS):
        failures.append("arms")
    if not frame.all_actions_finite.all():
        failures.append("nonfinite")
    for pair_id, group in frame.groupby("pair_id"):
        if set(group.arm) != set(ARMS) or group.initial_sim_state_sha256.nunique() != 1:
            failures.append(f"pair_contract:{pair_id}")
        if group.initial_prepared_input_sha256.nunique() != 1:
            failures.append(f"prepared_identity:{pair_id}")
        if len({row[0] for row in group.noise_sha256_by_replan}) != 1:
            failures.append(f"first_noise:{pair_id}")
    if failures:
        raise RuntimeError(f"integrity failures: {failures[:10]}")

    task_rows = []
    for task_id, task_group in frame.groupby("task_id"):
        rates = task_group.groupby("arm").success.mean().to_dict()
        row = {"task_id": int(task_id), **{arm: float(rates[arm]) for arm in ARMS}}
        for arm in ARMS[1:]:
            row[f"{arm}_delta"] = row[arm] - row["Vanilla"]
        task_rows.append(row)
    task = pd.DataFrame(task_rows).sort_values("task_id")
    task.to_csv(artifact / "task_success_summary.csv", index=False)
    rates = {
        arm: {
            "successes": int(frame[frame.arm == arm].success.sum()),
            "episodes": 100,
            "success_rate": float(frame[frame.arm == arm].success.mean()),
        }
        for arm in ARMS
    }
    comparisons = {
        arm: paired_comparison(frame, arm, SEED + index)
        for index, arm in enumerate(ARMS[1:], 1)
    }
    task_directions = {}
    for arm in ARMS[1:]:
        delta = task[f"{arm}_delta"]
        task_directions[arm] = {
            "positive": int((delta > 0).sum()),
            "tied": int((delta == 0).sum()),
            "negative": int((delta < 0).sum()),
        }
    secondary = {
        arm: {
            "mean_control_steps": float(frame[frame.arm == arm].control_steps.mean()),
            "mean_action_tv": float(frame[frame.arm == arm].action_total_variation.mean()),
            "mean_chunk_discontinuity": float(frame[frame.arm == arm].chunk_discontinuity.mean()),
            "median_replan_latency_seconds": float(
                frame[frame.arm == arm].median_replan_latency_seconds.median()
            ),
        }
        for arm in ARMS
    }
    best_cfg = max(("W3_CFG", "W4_CFG"), key=lambda arm: rates[arm]["success_rate"])
    rule = protocol["analysis"]["promising_rule"]
    cfg_gain = comparisons[best_cfg]["point_estimate"]
    direction = task_directions[best_cfg]
    weak_lower = rates["W4_only"]["success_rate"] < rates["Vanilla"]["success_rate"]
    gain_pattern = (
        cfg_gain >= rule["best_cfg_overall_gain_min"]
        and direction["positive"] >= rule["best_cfg_positive_tasks_min"]
        and direction["negative"] <= rule["best_cfg_negative_tasks_max"]
    )
    if gain_pattern and weak_lower:
        decision = "PROMISING_AUTOGUIDANCE_PATTERN_READY_FOR_SEPARATE_CONFIRMATION_PROTOCOL"
    elif gain_pattern:
        decision = "CFG_PROMISING_BUT_WEAK_ONLY_ORDER_NOT_ESTABLISHED"
    elif all(comparisons[arm]["point_estimate"] <= 0 for arm in ("W3_CFG", "W4_CFG")):
        decision = "NO_CROSS_TASK_STRUCTURAL_CFG_BENEFIT_STOP"
    else:
        decision = "TASK_HETEROGENEOUS_STRUCTURAL_CFG_SCREENING_NO_AUTOMATIC_FOLLOWUP"
    result = {
        "decision": decision,
        "scope": "400-episode screening; not confirmation",
        "integrity": "PASS",
        "rates": rates,
        "comparisons": comparisons,
        "task_directions": task_directions,
        "best_cfg_arm": best_cfg,
        "best_cfg_gain": cfg_gain,
        "w4_only_below_vanilla": weak_lower,
        "promising_rule_pass": gain_pattern and weak_lower,
        "secondary": secondary,
        "automatic_confirmation_authorized": False,
    }
    (artifact / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "integrity": "PASS",
                "automatic_confirmation_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    figure, axis = plt.subplots(figsize=(10, 4.8))
    x = np.arange(10)
    width = 0.2
    for index, arm in enumerate(ARMS):
        axis.bar(x + (index - 1.5) * width, task[arm], width, label=arm)
    axis.set_xticks(x, [str(value) for value in task.task_id])
    axis.set_ylim(0, 1)
    axis.set_xlabel("LIBERO-Spatial task")
    axis.set_ylabel("Success rate")
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(artifact / "plots/task_success_rates.png", dpi=180)
    plt.close(figure)

    report = [
        "# Structural Strong-Weak CFG Closed-Loop Screening",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "This is a 400-episode screening result, not paper-level confirmation.",
        "",
        "## Overall success",
        "",
        "| Arm | Success | Rate | Delta vs Vanilla (paired 95% CI) | McNemar p |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        comparison = comparisons.get(arm)
        delta = "-" if comparison is None else (
            f"{comparison['point_estimate']:+.3f} "
            f"[{comparison['paired_bootstrap_95_ci'][0]:+.3f},{comparison['paired_bootstrap_95_ci'][1]:+.3f}]"
        )
        p = "-" if comparison is None else f"{comparison['mcnemar_exact_p']:.4g}"
        report.append(
            f"| {arm} | {rates[arm]['successes']}/100 | {rates[arm]['success_rate']:.3f} | {delta} | {p} |"
        )
    report += [
        "",
        "## Per-task success rates",
        "",
        "| Task | Vanilla | W4-only | W3-CFG | W4-CFG |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in task.to_dict(orient="records"):
        report.append(
            f"| {int(row['task_id'])} | {row['Vanilla']:.2f} | {row['W4_only']:.2f} | "
            f"{row['W3_CFG']:.2f} | {row['W4_CFG']:.2f} |"
        )
    report += ["", "## Cross-task direction", ""]
    for arm in ARMS[1:]:
        value = task_directions[arm]
        report.append(
            f"- {arm} vs Vanilla: {value['positive']} positive / {value['tied']} tied / {value['negative']} negative tasks."
        )
    report += [
        "",
        "No EV gate, clipping, lambda tuning, task-specific rule, or outcome-based task/state filtering was used.",
        "",
    ]
    (artifact / "report.md").write_text("\n".join(report))
    (artifact / "status/analysis.complete").write_text(decision + "\n")
    (artifact / "status/integrity.pass").write_text("all 400 episode contracts passed\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
