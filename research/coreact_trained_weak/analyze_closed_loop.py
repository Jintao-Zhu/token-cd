from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ARMS = ("V", "W", "M", "G")


def percentile_interval(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    paths = sorted((artifact / "trained_weak_closed_loop_episodes").glob("*.json"))
    if len(paths) != 400:
        raise RuntimeError(f"expected 400 episodes, got {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    if any(row["status"] != "complete" or not row["all_actions_finite"] for row in rows):
        raise RuntimeError("incomplete or nonfinite episode")
    identities = {}
    for row in rows:
        unit = identities.setdefault(row["unit_id"], {"sim": set(), "prepared": set(), "noise0": set(), "arms": set()})
        unit["sim"].add(row["initial_sim_state_sha256"])
        unit["prepared"].add(row["initial_prepared_input_sha256"])
        unit["noise0"].add(row["noise_sha256_by_replan"][0])
        unit["arms"].add(row["arm"])
    if len(identities) != 100 or any(
        len(value["sim"]) != 1 or len(value["prepared"]) != 1 or len(value["noise0"]) != 1 or value["arms"] != set(ARMS)
        for value in identities.values()
    ):
        raise RuntimeError("paired identity audit failed")

    frame = pd.DataFrame(rows)
    wide = frame.pivot(index=["unit_id", "task_id", "init_state_id"], columns="arm", values="success").reset_index()
    wide.to_csv(artifact / "trained_weak_paired_outcomes.csv", index=False)
    arm_rates = {arm: float(wide[arm].mean()) for arm in ARMS}
    task_rows = []
    for task in range(10):
        subset = wide[wide.task_id == task]
        task_rows.append({"task_id": task, **{arm: int(subset[arm].sum()) for arm in ARMS},
                          "G_minus_V_pp": float((subset.G.mean() - subset.V.mean()) * 100),
                          "G_minus_M_pp": float((subset.G.mean() - subset.M.mean()) * 100)})
    pd.DataFrame(task_rows).to_csv(artifact / "trained_weak_task_success.csv", index=False)

    differences = wide.G.astype(int).to_numpy() - wide.V.astype(int).to_numpy()
    rescued, harmed = int((differences == 1).sum()), int((differences == -1).sum())
    rng = np.random.default_rng(1729)
    paired_boot = np.empty(100000)
    for start in range(0, 100000, 5000):
        indices = rng.integers(0, len(differences), (5000, len(differences)))
        paired_boot[start : start + 5000] = differences[indices].mean(axis=1) * 100
    task_groups = [wide[wide.task_id == task] for task in range(10)]
    task_differences = np.stack([
        group.G.astype(int).to_numpy() - group.V.astype(int).to_numpy()
        for group in task_groups
    ])
    stratified_boot = np.empty(100000)
    for start in range(0, 100000, 5000):
        indices = rng.integers(0, 10, (5000, 10, 10))
        sampled = np.take_along_axis(task_differences[None, :, :], indices, axis=2)
        stratified_boot[start : start + 5000] = sampled.mean(axis=(1, 2)) * 100
    discordant = rescued + harmed
    mcnemar_p = float(binomtest(min(rescued, harmed), discordant, 0.5).pvalue) if discordant else 1.0
    nonnegative_tasks = sum(row["G_minus_V_pp"] >= 0 for row in task_rows)
    positive_tasks = sum(row["G_minus_V_pp"] > 0 for row in task_rows)
    negative_tasks = sum(row["G_minus_V_pp"] < 0 for row in task_rows)
    catastrophic = any(row["G_minus_V_pp"] < -10 for row in task_rows)
    effect = (arm_rates["G"] - arm_rates["V"]) * 100
    paired_ci = percentile_interval(paired_boot)
    stratified_ci = percentile_interval(np.asarray(stratified_boot))
    strong_go = effect >= 5 and paired_ci[0] > 0 and nonnegative_tasks >= 7 and arm_rates["G"] > arm_rates["M"] and not catastrophic
    promising = effect >= 5 and nonnegative_tasks >= 7 and arm_rates["W"] < arm_rates["V"] and arm_rates["G"] > arm_rates["M"]
    if strong_go:
        decision = "SAME_TRAJECTORY_TRAINED_WEAK_CFG_STRONG_GO"
    elif promising:
        decision = "SAME_TRAJECTORY_TRAINED_WEAK_CFG_PROMISING"
    else:
        decision = "SAME_TRAJECTORY_TRAINED_WEAK_CFG_NO_GENERAL_BENEFIT_STOP"
    analysis = {
        "decision": decision,
        "episodes": 400,
        "matched_units": 100,
        "success_rates": arm_rates,
        "G_minus_V_pp": effect,
        "G_minus_M_pp": (arm_rates["G"] - arm_rates["M"]) * 100,
        "W_minus_V_pp": (arm_rates["W"] - arm_rates["V"]) * 100,
        "M_minus_V_pp": (arm_rates["M"] - arm_rates["V"]) * 100,
        "paired_bootstrap_95_ci_pp": paired_ci,
        "task_stratified_bootstrap_95_ci_pp": stratified_ci,
        "mcnemar_exact_p": mcnemar_p,
        "rescued_units": rescued,
        "harmed_units": harmed,
        "tasks_G_positive_tie_negative_vs_V": [positive_tasks, nonnegative_tasks - positive_tasks, negative_tasks],
        "tasks_G_nonnegative_vs_V": nonnegative_tasks,
        "catastrophic_task_harm_gt_10pp": catastrophic,
        "paired_hash_audit": "PASS",
        "all_actions_finite": True,
        "lambda": 0.5,
        "posthoc_tuning": False,
    }
    (artifact / "trained_weak_closed_loop_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (artifact / "trained_weak_decision.json").write_text(json.dumps({"decision": decision, "analysis": analysis}, indent=2) + "\n")
    task_table = "\n".join(
        f"| {row['task_id']} | {row['V']} | {row['W']} | {row['M']} | {row['G']} | {row['G_minus_V_pp']:+.0f}pp |"
        for row in task_rows
    )
    report = f"""# Trained-Weak AutoGuidance Closed-Loop Report

## Decision

`{decision}`

## Primary result

| Arm | Success |
|---|---:|
| Strong Vanilla (V) | {arm_rates['V']:.0%} |
| Trained Weak (W) | {arm_rates['W']:.0%} |
| Midpoint (M) | {arm_rates['M']:.0%} |
| AutoGuidance (G) | {arm_rates['G']:.0%} |

- G - V: {effect:+.1f}pp
- Paired bootstrap 95% CI: [{paired_ci[0]:+.1f}, {paired_ci[1]:+.1f}]pp
- Task-stratified bootstrap 95% CI: [{stratified_ci[0]:+.1f}, {stratified_ci[1]:+.1f}]pp
- McNemar exact p: {mcnemar_p:.6g}
- Rescued / harmed: {rescued} / {harmed}
- G positive / tie / negative tasks: {positive_tasks} / {nonnegative_tasks-positive_tasks} / {negative_tasks}
- G - M: {(arm_rates['G']-arm_rates['M'])*100:+.1f}pp

## Task results (successes /10)

| Task | V | W | M | G | G-V |
|---:|---:|---:|---:|---:|---:|
{task_table}

## Integrity

400/400 episodes complete; 100/100 four-arm paired identities passed exact initial simulator, prepared-input, and first-noise hash checks. All actions were finite. Strong=15k and Weak=10k remained locked; lambda=0.5 was selected from 6000 offline points without closed-loop outcomes. No outcome-driven rerun, pair change, or lambda tuning was performed.
"""
    (artifact / "trained_weak_final_report.md").write_text(report)
    print(json.dumps(analysis))


if __name__ == "__main__":
    main()
