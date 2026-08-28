"""Audit and analyze the locked AR token counterfactual qualification."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def cluster_bootstrap(values_by_task: dict[int, list[float]], statistic, seed: int = 20260808, replicates: int = 2000):
    rng = np.random.default_rng(seed)
    tasks = sorted(values_by_task)
    estimates = []
    for _ in range(replicates):
        sample = []
        for task in rng.choice(tasks, size=len(tasks), replace=True):
            values = np.asarray(values_by_task[int(task)], dtype=float)
            sample.extend(rng.choice(values, size=len(values), replace=True).tolist())
        estimates.append(float(statistic(sample)))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    rows = [json.loads(line) for line in (artifact / "token_effects.jsonl").read_text().splitlines() if line]
    expected_rows = 150 * 16 * 7
    key_counts = Counter((row["snapshot_id"], row["visual_token_idx"], row["action_position"]) for row in rows)
    duplicates = sum(count - 1 for count in key_counts.values() if count > 1)
    nonfinite = sum(not row["all_finite"] for row in rows)
    if len(rows) != expected_rows or len(key_counts) != expected_rows or duplicates or nonfinite:
        raise RuntimeError(f"Raw effects integrity failure: rows={len(rows)}, keys={len(key_counts)}, duplicates={duplicates}, nonfinite={nonfinite}")

    csv_path = artifact / "token_effects.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    token_rows = defaultdict(list)
    for row in rows:
        token_rows[(row["snapshot_id"], row["visual_token_idx"])].append(row)
    token_effects = []
    for (_, _), group in token_rows.items():
        first = group[0]
        token_effects.append({
            "snapshot_id": first["snapshot_id"], "task_id": first["task_id"], "phase": first["phase"],
            "token": first["visual_token_idx"], "category": first["selection_category"],
            "attention": first["attention_score"], "mean_js": float(np.mean([row["js_div"] for row in group])),
            "any_flip": any(row["argmax_flip"] for row in group),
            "free_hamming": first["free_running_sequence_hamming"], "free_l2": first["free_running_action_l2"],
        })

    by_state = defaultdict(list)
    for row in token_effects:
        by_state[row["snapshot_id"]].append(row)
    state_metrics = []
    for snapshot_id, group in by_state.items():
        rho = float(spearmanr([row["attention"] for row in group], [row["mean_js"] for row in group]).statistic)
        category_means = {category: float(np.mean([row["mean_js"] for row in group if row["category"] == category])) for category in (
            "attention_top", "attention_middle", "attention_bottom", "deterministic_random"
        )}
        state_metrics.append({
            "snapshot_id": snapshot_id, "task_id": group[0]["task_id"], "phase": group[0]["phase"], "spearman": rho,
            **category_means,
            "top_minus_random": category_means["attention_top"] - category_means["deterministic_random"],
            "top_minus_bottom": category_means["attention_top"] - category_means["attention_bottom"],
        })
    with (artifact / "state_level_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_metrics[0]))
        writer.writeheader(); writer.writerows(state_metrics)

    rho_by_task = defaultdict(list)
    top_random_by_task = defaultdict(list)
    top_bottom_by_task = defaultdict(list)
    for row in state_metrics:
        rho_by_task[row["task_id"]].append(row["spearman"])
        top_random_by_task[row["task_id"]].append(row["top_minus_random"])
        top_bottom_by_task[row["task_id"]].append(row["top_minus_bottom"])
    rho_values = [row["spearman"] for row in state_metrics]
    rho_ci = cluster_bootstrap(rho_by_task, np.median)
    top_random_ci = cluster_bootstrap(top_random_by_task, np.mean, seed=20260809)
    top_bottom_ci = cluster_bootstrap(top_bottom_by_task, np.mean, seed=20260810)
    flip_fraction = float(np.mean([row["argmax_flip"] for row in rows]))

    task_direction = {str(task): float(np.median(values)) for task, values in sorted(rho_by_task.items())}
    action_directions = {}
    for position in range(7):
        top = [row["js_div"] for row in rows if row["action_position"] == position and row["selection_category"] == "attention_top"]
        random = [row["js_div"] for row in rows if row["action_position"] == position and row["selection_category"] == "deterministic_random"]
        action_directions[str(position)] = float(np.mean(top) - np.mean(random))
    criteria = {
        "spearman_predictive": float(np.median(rho_values)) >= 0.30 and rho_ci[0] > 0,
        "top_exceeds_controls": float(np.mean([row["top_minus_random"] for row in state_metrics])) > 0 and top_random_ci[0] > 0 and top_bottom_ci[0] > 0,
        "argmax_flip_signal": flip_fraction >= 0.05,
        "cross_task_action_repeatability": all(value > 0 for value in task_direction.values()) and sum(value > 0 for value in action_directions.values()) >= 4,
    }
    passed = sum(criteria.values())
    decision = "AR_TOKEN_GO" if passed >= 3 else "AR_TOKEN_NO_GO"
    summary = {
        "raw_rows": len(rows), "states": len(state_metrics), "tokens": len(token_effects), "missing_rows": 0, "duplicates": 0,
        "state_spearman_median": float(np.median(rho_values)),
        "state_spearman_iqr": [float(np.quantile(rho_values, 0.25)), float(np.quantile(rho_values, 0.75))],
        "state_spearman_cluster_bootstrap_95ci": rho_ci,
        "top_minus_random_mean": float(np.mean([row["top_minus_random"] for row in state_metrics])),
        "top_minus_random_cluster_bootstrap_95ci": top_random_ci,
        "top_minus_bottom_mean": float(np.mean([row["top_minus_bottom"] for row in state_metrics])),
        "top_minus_bottom_cluster_bootstrap_95ci": top_bottom_ci,
        "argmax_flip_fraction": flip_fraction,
        "task_median_spearman": task_direction,
        "top_minus_random_by_action_position": action_directions,
        "free_running_any_token_flip_fraction": float(np.mean([row["free_hamming"] > 0 for row in token_effects])),
        "free_running_action_l2_median": float(np.median([row["free_l2"] for row in token_effects])),
        "criteria": criteria, "criteria_passed": passed, "decision": decision,
    }
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"decision": decision, "criteria": criteria, "criteria_passed": passed}, indent=2, sort_keys=True) + "\n")
    report = f"""# AR Token Counterfactual Qualification v1

## Scope

Official OpenVLA-7B LIBERO-Spatial, post-projector single-token replacement. Attention is evaluated only as a ranking proxy, not causal attribution. Primary effects use identical clean action prefixes under teacher forcing.

## Integrity

- Raw rows: {len(rows)}/{expected_rows}; states: {len(state_metrics)}/150; token interventions: {len(token_effects)}/2400.
- Missing, duplicate, and nonfinite rows: 0.
- Pre-baseline model integrity: see `integrity_report.json`.

## Results

- State-level attention/effect Spearman median: {summary['state_spearman_median']:.4f}, cluster bootstrap 95% CI {rho_ci}.
- Top-minus-random mean JS: {summary['top_minus_random_mean']:.6g}, 95% CI {top_random_ci}.
- Top-minus-bottom mean JS: {summary['top_minus_bottom_mean']:.6g}, 95% CI {top_bottom_ci}.
- Action-position argmax flip fraction: {flip_fraction:.4%}.
- Free-running sequences with any token flip: {summary['free_running_any_token_flip_fraction']:.4%}.

## Decision

`{decision}` ({passed}/4 preregistered criteria passed).

This result concerns action-logit sensitivity in a frozen AR policy. It does not establish token contribution sign, closed-loop utility, or success improvement. Closed-loop calibration and VCAD remain prohibited unless the decision is `AR_TOKEN_GO`.
"""
    (artifact / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"decision": decision, "criteria_passed": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
