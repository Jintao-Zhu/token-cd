from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .common import file_sha256, read_jsonl, write_json


CONDITIONS = ("vanilla", "mask_top8_effect", "mask_bottom8_effect", "mask_random8")


def hierarchical_sample_indices(task_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    tasks = np.unique(task_ids)
    sampled = []
    for task in rng.choice(tasks, size=len(tasks), replace=True):
        indices = np.flatnonzero(task_ids == task)
        sampled.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
    return np.asarray(sampled, dtype=int)


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest

        p = float(binomtest(min(a_only, b_only), n=n, p=0.5, alternative="two-sided").pvalue)
    return {"first_only": a_only, "second_only": b_only, "discordant": n, "exact_p": p}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    manifest = read_jsonl(artifact / "rollout_manifest.lock.jsonl")
    expected = {row["episode_id"] for row in manifest}
    files = sorted((artifact / "episodes").glob("*.json"))
    if len(files) != 300:
        raise RuntimeError(f"Refusing partial analysis: {len(files)}/300 episode files")
    results = [json.loads(path.read_text()) for path in files]
    actual = {row["episode_id"] for row in results}
    if actual != expected or len(actual) != len(results):
        raise RuntimeError("Missing or duplicate episode identities")
    if any(not row["all_actions_finite"] or not row["single_first_action_intervention_verified"] for row in results):
        raise RuntimeError("Episode integrity flag failure")
    by_snapshot: dict[str, dict[str, dict]] = {}
    for row in results:
        by_snapshot.setdefault(row["snapshot_id"], {})[row["condition"]] = row
    if len(by_snapshot) != 75 or any(set(group) != set(CONDITIONS) for group in by_snapshot.values()):
        raise RuntimeError("Incomplete paired snapshots")

    paired = []
    for snapshot_id, group in sorted(by_snapshot.items()):
        vanilla = group["vanilla"]
        for condition in CONDITIONS[1:]:
            treated = group[condition]
            paired.append({
                "snapshot_id": snapshot_id,
                "task_id": vanilla["task_id"],
                "condition": condition,
                "offline_effect": treated["offline_effect"],
                "vanilla_success": int(vanilla["success"]),
                "treated_success": int(treated["success"]),
                "success_discordance": int(vanilla["success"] != treated["success"]),
                "object_goal_abs_effect": abs(treated["final_progress"]["object_goal_distance"] - vanilla["final_progress"]["object_goal_distance"]),
                "eef_object_abs_effect": abs(treated["final_progress"]["eef_object_distance"] - vanilla["final_progress"]["eef_object_distance"]),
                "grasp_discordance": int(treated["final_progress"]["grasped"] != vanilla["final_progress"]["grasped"]),
                "control_step_delta": treated["control_steps"] - vanilla["control_steps"],
                "action_total_variation_delta": treated["action_total_variation"] - vanilla["action_total_variation"],
                "latency_delta_ms": treated["mean_inference_latency_ms"] - vanilla["mean_inference_latency_ms"],
            })
    columns = list(paired[0])
    with (artifact / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(paired)

    snapshots = sorted(by_snapshot)
    task_ids = np.asarray([by_snapshot[s]["vanilla"]["task_id"] for s in snapshots])
    metrics = {}
    for condition in CONDITIONS[1:]:
        rows = [next(row for row in paired if row["snapshot_id"] == s and row["condition"] == condition) for s in snapshots]
        metrics[condition] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("success_discordance", "object_goal_abs_effect", "eef_object_abs_effect", "grasp_discordance")
        }
    rng = np.random.default_rng(20260809)
    boot_success_max_near, boot_success_max_random = [], []
    boot_goal_max_near, boot_goal_max_random, boot_rho = [], [], []
    condition_rows = {c: [next(row for row in paired if row["snapshot_id"] == s and row["condition"] == c) for s in snapshots] for c in CONDITIONS[1:]}
    for _ in range(2000):
        indices = hierarchical_sample_indices(task_ids, rng)
        def mean(c: str, key: str) -> float:
            return float(np.mean([condition_rows[c][i][key] for i in indices]))
        boot_success_max_near.append(mean("mask_top8_effect", "success_discordance") - mean("mask_bottom8_effect", "success_discordance"))
        boot_success_max_random.append(mean("mask_top8_effect", "success_discordance") - mean("mask_random8", "success_discordance"))
        boot_goal_max_near.append(mean("mask_top8_effect", "object_goal_abs_effect") - mean("mask_bottom8_effect", "object_goal_abs_effect"))
        boot_goal_max_random.append(mean("mask_top8_effect", "object_goal_abs_effect") - mean("mask_random8", "object_goal_abs_effect"))
        flat = [paired_row for i in indices for paired_row in (condition_rows[c][i] for c in CONDITIONS[1:])]
        rho = spearmanr([row["offline_effect"] for row in flat], [row["object_goal_abs_effect"] for row in flat]).statistic
        if np.isfinite(rho):
            boot_rho.append(float(rho))
    all_nonvanilla = [row for row in paired]
    rho = float(spearmanr([row["offline_effect"] for row in all_nonvanilla], [row["object_goal_abs_effect"] for row in all_nonvanilla]).statistic)
    comparisons = {
        "success_top8_minus_bottom8": {"point": metrics["mask_top8_effect"]["success_discordance"] - metrics["mask_bottom8_effect"]["success_discordance"], "ci95": interval(boot_success_max_near)},
        "success_top8_minus_random8": {"point": metrics["mask_top8_effect"]["success_discordance"] - metrics["mask_random8"]["success_discordance"], "ci95": interval(boot_success_max_random)},
        "object_goal_top8_minus_bottom8": {"point": metrics["mask_top8_effect"]["object_goal_abs_effect"] - metrics["mask_bottom8_effect"]["object_goal_abs_effect"], "ci95": interval(boot_goal_max_near)},
        "object_goal_top8_minus_random8": {"point": metrics["mask_top8_effect"]["object_goal_abs_effect"] - metrics["mask_random8"]["object_goal_abs_effect"], "ci95": interval(boot_goal_max_random)},
        "offline_effect_vs_object_goal_abs_spearman": {"point": rho, "ci95": interval(boot_rho)},
    }
    criteria = {
        "success_magnitude": comparisons["success_top8_minus_bottom8"]["ci95"][0] > 0 and comparisons["success_top8_minus_random8"]["ci95"][0] > 0,
        "object_goal_magnitude": comparisons["object_goal_top8_minus_bottom8"]["ci95"][0] > 0 and comparisons["object_goal_top8_minus_random8"]["ci95"][0] > 0,
        "continuous_predictiveness": rho >= 0.30 and comparisons["offline_effect_vs_object_goal_abs_spearman"]["ci95"][0] > 0,
    }
    passed = sum(criteria.values())
    decision = "AR_CLOSED_LOOP_MAGNITUDE_GO" if passed >= 2 else "AR_CLOSED_LOOP_MAGNITUDE_NO_GO"
    success = {condition: float(np.mean([group[condition]["success"] for group in by_snapshot.values()])) for condition in CONDITIONS}
    mcnemar = {condition: exact_mcnemar(
        np.asarray([group["vanilla"]["success"] for group in by_snapshot.values()], dtype=int),
        np.asarray([group[condition]["success"] for group in by_snapshot.values()], dtype=int),
    ) for condition in CONDITIONS[1:]}
    summary = {
        "completion": {"episodes": 300, "paired_snapshots": 75, "tasks": 3, "missing": 0, "duplicates": 0},
        "success_rates": success,
        "condition_magnitude_metrics": metrics,
        "comparisons": comparisons,
        "mcnemar_vs_vanilla": mcnemar,
        "go_criteria": criteria,
        "criteria_passed": passed,
        "decision": decision,
    }
    write_json(artifact / "summary.json", summary)
    write_json(artifact / "decision.json", {"decision": decision, "criteria_passed": passed, "criteria": criteria})
    report = "# AR Token Closed-loop Causal Magnitude Calibration\n\n"
    report += f"Completed 300/300 rollouts over 75 fully paired snapshots and tasks 2, 9, 3. This is development calibration, not independent confirmation.\n\n"
    report += "## Success rates\n\n" + "\n".join(f"- {key}: {value:.1%}" for key, value in success.items()) + "\n\n"
    report += "## Locked causal magnitude tests\n\n```json\n" + json.dumps(comparisons, indent=2) + "\n```\n\n"
    report += "## Decision\n\n" + f"**{decision}** ({passed}/3 locked criteria passed). No sign claim is made from effect magnitude.\n"
    (artifact / "report.md").write_text(report, encoding="utf-8")
    audit_paths = [path for path in artifact.rglob("*") if path.is_file() and path.name != "sha256_audit.json"]
    write_json(artifact / "sha256_audit.json", {str(path.relative_to(artifact)): file_sha256(path) for path in sorted(audit_paths)})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
