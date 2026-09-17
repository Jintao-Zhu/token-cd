"""Offline audit of current-step positive confidence versus L11 CD intervention."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from research.semantic_token_cd.audit_prompt_attn_l11_risk_signal import (
    SHORT,
    SOURCE_BY_TASK,
    TASKS,
    classify,
    percentile,
    write_csv,
)
from research.semantic_token_cd.prompt_attn_l11_risk_gated_policy import (
    PROBE_LAMBDA,
    openvla_normalized_token_values,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


BOOTSTRAP_SAMPLES = 2000
EPS = 1e-12
STEP_METRICS = (
    "positive_entropy",
    "positive_confidence",
    "positive_margin",
    "intervention_magnitude",
    "confidence_x_magnitude",
    "margin_x_magnitude",
    "positive_negative_js",
)
RISK_METRICS = (
    "positive_confidence",
    "positive_margin",
    "intervention_magnitude",
    "confidence_x_magnitude",
    "margin_x_magnitude",
    "positive_negative_js",
)
PRIMARY_METRICS = ("confidence_x_magnitude", "margin_x_magnitude")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def expected_actions(probabilities: np.ndarray) -> np.ndarray:
    return probabilities @ openvla_normalized_token_values()


def step_metrics(positive_logits: np.ndarray, negative_logits: np.ndarray) -> dict[str, np.ndarray]:
    positive_probabilities = softmax(positive_logits)
    negative_probabilities = softmax(negative_logits)
    probe_logits = (1.0 + PROBE_LAMBDA) * positive_logits - PROBE_LAMBDA * negative_logits
    probe_probabilities = softmax(probe_logits)

    normalized_entropy = -np.sum(
        positive_probabilities * np.log(np.maximum(positive_probabilities, EPS)), axis=-1
    ) / np.log(positive_probabilities.shape[-1])
    entropy = normalized_entropy.mean(axis=-1)
    confidence = 1.0 - entropy
    top_two = np.partition(positive_probabilities, -2, axis=-1)[..., -2:]
    margin = (top_two[..., 1] - top_two[..., 0]).mean(axis=-1)
    positive_actions = expected_actions(positive_probabilities)
    probe_actions = expected_actions(probe_probabilities)
    intervention_magnitude = np.linalg.norm(probe_actions - positive_actions, axis=-1)

    midpoint = 0.5 * (positive_probabilities + negative_probabilities)
    positive_kl = np.sum(
        positive_probabilities
        * (np.log(np.maximum(positive_probabilities, EPS)) - np.log(np.maximum(midpoint, EPS))),
        axis=-1,
    )
    negative_kl = np.sum(
        negative_probabilities
        * (np.log(np.maximum(negative_probabilities, EPS)) - np.log(np.maximum(midpoint, EPS))),
        axis=-1,
    )
    js = (0.5 * positive_kl + 0.5 * negative_kl).mean(axis=-1) / np.log(2.0)
    return {
        "positive_entropy": entropy,
        "positive_confidence": confidence,
        "positive_margin": margin,
        "intervention_magnitude": intervention_magnitude,
        "confidence_x_magnitude": confidence * intervention_magnitude,
        "margin_x_magnitude": margin * intervention_magnitude,
        "positive_negative_js": js,
    }


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        values.append(float(roc_auc_score(labels[sample], scores[sample])))
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "p2_5": percentile(values, 2.5),
        "median": percentile(values, 50),
        "p97_5": percentile(values, 97.5),
    }


def aggregate_episode(values: np.ndarray, metric: str) -> dict:
    return {
        f"{metric}_mean": float(np.mean(values)),
        f"{metric}_p90": percentile(values.tolist(), 90),
        f"{metric}_max": float(np.max(values)),
    }


def distribution_summary(episodes: list[dict], category: str, metric: str) -> dict:
    values = [episode[f"{metric}_p90"] for episode in episodes if episode["category"] == category]
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p25": percentile(values, 25) if values else None,
        "p75": percentile(values, 75) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=49)
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifact = args.artifact.resolve()
    baseline_csv = repo / "artifacts/prompt_attn_l11_matched_full_9task_0_299_v1/paired_results_0_299.csv"
    baselines = {}
    with baseline_csv.open() as handle:
        for row in csv.DictReader(handle):
            baselines[(row["task"], int(row["seed"]))] = {
                "vanilla": row["vanilla"] == "True",
                "shr": row["shr"] == "True",
                "l11": row["l11_matched"] == "True",
            }

    episodes = []
    steps = []
    for task in TASKS:
        source = repo / "artifacts" / SOURCE_BY_TASK[task] / "episodes" / task / "prompt_single"
        for seed in range(args.seed_start, args.seed_end + 1):
            summary_path = source / f"episode_{seed:03d}_summary.json"
            arrays_path = source / f"episode_{seed:03d}_arrays.npz"
            summary = json.loads(summary_path.read_text())
            arrays = np.load(arrays_path)
            baseline = baselines[(task, seed)]
            if summary.get("technical_pass") is not True or bool(summary["success"]) != baseline["l11"]:
                raise RuntimeError(f"fixed L11 audit mismatch: {task} seed={seed}")
            positive_logits = arrays["positive"][:, :6].astype(np.float64)
            negative_logits = arrays["negative"][:, :6].astype(np.float64)
            metrics = step_metrics(positive_logits, negative_logits)
            category = classify(baseline["vanilla"], baseline["l11"])
            episode = {
                "task": task,
                "task_short": SHORT[task],
                "seed": seed,
                **baseline,
                "category": category,
                "control_steps": positive_logits.shape[0],
            }
            for metric in STEP_METRICS:
                episode.update(aggregate_episode(metrics[metric], metric))
            episodes.append(episode)
            for step_index in range(positive_logits.shape[0]):
                steps.append({
                    "task": SHORT[task],
                    "seed": seed,
                    "category": category,
                    "step_index": step_index,
                    **{metric: float(metrics[metric][step_index]) for metric in STEP_METRICS},
                })

    comparison = [episode for episode in episodes if episode["category"] in ("fixed_cd_harm", "rescue")]
    labels = np.asarray([episode["category"] == "fixed_cd_harm" for episode in comparison], dtype=int)
    auc_rows = []
    for metric_index, metric in enumerate(RISK_METRICS):
        for aggregation in ("mean", "p90", "max"):
            key = f"{metric}_{aggregation}"
            scores = np.asarray([episode[key] for episode in comparison], dtype=float)
            auc = float(roc_auc_score(labels, scores))
            bootstrap = bootstrap_auc(labels, scores, 20260913 + 10 * metric_index + len(aggregation))
            leave_one_task_out = {}
            for held_out in TASKS:
                subset = [episode for episode in comparison if episode["task"] != held_out]
                subset_labels = np.asarray(
                    [episode["category"] == "fixed_cd_harm" for episode in subset], dtype=int
                )
                if len(np.unique(subset_labels)) < 2:
                    leave_one_task_out[SHORT[held_out]] = None
                else:
                    leave_one_task_out[SHORT[held_out]] = float(roc_auc_score(
                        subset_labels, [episode[key] for episode in subset]
                    ))
            auc_rows.append({
                "metric": metric,
                "aggregation": aggregation,
                "auc_harm_vs_rescue": auc,
                "bootstrap_p2_5": bootstrap["p2_5"],
                "bootstrap_median": bootstrap["median"],
                "bootstrap_p97_5": bootstrap["p97_5"],
                "leave_one_task_out": leave_one_task_out,
            })

    primary_rows = [
        row for row in auc_rows
        if row["metric"] in PRIMARY_METRICS and row["aggregation"] == "p90"
    ]
    best_primary = max(primary_rows, key=lambda row: row["auc_harm_vs_rescue"])
    go_no_go = {
        "primary_auc_at_least_0_65": best_primary["auc_harm_vs_rescue"] >= 0.65,
        "best_primary_metric": best_primary["metric"],
        "best_primary_auc": best_primary["auc_harm_vs_rescue"],
    }
    go_no_go["passed"] = go_no_go["primary_auc_at_least_0_65"]

    distributions = {
        metric: {
            category: distribution_summary(episodes, category, metric)
            for category in ("fixed_cd_harm", "rescue", "stable_success", "stable_fail")
        }
        for metric in STEP_METRICS
    }
    write_csv(artifact / "confidence_audit/episode_metrics.csv", episodes)
    write_csv(artifact / "confidence_audit/per_step_metrics.csv", steps)
    write_csv(artifact / "confidence_audit/auc_results.csv", [
        {key: value for key, value in row.items() if key != "leave_one_task_out"}
        for row in auc_rows
    ])
    payload = {
        "protocol_id": "PROMPT_ATTN_L11_CURRENT_CONFIDENCE_AUDIT_V1",
        "discovery_seeds": [args.seed_start, args.seed_end],
        "tasks": list(TASKS),
        "episode_count": len(episodes),
        "category_counts": dict(Counter(episode["category"] for episode in episodes)),
        "primary_metrics": list(PRIMARY_METRICS),
        "go_no_go": go_no_go,
        "auc_results": auc_rows,
        "p90_distributions": distributions,
    }
    atomic_json(artifact / "confidence_audit/CURRENT_CONFIDENCE_AUDIT.json", payload)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_categories = ("fixed_cd_harm", "rescue", "stable_success")
    colors = {"fixed_cd_harm": "tab:red", "rescue": "tab:green", "stable_success": "tab:blue"}
    for axis, metric in zip(axes, PRIMARY_METRICS):
        for category in plot_categories:
            values = [episode[f"{metric}_p90"] for episode in episodes if episode["category"] == category]
            axis.hist(values, bins=15, alpha=0.45, density=True, color=colors[category], label=category)
        axis.set(xlabel=f"Episode P90 {metric}", ylabel="Density")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(artifact / "confidence_audit/confidence_intervention_distributions.png", dpi=180)
    plt.close(figure)

    lines = [
        "# L11 Current-step Confidence × Intervention Audit",
        "",
        f"Discovery set: four tasks, seeds {args.seed_start}--{args.seed_end}, {len(episodes)} fixed-L11 episodes.",
        "",
        f"Category counts: `{payload['category_counts']}`",
        "",
        "| Metric | Aggregation | Harm-vs-Rescue AUC | Bootstrap 95% interval |",
        "|---|---|---:|---:|",
    ]
    for row in sorted(auc_rows, key=lambda value: value["auc_harm_vs_rescue"], reverse=True):
        lines.append(
            f'| {row["metric"]} | {row["aggregation"]} | {row["auc_harm_vs_rescue"]:.4f} | '
            f'[{row["bootstrap_p2_5"]:.4f}, {row["bootstrap_p97_5"]:.4f}] |'
        )
    lines.extend([
        "",
        f"Primary decision: **{'PASS' if go_no_go['passed'] else 'STOP'}**.",
        f"Best primary P90 metric: `{best_primary['metric']}` with AUC **{best_primary['auc_harm_vs_rescue']:.4f}**.",
    ])
    (artifact / "confidence_audit/CURRENT_CONFIDENCE_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
