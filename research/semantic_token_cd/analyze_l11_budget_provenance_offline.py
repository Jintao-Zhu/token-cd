#!/usr/bin/env python3
"""Analyze the offline L11 Matched budget-provenance replay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ARMS = (
    "true_matched",
    "wrong_entity",
    "random_cluster",
    "source_only",
    "target_only",
    "top_p80",
    "true_scale_150",
)
LABELS = {
    "true_matched": "True Matched",
    "wrong_entity": "Wrong Entity",
    "random_cluster": "Random Cluster",
    "source_only": "Source Only",
    "target_only": "Target Only",
    "top_p80": "Top-p80",
    "true_scale_150": "Matched x1.5",
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    paths = sorted((artifact / "states").glob("*/seed_*/*.json"))
    if len(paths) != 60:
        raise RuntimeError(f"expected 60 completed states, got {len(paths)}")

    rows: list[dict] = []
    for path in paths:
        item = json.loads(path.read_text())
        if not item["technical"]["matched_count_exact"]:
            raise RuntimeError(f"matched count audit failed: {path}")
        if item["technical"]["l11_attention_max_abs_diff"] != 0.0:
            raise RuntimeError(f"attention replay audit failed: {path}")
        for arm in ARMS:
            metric = item["metrics"][arm]
            rows.append({
                "state_id": item["state_id"],
                "task": item["task"],
                "seed": item["seed"],
                "step": item["step"],
                "phase": item["phase"],
                "true_entity_count": len(item["true_entities"]),
                "true_distinct_group_count": len(item["true_group_ids"]),
                "arm": arm,
                "count": metric["count"],
                "selected_attention_mass": metric["selected_attention_mass"],
                "feature_perturbation_relative": metric["feature_perturbation_relative"],
                "jaccard_vs_true": metric["jaccard_vs_true"],
            })

    fields = list(rows[0])
    with (artifact / "STATE_ARM_METRICS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_state = {item["state_id"]: item for item in map(lambda p: json.loads(p.read_text()), paths)}
    true_counts = np.asarray([
        by_state[state]["metrics"]["true_matched"]["count"] for state in sorted(by_state)
    ], dtype=float)
    summary: dict[str, dict] = {}
    aggregate_rows: list[dict] = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        counts = np.asarray([row["count"] for row in selected], dtype=float)
        perturbation = np.asarray([row["feature_perturbation_relative"] for row in selected])
        differences = counts - true_counts
        correlation = spearmanr(counts, true_counts)
        dose_correlation = spearmanr(counts, perturbation)
        summary[arm] = {
            "states": len(selected),
            "mean_count": float(counts.mean()),
            "std_count": float(counts.std()),
            "min_count": int(counts.min()),
            "max_count": int(counts.max()),
            "mean_abs_count_difference_vs_true": float(np.abs(differences).mean()),
            "same_count_rate_vs_true": float(np.mean(differences == 0)),
            "spearman_count_vs_true": float(correlation.statistic),
            "spearman_count_vs_corruption": float(dose_correlation.statistic),
            "count_vs_corruption_p": float(dose_correlation.pvalue),
            "mean_feature_perturbation_relative": float(perturbation.mean()),
            "mean_jaccard_vs_true": float(np.mean([row["jaccard_vs_true"] for row in selected])),
        }
        aggregate_rows.append({"arm": arm, **summary[arm]})

    task_rows: list[dict] = []
    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        for arm in ARMS:
            selected = [row for row in rows if row["task"] == task and row["arm"] == arm]
            task_rows.append({
                "task": task,
                "arm": arm,
                "states": len(selected),
                "mean_count": float(np.mean([row["count"] for row in selected])),
                "mean_abs_count_difference_vs_true": float(np.mean([
                    abs(row["count"] - by_state[row["state_id"]]["metrics"]["true_matched"]["count"])
                    for row in selected
                ])),
                "mean_feature_perturbation_relative": float(np.mean([
                    row["feature_perturbation_relative"] for row in selected
                ])),
            })

    for filename, data in (
        ("AGGREGATE_RESULTS.csv", aggregate_rows),
        ("TASK_RESULTS.csv", task_rows),
    ):
        with (artifact / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    component = {
        "single_entity_states": sum(len(item["true_entities"]) == 1 for item in by_state.values()),
        "two_entity_states": sum(len(item["true_entities"]) == 2 for item in by_state.values()),
        "two_entities_same_group_states": sum(item["entities_share_group"] for item in by_state.values()),
    }
    payload = {
        "complete": True,
        "created_date": "2026-09-16",
        "states": len(by_state),
        "summary": summary,
        "entity_structure": component,
        "technical": {
            "all_matched_counts_exact": True,
            "max_attention_replay_difference": 0.0,
            "same_l11_ranking_all_arms": True,
            "closed_loop_used": False,
            "action_metrics_used": False,
        },
    }
    atomic_json(artifact / "FINAL_RESULTS.json", payload)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    positions = np.arange(len(ARMS))
    axes[0].bar(positions, [summary[arm]["mean_count"] for arm in ARMS], color="#4776b4")
    axes[0].set_xticks(positions, [LABELS[arm] for arm in ARMS], rotation=30, ha="right")
    axes[0].set_ylabel("Mean selected-token budget")
    axes[0].set_title("Budget provenance on identical states")
    axes[1].bar(positions, [summary[arm]["mean_abs_count_difference_vs_true"] for arm in ARMS], color="#d17457")
    axes[1].set_xticks(positions, [LABELS[arm] for arm in ARMS], rotation=30, ha="right")
    axes[1].set_ylabel("Mean |count - True Matched|")
    axes[1].set_title("How much each control changes intervention dose")
    figure.tight_layout()
    figure.savefig(artifact / "figure_budget_provenance.png", dpi=180)
    plt.close(figure)

    lines = [
        "# L11 Matched Budget Provenance: Offline Audit",
        "",
        "同一60个保存状态；不跑环境闭环，不使用动作指标。所有 arm 使用同一条 L11 排名，只改变预算来源。",
        "",
        "## 技术审计",
        "",
        "- 60/60 Matched 数量逐状态精确复现。",
        "- 60/60 L11 attention 与历史缓存 bit-exact（最大差异 0）。",
        "- 所有对照都选择 L11 Top-m；因此 mask 差异只来自 m。",
        "",
        "## 汇总",
        "",
        "| Budget source | Mean count | Mean abs diff vs true | Same-count rate | Spearman vs true | Count→corruption ρ | Mean corruption |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = summary[arm]
        lines.append(
            f"| {LABELS[arm]} | {item['mean_count']:.2f} | "
            f"{item['mean_abs_count_difference_vs_true']:.2f} | "
            f"{item['same_count_rate_vs_true']:.1%} | "
            f"{item['spearman_count_vs_true']:.3f} | "
            f"{item['spearman_count_vs_corruption']:.3f} | "
            f"{item['mean_feature_perturbation_relative']:.4f} |"
        )
    lines.extend([
        "",
        "## 边界",
        "",
        "本报告只回答不同预算来源在同一状态上产生什么数量和 feature-corruption 差异。",
        "它不使用成功结果，因此不能单独证明哪一种预算提高闭环成功率；该因果问题必须由冻结闭环对照回答。",
        "",
        "![Budget provenance](figure_budget_provenance.png)",
    ])
    (artifact / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    atomic_json(artifact / "COMPLETE.json", {"complete": True, "states": 60})
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
