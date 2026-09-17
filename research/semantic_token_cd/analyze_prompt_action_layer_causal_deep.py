#!/usr/bin/env python3
"""Deep robustness analysis for the expanded prompt-to-action layer screen."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_action_layer_causal_expanded_v1"
SOURCE = REPO / "artifacts/prompt_attn_layer_selection_v1/states"
METRICS = (
    "top_alignment", "top_minus_random", "top_minus_bottom",
    "top_minus_wrong_prompt", "causal_margin",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bootstrap(values: list[float], seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def metadata_by_state() -> dict[str, dict]:
    result = {}
    for path in SOURCE.glob("**/step_*.json"):
        data = json.loads(path.read_text())
        result[data["state_id"]] = data
    return result


def main() -> None:
    metadata = metadata_by_state()
    rows = []
    for path in sorted((ROOT / "results").glob("*.json")):
        data = json.loads(path.read_text())
        state_meta = metadata[data["state_id"]]
        random_alignment = float(data["random"]["alignment"])
        for layer in data["layers"]:
            top = layer["branches"]["top"]
            bottom = layer["branches"]["bottom"]
            wrong = layer["branches"]["wrong_prompt"]
            top_set = set(top["selected_token_ids"])
            wrong_set = set(wrong["selected_token_ids"])
            overlap = len(top_set & wrong_set)
            rows.append({
                "state_id": data["state_id"], "task": data["task"].removeprefix("google_robot_"),
                "seed": int(data["seed"]), "split": data["split"], "phase": state_meta["phase"],
                "layer": int(layer["layer"]), "top_alignment": float(top["alignment"]),
                "bottom_alignment": float(bottom["alignment"]),
                "wrong_prompt_alignment": float(wrong["alignment"]),
                "random_alignment": random_alignment,
                "top_minus_random": float(layer["top_minus_random"]),
                "top_minus_bottom": float(layer["top_minus_bottom"]),
                "top_minus_wrong_prompt": float(layer["top_minus_wrong_prompt"]),
                "causal_margin": float(layer["causal_margin"]),
                "top_wrong_jaccard": overlap / max(1, len(top_set | wrong_set)),
                "top_wrong_replacement_count": len(top_set - wrong_set),
            })

    episode_groups = defaultdict(list)
    for row in rows:
        episode_groups[(row["split"], row["task"], row["seed"], row["layer"])].append(row)
    episodes = []
    for (split, task, seed, layer), items in sorted(episode_groups.items()):
        episodes.append({
            "split": split, "task": task, "seed": seed, "layer": layer,
            **{key: float(np.mean([item[key] for item in items])) for key in (
                *METRICS, "top_wrong_jaccard", "top_wrong_replacement_count"
            )},
        })

    taskwise = []
    for split in ("exploration", "validation"):
        for task in sorted({row["task"] for row in episodes}):
            for layer in range(32):
                items = [row for row in episodes if row["split"] == split and row["task"] == task and row["layer"] == layer]
                record = {"split": split, "task": task, "layer": layer, "episodes": len(items)}
                for metric in (*METRICS, "top_wrong_jaccard", "top_wrong_replacement_count"):
                    values = [row[metric] for row in items]
                    record[metric] = float(np.mean(values))
                    if metric == "causal_margin":
                        record[f"{metric}_ci95_low"], record[f"{metric}_ci95_high"] = bootstrap(
                            values, 20260915 + layer + sum(map(ord, task + split))
                        )
                taskwise.append(record)
    write_csv(ROOT / "TASKWISE_DEEP_RESULTS.csv", taskwise)

    phasewise = []
    for split in ("exploration", "validation"):
        for phase in ("early", "middle", "late"):
            for layer in range(32):
                items = [row for row in rows if row["split"] == split and row["phase"] == phase and row["layer"] == layer]
                phasewise.append({
                    "split": split, "phase": phase, "layer": layer, "states": len(items),
                    **{metric: float(np.mean([row[metric] for row in items])) for metric in METRICS},
                })
    write_csv(ROOT / "PHASEWISE_DEEP_RESULTS.csv", phasewise)

    aggregate = list(csv.DictReader((ROOT / "LAYER_RESULTS.csv").open()))
    rank_rows = []
    for split in ("exploration", "validation"):
        items = [row for row in aggregate if row["split"] == split]
        for metric in METRICS:
            ranked = sorted(items, key=lambda row: (-float(row[metric]), int(row["layer"])))
            for rank, row in enumerate(ranked, 1):
                rank_rows.append({
                    "split": split, "metric": metric, "layer": int(row["layer"]),
                    "rank": rank, "value": float(row[metric]),
                })
    write_csv(ROOT / "METRIC_RANKS.csv", rank_rows)

    rank_lookup = {(row["split"], row["metric"], row["layer"]): row["rank"] for row in rank_rows}
    robust = []
    for layer in range(32):
        ranks = [rank_lookup[(split, metric, layer)] for split in ("exploration", "validation") for metric in METRICS]
        robust.append({
            "layer": layer, "mean_rank": float(np.mean(ranks)), "worst_rank": max(ranks),
            "exploration_mean_rank": float(np.mean([rank_lookup[("exploration", metric, layer)] for metric in METRICS])),
            "validation_mean_rank": float(np.mean([rank_lookup[("validation", metric, layer)] for metric in METRICS])),
        })
    robust.sort(key=lambda row: (row["mean_rank"], row["worst_rank"], row["layer"]))
    write_csv(ROOT / "ROBUST_LAYER_RANKING.csv", robust)

    correlations = []
    for metric in METRICS:
        exploration = [next(float(row["value"]) for row in rank_rows if row["split"] == "exploration" and row["metric"] == metric and row["layer"] == layer) for layer in range(32)]
        validation = [next(float(row["value"]) for row in rank_rows if row["split"] == "validation" and row["metric"] == metric and row["layer"] == layer) for layer in range(32)]
        rho, p = spearmanr(exploration, validation)
        correlations.append({"metric": metric, "spearman_rho": float(rho), "p_value": float(p)})
    write_csv(ROOT / "SPLIT_CORRELATIONS.csv", correlations)

    chosen = (7, 8, 9, 10, 11, 13, 14, 16)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for axis, (split, task) in zip(axes.flat, [(s, t) for s in ("exploration", "validation") for t in ("move_near", "open_drawer", "pick_coke_can")]):
        values = [next(row for row in taskwise if row["split"] == split and row["task"] == task and row["layer"] == layer)["causal_margin"] for layer in chosen]
        axis.bar([f"L{x}" for x in chosen], values, color=["#c93434" if x == 11 else "#315b7d" for x in chosen])
        axis.axhline(0, color="black", linewidth=.8); axis.grid(axis="y", alpha=.2)
        axis.set_title(f"{split} — {task}")
    fig.supylabel("Episode-mean causal margin")
    fig.tight_layout(); fig.savefig(ROOT / "figure_taskwise_causal_margin.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(19, 4))
    aggregate_lookup = {(row["split"], int(row["layer"])): row for row in aggregate}
    for axis, metric in zip(axes, METRICS):
        x = [float(aggregate_lookup[("exploration", layer)][metric]) for layer in range(32)]
        y = [float(aggregate_lookup[("validation", layer)][metric]) for layer in range(32)]
        axis.scatter(x, y, color="#315b7d")
        axis.scatter(x[11], y[11], color="#c93434")
        axis.annotate("L11", (x[11], y[11]))
        axis.axhline(0, color="black", linewidth=.5); axis.axvline(0, color="black", linewidth=.5)
        axis.set_title(metric); axis.set_xlabel("exploration"); axis.grid(alpha=.2)
    axes[0].set_ylabel("validation")
    fig.tight_layout(); fig.savefig(ROOT / "figure_split_metric_stability.png", dpi=200); plt.close(fig)

    l11_task = [row for row in taskwise if row["layer"] == 11]
    lines = [
        "# Expanded action-level layer robustness audit", "",
        "## Direct conclusions", "",
        "- Robust mean-rank top 8: " + ", ".join(
            f"L{x['layer']} ({x['mean_rank']:.1f})" for x in robust[:8]
        ) + ".",
        f"- L11 robust mean rank: {next(x['mean_rank'] for x in robust if x['layer'] == 11):.1f}/32; worst rank across split/metrics: {next(x['worst_rank'] for x in robust if x['layer'] == 11)}/32.",
        "- No validation layer has positive aggregate causal margin against all three controls.",
        "- Therefore the strict causal-margin criterion does not identify a stable winner.", "",
        "## Split rank correlations", "",
        "| Metric | Spearman rho | p |", "|---|---:|---:|",
    ]
    for row in correlations:
        lines.append(f"| {row['metric']} | {row['spearman_rho']:.3f} | {row['p_value']:.3g} |")
    lines.extend(["", "## L11 taskwise", "", "| Split | Task | Top alignment | Margin | Top-vs-wrong | Mask Jaccard |", "|---|---|---:|---:|---:|---:|"])
    for row in l11_task:
        lines.append(f"| {row['split']} | {row['task']} | {row['top_alignment']:.4f} | {row['causal_margin']:+.4f} | {row['top_minus_wrong_prompt']:+.4f} | {row['top_wrong_jaccard']:.3f} |")
    lines.extend([
        "", "## Interpretation", "",
        "- L11 is usually strong on raw action alignment and against Random/Bottom controls.",
        "- Its strict margin is weakened because the alternate-Prompt mask often produces a similarly aligned action residual.",
        "- This means the current metric verifies Prompt-sensitive action mediation, but does not uniquely validate the original-Prompt Top-m mask.",
        "- L11 remains the best tested closed-loop single layer only because other single layers have not yet received closed-loop evaluation.",
        "", "![Taskwise](figure_taskwise_causal_margin.png)", "", "![Split stability](figure_split_metric_stability.png)",
    ])
    (ROOT / "DEEP_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"robust_top8": robust[:8], "correlations": correlations}, indent=2))


if __name__ == "__main__":
    main()
