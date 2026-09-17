#!/usr/bin/env python3
"""Aggregate semantic invariance/selectivity and join action-alignment metrics."""
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
ROOT = REPO / "artifacts/prompt_action_layer_semantic_consistency_v1"
ACTION_ROOT = REPO / "artifacts/prompt_action_layer_causal_expanded_v1"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    files = sorted((ROOT / "results").glob("*.json"))
    if len(files) != 90:
        raise RuntimeError(f"expected 90 results, got {len(files)}")
    action_rows = list(csv.DictReader((ACTION_ROOT / "STATEWISE_RESULTS.csv").open()))
    action = {(row["state_id"], int(row["layer"])): row for row in action_rows}
    state_rows = []
    for path in files:
        data = json.loads(path.read_text())
        for layer in data["layers"]:
            prior = action[(data["state_id"], int(layer["layer"]))]
            state_rows.append({
                "state_id": data["state_id"], "task": data["task"].removeprefix("google_robot_"),
                "seed": int(data["seed"]), "split": data["split"], "layer": int(layer["layer"]),
                "top_alignment": float(prior["top_alignment"]),
                "top_minus_random": float(prior["top_minus_random"]),
                "synonym_residual_cosine": float(layer["synonym_residual_cosine"]),
                "target_residual_cosine": float(layer["target_residual_cosine"]),
                "semantic_separation": float(layer["semantic_separation"]),
                "synonym_mask_jaccard": float(layer["synonym_mask_jaccard"]),
                "target_mask_jaccard": float(layer["target_mask_jaccard"]),
                "mask_jaccard_separation": float(layer["mask_jaccard_separation"]),
            })
    write_csv(ROOT / "STATEWISE_RESULTS.csv", state_rows)

    groups = defaultdict(list)
    for row in state_rows:
        groups[(row["split"], row["task"], row["seed"], row["layer"])].append(row)
    episodes = []
    metrics = (
        "top_alignment", "top_minus_random", "synonym_residual_cosine", "target_residual_cosine",
        "semantic_separation", "synonym_mask_jaccard", "target_mask_jaccard", "mask_jaccard_separation",
    )
    for (split, task, seed, layer), items in sorted(groups.items()):
        episodes.append({
            "split": split, "task": task, "seed": seed, "layer": layer,
            **{metric: float(np.mean([row[metric] for row in items])) for metric in metrics},
        })
    write_csv(ROOT / "EPISODE_RESULTS.csv", episodes)

    aggregate = []
    for split in ("exploration", "validation"):
        for layer in range(32):
            items = [row for row in episodes if row["split"] == split and row["layer"] == layer]
            aggregate.append({
                "split": split, "layer": layer, "episodes": len(items),
                **{metric: float(np.mean([row[metric] for row in items])) for metric in metrics},
            })

    ranked_metrics = (
        "top_alignment", "top_minus_random", "synonym_residual_cosine", "semantic_separation",
    )
    rank_lookup = {}
    for split in ("exploration", "validation"):
        items = [row for row in aggregate if row["split"] == split]
        for metric in ranked_metrics:
            for rank, row in enumerate(sorted(items, key=lambda x: (-x[metric], x["layer"])), 1):
                rank_lookup[(split, metric, row["layer"])] = rank
    for row in aggregate:
        row["mean_rank"] = float(np.mean([
            rank_lookup[(row["split"], metric, row["layer"])] for metric in ranked_metrics
        ]))
    write_csv(ROOT / "LAYER_RESULTS.csv", aggregate)

    robust = []
    for layer in range(32):
        ranks = [rank_lookup[(split, metric, layer)] for split in ("exploration", "validation") for metric in ranked_metrics]
        robust.append({"layer": layer, "mean_rank": float(np.mean(ranks)), "worst_rank": max(ranks)})
    robust.sort(key=lambda row: (row["mean_rank"], row["worst_rank"], row["layer"]))
    write_csv(ROOT / "ROBUST_RANKING.csv", robust)

    correlations = []
    by = {(row["split"], row["layer"]): row for row in aggregate}
    for metric in ranked_metrics:
        rho, p = spearmanr(
            [by[("exploration", layer)][metric] for layer in range(32)],
            [by[("validation", layer)][metric] for layer in range(32)],
        )
        correlations.append({"metric": metric, "spearman_rho": float(rho), "p_value": float(p)})
    write_csv(ROOT / "SPLIT_CORRELATIONS.csv", correlations)

    chosen = [row["layer"] for row in robust[:10]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, split in zip(axes, ("exploration", "validation")):
        items = [by[(split, layer)] for layer in chosen]
        x = np.arange(len(items))
        axis.bar(x - .2, [row["synonym_residual_cosine"] for row in items], .4, label="synonym residual cosine")
        axis.bar(x + .2, [row["target_residual_cosine"] for row in items], .4, label="changed-target residual cosine")
        axis.set_xticks(x, [f"L{row['layer']}" for row in items]); axis.set_title(split)
        axis.grid(axis="y", alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(ROOT / "figure_semantic_consistency.png", dpi=200); plt.close(fig)

    l11 = {split: by[(split, 11)] for split in ("exploration", "validation")}
    lines = [
        "# Box-free action-level semantic consistency audit", "",
        "## Question", "",
        "Does a layer produce similar intervention effects for paraphrases, but different effects when the task target changes? No image boxes are used.", "",
        "## Robust ranking", "",
        "| Rank | Layer | Mean rank | Worst rank |", "|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(robust[:12], 1):
        lines.append(f"| {rank} | L{row['layer']} | {row['mean_rank']:.2f} | {row['worst_rank']} |")
    lines.extend(["", "## L11", "", "| Split | Action alignment | Synonym cosine | Target cosine | Separation |", "|---|---:|---:|---:|---:|"])
    for split in ("exploration", "validation"):
        row = l11[split]
        lines.append(f"| {split} | {row['top_alignment']:.4f} | {row['synonym_residual_cosine']:.4f} | {row['target_residual_cosine']:.4f} | {row['semantic_separation']:+.4f} |")
    lines.extend(["", "## Split correlations", "", "| Metric | rho | p |", "|---|---:|---:|"])
    for row in correlations:
        lines.append(f"| {row['metric']} | {row['spearman_rho']:.3f} | {row['p_value']:.3g} |")
    lines.extend(["", "![Semantic consistency](figure_semantic_consistency.png)", "", "## Boundary", "", "This remains an offline action-level proxy. It can shortlist single layers but cannot replace closed-loop success evaluation."])
    (ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"robust_top10": robust[:10], "l11": l11, "correlations": correlations}, indent=2))


if __name__ == "__main__":
    main()
