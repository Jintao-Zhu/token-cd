#!/usr/bin/env python3
"""Aggregate the box-free all-layer semantic-selectivity scan."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_attention_semantic_layer_scan_v1"
METRICS = (
    "real_paraphrase_distance", "real_semantic_distance", "real_irrelevant_distance",
    "real_semantic_margin", "real_semantic_ratio", "mismatch_paraphrase_distance",
    "mismatch_semantic_distance", "mismatch_irrelevant_distance", "mismatch_semantic_margin",
    "grounded_semantic_margin", "real_visual_attention_mass",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    files = sorted((ROOT / "results").glob("*.json"))
    if len(files) != 90:
        raise RuntimeError(f"expected 90 states, got {len(files)}")
    states = []
    max_reproduction = 0.0
    for path in files:
        data = json.loads(path.read_text())
        max_reproduction = max(max_reproduction, data["attention_reproduction_max_abs_difference"])
        for layer in data["layers"]:
            states.append({
                "state_id": data["state_id"], "task": data["task"].removeprefix("google_robot_"),
                "seed": data["seed"], "split": data["split"], "layer": layer["layer"],
                **{metric: layer[metric] for metric in METRICS},
            })
    write_csv(ROOT / "STATEWISE_RESULTS.csv", states)

    groups = defaultdict(list)
    for row in states:
        groups[(row["split"], row["task"], row["seed"], row["layer"])].append(row)
    episodes = []
    for (split, task, seed, layer), items in sorted(groups.items()):
        episodes.append({
            "split": split, "task": task, "seed": seed, "layer": layer,
            **{metric: float(np.mean([row[metric] for row in items])) for metric in METRICS},
        })
    write_csv(ROOT / "EPISODE_RESULTS.csv", episodes)

    aggregate = []
    ranks = {}
    for split in ("exploration", "validation"):
        split_rows = []
        for layer in range(32):
            items = [row for row in episodes if row["split"] == split and row["layer"] == layer]
            split_rows.append({
                "split": split, "layer": layer, "episodes": len(items),
                **{metric: float(np.mean([row[metric] for row in items])) for metric in METRICS},
            })
        for metric in ("real_semantic_margin", "grounded_semantic_margin"):
            for rank, row in enumerate(sorted(split_rows, key=lambda item: -item[metric]), 1):
                ranks[(split, metric, row["layer"])] = rank
        for row in split_rows:
            row["real_margin_rank"] = ranks[(split, "real_semantic_margin", row["layer"])]
            row["grounded_margin_rank"] = ranks[(split, "grounded_semantic_margin", row["layer"])]
        aggregate.extend(split_rows)
    write_csv(ROOT / "LAYER_RESULTS.csv", aggregate)

    lookup = {(row["split"], row["layer"]): row for row in aggregate}
    correlations = []
    for metric in ("real_semantic_margin", "grounded_semantic_margin"):
        rho, p = spearmanr(
            [lookup[("exploration", layer)][metric] for layer in range(32)],
            [lookup[("validation", layer)][metric] for layer in range(32)],
        )
        correlations.append({"metric": metric, "spearman_rho": float(rho), "p_value": float(p)})
    write_csv(ROOT / "SPLIT_CORRELATIONS.csv", correlations)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    panels = (
        ("exploration", "real_semantic_margin", "Exploration: semantic-change margin"),
        ("validation", "real_semantic_margin", "Validation: semantic-change margin"),
        ("exploration", "grounded_semantic_margin", "Exploration: image-grounded margin"),
        ("validation", "grounded_semantic_margin", "Validation: image-grounded margin"),
    )
    for axis, (split, metric, title) in zip(axes.flat, panels):
        values = [lookup[(split, layer)][metric] for layer in range(32)]
        axis.bar(range(32), values, color=["#d62728" if layer == 11 else "#8da0cb" for layer in range(32)])
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(title); axis.grid(axis="y", alpha=.2)
        axis.annotate(
            f"L11 rank {ranks[(split, metric, 11)]}/32",
            xy=(11, values[11]), xytext=(13, max(values)),
            arrowprops={"arrowstyle": "->", "color": "#d62728"}, color="#d62728",
        )
    for axis in axes[1]:
        axis.set_xlabel("OpenVLA layer"); axis.set_xticks(range(0, 32, 2))
    fig.suptitle("Semantic sensitivity without image-region labels", fontsize=14)
    fig.tight_layout(); fig.savefig(ROOT / "figure_all_layers_semantic_sensitivity.png", dpi=200); plt.close(fig)

    lines = [
        "# Box-free semantic layer scan", "",
        "## Definition", "",
        "A semantic layer should remain stable for paraphrases and irrelevant wording, but change when the task target or relation changes. No token is mapped back to an image region.", "",
        f"Attention replay maximum absolute difference: `{max_reproduction:.8g}`.", "",
        "## Top layers", "",
        "| Split | Criterion | Top five | L11 |", "|---|---|---|---|",
    ]
    for split in ("exploration", "validation"):
        for metric, label in (("real_semantic_margin", "semantic margin"), ("grounded_semantic_margin", "grounded margin")):
            ordered = sorted(range(32), key=lambda layer: -lookup[(split, layer)][metric])
            top = ", ".join(f"L{layer} ({lookup[(split, layer)][metric]:+.4f})" for layer in ordered[:5])
            lines.append(f"| {split} | {label} | {top} | {lookup[(split, 11)][metric]:+.4f}, rank {ranks[(split, metric, 11)]}/32 |")
    lines.extend(["", "## Split stability", "", "| Metric | Spearman rho | p |", "|---|---:|---:|"])
    for row in correlations:
        lines.append(f"| {row['metric']} | {row['spearman_rho']:.3f} | {row['p_value']:.3g} |")
    lines.extend(["", "![All layers](figure_all_layers_semantic_sensitivity.png)", "", "## Boundary", "", "This scan selects layers by semantic sensitivity, not by action effect or closed-loop success. A small closed-loop comparison is still needed after shortlisting."])
    (ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"max_reproduction": max_reproduction, "correlations": correlations}, indent=2))


if __name__ == "__main__":
    main()
