#!/usr/bin/env python3
"""Create a direct L0-L31 comparison focused on the evidence for L11."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "artifacts/prompt_action_layer_semantic_consistency_v1/LAYER_RESULTS.csv"
ROOT = REPO / "artifacts/l11_all_layer_direct_comparison_v1"
METRICS = ("top_alignment", "semantic_separation", "mask_jaccard_separation")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["layer"] = int(row["layer"])
        for metric in METRICS:
            row[metric] = float(row[metric])

    ranks = {}
    for split in ("exploration", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        for metric in METRICS:
            for rank, row in enumerate(sorted(split_rows, key=lambda item: -item[metric]), 1):
                ranks[(split, metric, row["layer"])] = rank

    output = []
    lookup = {(row["split"], row["layer"]): row for row in rows}
    for layer in range(32):
        item = {"layer": layer}
        for split in ("exploration", "validation"):
            row = lookup[(split, layer)]
            prefix = "exp" if split == "exploration" else "val"
            for metric in METRICS:
                item[f"{prefix}_{metric}"] = row[metric]
                item[f"{prefix}_{metric}_rank"] = ranks[(split, metric, layer)]
        item["mean_action_alignment"] = np.mean([
            item["exp_top_alignment"], item["val_top_alignment"]
        ])
        item["mean_semantic_separation"] = np.mean([
            item["exp_semantic_separation"], item["val_semantic_separation"]
        ])
        output.append(item)
    with (ROOT / "ALL_LAYER_COMPARISON.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)

    layers = np.arange(32)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    panels = (
        ("exploration", "top_alignment", "Exploration: action relevance"),
        ("validation", "top_alignment", "Validation: action relevance"),
        ("exploration", "semantic_separation", "Exploration: target selectivity"),
        ("validation", "semantic_separation", "Validation: target selectivity"),
    )
    for axis, (split, metric, title) in zip(axes.flat, panels):
        values = np.asarray([lookup[(split, layer)][metric] for layer in layers])
        colors = ["#d62728" if layer == 11 else "#8da0cb" for layer in layers]
        axis.bar(layers, values, color=colors)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
        axis.annotate(
            f"L11={values[11]:.3f}\nrank {ranks[(split, metric, 11)]}/32",
            xy=(11, values[11]), xytext=(13, values.max()),
            arrowprops={"arrowstyle": "->", "color": "#d62728"}, color="#d62728",
        )
    for axis in axes[1]:
        axis.set_xlabel("OpenVLA layer")
        axis.set_xticks(np.arange(0, 32, 2))
    fig.suptitle("Direct offline comparison of all 32 layers (L11 highlighted)", fontsize=14)
    fig.tight_layout()
    fig.savefig(ROOT / "figure_all_32_layers.png", dpi=200)
    plt.close(fig)

    lines = [
        "# Direct L0-L31 comparison", "",
        "This report compares all 32 single layers with the same states and the same matched token count. Budget scaling is not part of this comparison.", "",
        "## L11 ranks", "",
        "| Metric | Exploration | Validation |", "|---|---:|---:|",
    ]
    for metric, label in (("top_alignment", "Action relevance"), ("semantic_separation", "Target selectivity"), ("mask_jaccard_separation", "Mask target selectivity")):
        lines.append(
            f"| {label} | {lookup[('exploration', 11)][metric]:.4f} (rank {ranks[('exploration', metric, 11)]}/32) | "
            f"{lookup[('validation', 11)][metric]:.4f} (rank {ranks[('validation', metric, 11)]}/32) |"
        )
    lines.extend(["", "## Honest result", "", "L11 is one of the strongest balanced layers: it is near the top for action relevance and at the top for sensitivity to a real target change. It is not the numerical winner of every metric, so this offline scan does not prove that L11 is the unique best layer. L9 and L14 remain the closest alternatives; only closed-loop evaluation can resolve success rate.", "", "![All layers](figure_all_32_layers.png)"])
    (ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
