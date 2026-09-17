#!/usr/bin/env python3
"""Analyze shortlisted layers across matched-budget scales."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_action_layer_budget_robustness_v2"
METRICS = (
    "top_alignment", "top_minus_random", "synonym_residual_cosine",
    "target_residual_cosine", "semantic_separation", "synonym_mask_jaccard",
    "target_mask_jaccard", "mask_jaccard_separation",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    files = sorted((ROOT / "results").glob("*.json"))
    if len(files) != 90:
        raise RuntimeError(f"expected 90 results, got {len(files)}")
    states = []
    max_clean_difference = 0.0
    for path in files:
        data = json.loads(path.read_text())
        max_clean_difference = max(
            max_clean_difference, float(data["cached_clean_action_max_abs_difference"])
        )
        for row in data["results"]:
            states.append({
                "state_id": data["state_id"],
                "task": data["task"].removeprefix("google_robot_"),
                "seed": int(data["seed"]),
                "split": data["split"],
                "matched_count": int(data["count"]),
                **{key: row[key] for key in ("layer", "scale", "count", *METRICS)},
            })
    write_csv(ROOT / "STATEWISE_RESULTS.csv", states)

    groups = defaultdict(list)
    for row in states:
        groups[(row["split"], row["task"], row["seed"], row["layer"], row["scale"])].append(row)
    episodes = []
    for (split, task, seed, layer, scale), items in sorted(groups.items()):
        episodes.append({
            "split": split, "task": task, "seed": seed, "layer": layer, "scale": scale,
            "mean_count": float(np.mean([row["count"] for row in items])),
            **{metric: float(np.mean([row[metric] for row in items])) for metric in METRICS},
        })
    write_csv(ROOT / "EPISODE_RESULTS.csv", episodes)

    aggregates = []
    for split in ("exploration", "validation"):
        for layer in (9, 11, 14):
            for scale in (0.75, 1.0, 1.25):
                items = [
                    row for row in episodes
                    if row["split"] == split and row["layer"] == layer and row["scale"] == scale
                ]
                aggregates.append({
                    "split": split, "layer": layer, "scale": scale, "episodes": len(items),
                    "mean_count": float(np.mean([row["mean_count"] for row in items])),
                    **{metric: float(np.mean([row[metric] for row in items])) for metric in METRICS},
                })
    write_csv(ROOT / "AGGREGATE_RESULTS.csv", aggregates)

    deltas = []
    lookup = {(row["split"], row["layer"], row["scale"]): row for row in aggregates}
    for split in ("exploration", "validation"):
        for layer in (9, 11, 14):
            baseline = lookup[(split, layer, 1.0)]
            for scale in (0.75, 1.25):
                row = lookup[(split, layer, scale)]
                deltas.append({
                    "split": split, "layer": layer, "scale": scale,
                    **{f"delta_{metric}": row[metric] - baseline[metric] for metric in METRICS},
                })
    write_csv(ROOT / "DELTA_VS_MATCHED.csv", deltas)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    displayed = (
        ("top_alignment", "Action alignment"),
        ("top_minus_random", "Alignment minus random"),
        ("semantic_separation", "Target semantic separation"),
        ("mask_jaccard_separation", "Mask semantic separation"),
    )
    styles = {"exploration": "-", "validation": "--"}
    for axis, (metric, title) in zip(axes.flat, displayed):
        for split in ("exploration", "validation"):
            for layer in (9, 11, 14):
                values = [lookup[(split, layer, scale)][metric] for scale in (0.75, 1.0, 1.25)]
                axis.plot(
                    (0.75, 1.0, 1.25), values, marker="o", linestyle=styles[split],
                    label=f"L{layer} {split}" if metric == "top_alignment" else None,
                )
        axis.axvline(1.0, color="black", alpha=.2)
        axis.set_title(title)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    for axis in axes[1]:
        axis.set_xlabel("Matched budget scale")
    fig.tight_layout()
    fig.savefig(ROOT / "figure_budget_robustness.png", dpi=200)
    plt.close(fig)

    lines = [
        "# Prompt-action layer budget robustness", "",
        "## Protocol", "",
        "L9, L11, and L14 are evaluated on the same 90 cached states using budget scales 0.75, 1.0, and 1.25. No image boxes or closed-loop rollouts are used.", "",
        f"Technical replay audit: maximum clean-action-logit difference = `{max_clean_difference:.8g}`.", "",
        "## Aggregate results", "",
        "| Split | Layer | Scale | Mean tokens | Action alignment | Minus random | Semantic separation | Mask separation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['split']} | L{row['layer']} | {row['scale']:.2f} | {row['mean_count']:.2f} | "
            f"{row['top_alignment']:.4f} | {row['top_minus_random']:.4f} | "
            f"{row['semantic_separation']:.4f} | {row['mask_jaccard_separation']:.4f} |"
        )
    lines.extend(["", "![Budget robustness](figure_budget_robustness.png)", "", "## Boundary", "", "These are offline action-level proxies. They test whether the layer shortlist depends on the exact matched budget, but they cannot determine closed-loop success."])
    (ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"max_clean_difference": max_clean_difference, "aggregate": aggregates}, indent=2))


if __name__ == "__main__":
    main()
