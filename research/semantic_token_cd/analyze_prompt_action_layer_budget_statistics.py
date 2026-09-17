#!/usr/bin/env python3
"""Paired statistical and taskwise audit for layer budget robustness."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_action_layer_budget_robustness_v2"
METRICS = ("top_alignment", "top_minus_random", "semantic_separation", "mask_jaccard_separation")


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    means = values[generator.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main() -> None:
    episodes = read_csv(ROOT / "EPISODE_RESULTS.csv")
    for row in episodes:
        row["seed"] = int(row["seed"])
        row["layer"] = int(row["layer"])
        row["scale"] = float(row["scale"])
        for metric in METRICS:
            row[metric] = float(row[metric])

    task_groups = defaultdict(list)
    for row in episodes:
        task_groups[(row["split"], row["task"], row["layer"], row["scale"])].append(row)
    taskwise = []
    for (split, task, layer, scale), items in sorted(task_groups.items()):
        taskwise.append({
            "split": split, "task": task, "layer": layer, "scale": scale,
            "episodes": len(items),
            **{metric: float(np.mean([row[metric] for row in items])) for metric in METRICS},
        })
    write_csv(ROOT / "TASKWISE_RESULTS.csv", taskwise)

    lookup = {
        (row["split"], row["task"], row["seed"], row["layer"], row["scale"]): row
        for row in episodes
    }
    paired = []
    comparison_index = 0
    for split in ("exploration", "validation"):
        keys = sorted({(row["task"], row["seed"]) for row in episodes if row["split"] == split})
        for layer in (9, 11, 14):
            for scale in (0.75, 1.25):
                for metric in METRICS:
                    differences = np.asarray([
                        lookup[(split, task, seed, layer, scale)][metric]
                        - lookup[(split, task, seed, layer, 1.0)][metric]
                        for task, seed in keys
                    ])
                    mean, low, high = bootstrap_ci(differences, 20260915 + comparison_index)
                    comparison_index += 1
                    paired.append({
                        "split": split, "comparison": f"L{layer} scale {scale:.2f} - 1.00",
                        "metric": metric, "episodes": len(differences),
                        "mean_difference": mean, "ci_low": low, "ci_high": high,
                        "positive_fraction": float(np.mean(differences > 0)),
                    })
        for scale in (0.75, 1.0, 1.25):
            for other in (9, 14):
                for metric in METRICS:
                    differences = np.asarray([
                        lookup[(split, task, seed, 11, scale)][metric]
                        - lookup[(split, task, seed, other, scale)][metric]
                        for task, seed in keys
                    ])
                    mean, low, high = bootstrap_ci(differences, 20260915 + comparison_index)
                    comparison_index += 1
                    paired.append({
                        "split": split, "comparison": f"L11 - L{other} at scale {scale:.2f}",
                        "metric": metric, "episodes": len(differences),
                        "mean_difference": mean, "ci_low": low, "ci_high": high,
                        "positive_fraction": float(np.mean(differences > 0)),
                    })
    write_csv(ROOT / "PAIRED_BOOTSTRAP.csv", paired)

    def pair(split: str, comparison: str, metric: str) -> dict:
        return next(row for row in paired if row["split"] == split and row["comparison"] == comparison and row["metric"] == metric)

    lines = [
        "# Statistical audit", "",
        "## Direct answer", "",
        "The exact matched scale `1.0` is not uniquely optimal for the offline proxies. Reducing the budget to `0.75×` consistently strengthens L11's target selectivity, while action alignment remains similar or slightly better. However, the 10-episode validation split is too small for most paired confidence intervals to exclude zero.", "",
        "## L11 budget differences", "",
        "Values are paired episode means relative to scale 1.0.", "",
        "| Split | Scale | Metric | Difference | 95% bootstrap CI |", "|---|---:|---|---:|---:|",
    ]
    for split in ("exploration", "validation"):
        for scale in (0.75, 1.25):
            comparison = f"L11 scale {scale:.2f} - 1.00"
            for metric in ("top_alignment", "semantic_separation", "mask_jaccard_separation"):
                row = pair(split, comparison, metric)
                lines.append(
                    f"| {split} | {scale:.2f} | {metric} | {row['mean_difference']:+.4f} | "
                    f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
                )
    lines.extend(["", "## Layer comparison at 0.75×", "", "| Split | Comparison | Metric | Difference | 95% bootstrap CI |", "|---|---|---|---:|---:|"])
    for split in ("exploration", "validation"):
        for other in (9, 14):
            comparison = f"L11 - L{other} at scale 0.75"
            for metric in ("top_alignment", "semantic_separation"):
                row = pair(split, comparison, metric)
                lines.append(
                    f"| {split} | L11 - L{other} | {metric} | {row['mean_difference']:+.4f} | "
                    f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
                )
    lines.extend([
        "", "## Interpretation", "",
        "- L11 is the only shortlisted layer for which `0.75×` improves action alignment, target separation, and mask separation in both splits.",
        "- L14 at `0.75×` gains semantic separation but does not improve action alignment consistently.",
        "- L9 is less stable: its exploration action alignment drops strongly at `0.75×`.",
        "- L11 at `1.25×` raises validation action alignment but loses semantic and mask selectivity, suggesting that larger masks increasingly include prompt-insensitive context.",
        "- These results strengthen L11 as the most budget-robust offline candidate, but still do not prove superior closed-loop success.",
    ])
    (ROOT / "STATISTICAL_AUDIT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
