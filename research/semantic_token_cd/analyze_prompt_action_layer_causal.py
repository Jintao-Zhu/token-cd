#!/usr/bin/env python3
"""Aggregate the box-free prompt-to-action layer causal screen."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(20000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    files = sorted((root / "results").glob("*.json"))
    config = json.loads((root / "CONFIG_LOCK.json").read_text())
    expected_states = int(config["expected_states"])
    if len(files) != expected_states:
        raise RuntimeError(f"expected {expected_states} completed states, got {len(files)}")

    state_rows = []
    for path in files:
        data = json.loads(path.read_text())
        random_alignment = float(data["random"]["alignment"])
        for layer in data["layers"]:
            branches = layer["branches"]
            state_rows.append({
                "state_id": data["state_id"],
                "task": data["task"].removeprefix("google_robot_"),
                "seed": data["seed"],
                "split": data["split"],
                "layer": layer["layer"],
                "top_alignment": branches["top"]["alignment"],
                "bottom_alignment": branches["bottom"]["alignment"],
                "wrong_prompt_alignment": branches["wrong_prompt"]["alignment"],
                "random_alignment": random_alignment,
                "top_strength_ratio": branches["top"]["strength_ratio"],
                "top_minus_random": layer["top_minus_random"],
                "top_minus_bottom": layer["top_minus_bottom"],
                "top_minus_wrong_prompt": layer["top_minus_wrong_prompt"],
                "causal_margin": layer["causal_margin"],
            })
    write_csv(root / "STATEWISE_RESULTS.csv", state_rows)

    episode_groups = defaultdict(list)
    for row in state_rows:
        episode_groups[(row["split"], row["task"], int(row["seed"]), int(row["layer"]))].append(row)
    episode_rows = []
    metric_fields = (
        "top_alignment", "bottom_alignment", "wrong_prompt_alignment", "random_alignment",
        "top_strength_ratio", "top_minus_random", "top_minus_bottom",
        "top_minus_wrong_prompt", "causal_margin",
    )
    for (split, task, seed, layer), rows in sorted(episode_groups.items()):
        episode_rows.append({
            "split": split, "task": task, "seed": seed, "layer": layer, "states": len(rows),
            **{field: mean(rows, field) for field in metric_fields},
        })
    write_csv(root / "EPISODE_RESULTS.csv", episode_rows)

    layer_rows = []
    for split in ("exploration", "validation", "all"):
        subset = episode_rows if split == "all" else [row for row in episode_rows if row["split"] == split]
        groups = defaultdict(list)
        for row in subset:
            groups[int(row["layer"])].append(row)
        for layer, rows in sorted(groups.items()):
            low, high = bootstrap_ci([float(row["causal_margin"]) for row in rows], 20260915 + layer)
            layer_rows.append({
                "split": split,
                "layer": layer,
                "states": len(rows),
                "top_alignment": mean(rows, "top_alignment"),
                "bottom_alignment": mean(rows, "bottom_alignment"),
                "wrong_prompt_alignment": mean(rows, "wrong_prompt_alignment"),
                "random_alignment": mean(rows, "random_alignment"),
                "top_strength_ratio": mean(rows, "top_strength_ratio"),
                "top_minus_random": mean(rows, "top_minus_random"),
                "top_minus_bottom": mean(rows, "top_minus_bottom"),
                "top_minus_wrong_prompt": mean(rows, "top_minus_wrong_prompt"),
                "causal_margin": mean(rows, "causal_margin"),
                "causal_margin_ci95_low": low,
                "causal_margin_ci95_high": high,
                "top_wins_all_controls": int(all(row["causal_margin"] > 0 for row in rows)),
            })
    write_csv(root / "LAYER_RESULTS.csv", layer_rows)

    exploration = [row for row in layer_rows if row["split"] == "exploration"]
    validation = {int(row["layer"]): row for row in layer_rows if row["split"] == "validation"}
    ranked = sorted(exploration, key=lambda row: (-row["causal_margin"], -row["top_alignment"], row["layer"]))
    top = ranked[:8]

    layers = np.arange(32)
    exp_by_layer = {int(row["layer"]): row for row in exploration}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(layers, [exp_by_layer[i]["top_alignment"] for i in layers], marker="o", label="Top-m")
    axes[0].plot(layers, [exp_by_layer[i]["bottom_alignment"] for i in layers], alpha=.75, label="Bottom-m")
    axes[0].plot(layers, [exp_by_layer[i]["wrong_prompt_alignment"] for i in layers], alpha=.75, label="Wrong-prompt Top-m")
    axes[0].plot(layers, [exp_by_layer[i]["random_alignment"] for i in layers], linestyle="--", label="Random-m")
    axes[0].set_ylabel("Prompt-to-action alignment")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=.2)
    axes[1].bar(layers, [exp_by_layer[i]["causal_margin"] for i in layers], color=["#c93434" if i == 11 else "#315b7d" for i in layers])
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_ylabel("Top-m minus strongest control")
    axes[1].set_xlabel("Layer index")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Box-free action-level layer screen — exploration split")
    fig.tight_layout()
    fig.savefig(root / "figure_action_alignment_layers.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [f"L{row['layer']}" for row in top]
    x = np.arange(len(top))
    ax.bar(x - .18, [row["causal_margin"] for row in top], width=.36, label="Exploration margin")
    ax.bar(x + .18, [validation[int(row["layer"])]["causal_margin"] for row in top], width=.36, label="Frozen validation margin")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("Top-m minus strongest control")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "figure_exploration_validation.png", dpi=200)
    plt.close(fig)

    summary = [
        "# Prompt-to-action layer causal screen",
        "",
        "This audit uses no target boxes. It ranks layers by whether reconstructing their Prompt-Top-m tokens produces an action-logit residual aligned with the residual caused by changing the Prompt target.",
        "",
        "## Protocol",
        "",
        f"- {expected_states} same-image target-switch states, aggregated into episodes before layer ranking.",
        "- Primary metric: mean per-action-dimension cosine between intervention residual and target-Prompt residual.",
        "- Causal margin: Top-m alignment minus the strongest of Random-m, Bottom-m, and wrong-Prompt Top-m.",
        "- Manual target boxes are not read or used.",
        "",
        "## Exploration top 8",
        "",
        "| Rank | Layer | Top alignment | Causal margin | 95% CI | Validation alignment | Validation margin |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top, 1):
        valid = validation[int(row["layer"])]
        summary.append(
            f"| {rank} | L{row['layer']} | {row['top_alignment']:.4f} | {row['causal_margin']:+.4f} | "
            f"[{row['causal_margin_ci95_low']:+.4f}, {row['causal_margin_ci95_high']:+.4f}] | "
            f"{valid['top_alignment']:.4f} | {valid['causal_margin']:+.4f} |"
        )
    l11_exp = exp_by_layer[11]
    l11_val = validation[11]
    summary.extend([
        "",
        "## L11",
        "",
        f"- Exploration rank: {next(index for index, row in enumerate(ranked, 1) if int(row['layer']) == 11)}/32.",
        f"- Exploration Top alignment / causal margin: {l11_exp['top_alignment']:.4f} / {l11_exp['causal_margin']:+.4f}.",
        f"- Frozen validation Top alignment / causal margin: {l11_val['top_alignment']:.4f} / {l11_val['causal_margin']:+.4f}.",
        "",
        "## Figures",
        "",
        "![Layer alignment](figure_action_alignment_layers.png)",
        "",
        "![Exploration validation](figure_exploration_validation.png)",
        "",
        "## Boundary",
        "",
        "- This is an action-level offline causal proxy, not closed-loop success.",
        "- The target-switch sample is small; only frozen validation consistency should justify a closed-loop shortlist.",
    ])
    (root / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    print(json.dumps({
        "states": len(files),
        "exploration_best_layer": int(ranked[0]["layer"]),
        "exploration_best_margin": ranked[0]["causal_margin"],
        "l11_exploration_rank": next(index for index, row in enumerate(ranked, 1) if int(row["layer"]) == 11),
        "l11_validation_margin": l11_val["causal_margin"],
    }, indent=2))


if __name__ == "__main__":
    main()
