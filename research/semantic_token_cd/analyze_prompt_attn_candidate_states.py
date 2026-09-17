"""Aggregate frozen-candidate same-state mask -> reconstruction -> action effects."""
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


ARMS = ("standard_shr", "prompt_v1", "prompt_single", "prompt_sparse")
KEYS = ("num_components", "isolated_token_ratio", "largest_component_ratio",
        "feature_perturbation_total", "selected_mean_relative_perturbation",
        "centered_residual_norm", "winner_flip_count")


def mean(rows, key):
    return float(np.mean([float(x[key]) for x in rows]))


def summarize(rows):
    output = {key: mean(rows, key) for key in KEYS}
    output["standard_overlap"] = float(np.mean([x["standard_overlap"] for x in rows]))
    output["v1_overlap"] = float(np.mean([x["v1_overlap"] for x in rows]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted((args.artifact / "candidate_state_eval").glob("*/*/step_*.json")):
        candidate = json.loads(path.read_text())
        task, seed, step = candidate["task"], candidate["seed"], candidate["source_step"]
        source_path = args.source / "states" / task / f"seed_{seed:03d}" / f"step_{step:03d}.json"
        source = json.loads(source_path.read_text())
        source_npz = np.load(source_path.with_suffix(".npz"))
        masks = {
            "standard_shr": source_npz["standard_shr__mask"],
            "prompt_v1": source_npz["prompt_attn_shr__mask"],
        }
        candidate_npz = np.load(path.with_suffix(".npz"))
        masks.update({arm: candidate_npz[f"{arm}__mask"] for arm in ("prompt_single", "prompt_sparse")})
        metrics = {
            "standard_shr": source["metrics"]["standard_shr"],
            "prompt_v1": source["metrics"]["prompt_attn_shr"],
            **candidate["metrics"],
        }
        for arm in ARMS:
            selected = set(np.flatnonzero(masks[arm]))
            standard = set(np.flatnonzero(masks["standard_shr"]))
            v1 = set(np.flatnonzero(masks["prompt_v1"]))
            row = {
                "state_id": candidate["state_id"], "task": task, "seed": seed,
                "source_step": step, "phase": candidate["phase"], "split": candidate["split"],
                "arm": arm, "m": candidate["m"],
                "standard_overlap": len(selected & standard) / candidate["m"],
                "v1_overlap": len(selected & v1) / candidate["m"],
            }
            row.update({key: metrics[arm][key] for key in KEYS})
            records.append(row)
    if len(records) != 360:
        raise RuntimeError(f"expected 360 state-arm rows, found {len(records)}")
    grouped = defaultdict(list)
    for row in records:
        grouped[("overall", "all", row["arm"])].append(row)
        grouped[("split", row["split"], row["arm"])].append(row)
        grouped[("task", row["task"], row["arm"])].append(row)
    results = {}
    for (kind, value, arm), rows in grouped.items():
        results.setdefault(kind, {}).setdefault(value, {})[arm] = summarize(rows)
    (args.artifact / "SAME_STATE_RESULTS.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    with (args.artifact / "same_state_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    overall = results["overall"]["all"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ["Standard", "Prompt-v1", "L11", "L11+L14"]
    for ax, key, title in zip(axes,
        ("num_components", "centered_residual_norm", "winner_flip_count"),
        ("Mask components", "Centered residual norm", "Winner flips / state")):
        ax.bar(labels, [overall[a][key] for a in ARMS]); ax.set_title(title); ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Frozen candidate same-state mechanism audit (90 states)")
    fig.tight_layout(); fig.savefig(args.artifact / "same_state_mechanism_summary.png", dpi=180); plt.close(fig)
    print(json.dumps(overall))


if __name__ == "__main__":
    main()
