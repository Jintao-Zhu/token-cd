"""Analyze the standard Fractal Pi0 Google Robot prompt-attention scan."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
OUT = ROOT / "token-cd/artifacts/pi0_fractal_google_layer_selection_v1"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
SEEDS = (100, 110, 120, 130, 150, 160)
N_LAYERS = 18
K = 32


def topk(scores: np.ndarray) -> list[int]:
    return np.argsort(-scores, kind="stable")[:K].tolist()


def jaccard(a, b) -> float:
    aa, bb = set(a), set(b)
    return len(aa & bb) / max(1, len(aa | bb))


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def percentile(values, value) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float((np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values))


def load_states():
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            directory = OUT / "states" / task / f"seed_{seed:03d}"
            meta = json.loads((directory / "meta.json").read_text())
            arrays = dict(np.load(directory / "layers.npz"))
            rows.append((meta, arrays))
    return rows


def evaluate(states, layer: int) -> dict:
    by_task: dict[str, list[dict]] = {task: [] for task in TASKS}
    for meta, arrays in states:
        original = arrays["original"][layer].astype(np.float64)
        synonym = arrays["synonym"][layer].astype(np.float64)
        switched = arrays["target_switch"][layer].astype(np.float64)
        original /= max(1e-12, float(original.sum()))
        synonym /= max(1e-12, float(synonym.sum()))
        switched /= max(1e-12, float(switched.sum()))
        old = arrays["old_target_mask"].astype(bool)
        new = arrays["new_target_mask"].astype(bool)
        new_gain = float(switched[new].sum() - original[new].sum())
        old_drop = float(original[old].sum() - switched[old].sum())
        old_top, synonym_top, switched_top = topk(original), topk(synonym), topk(switched)
        old_set, switched_set = set(old_top), set(switched_top)
        mask_gain = (len(switched_set & set(np.flatnonzero(new))) - len(old_set & set(np.flatnonzero(new)))) / K
        mask_drop = (len(old_set & set(np.flatnonzero(old))) - len(switched_set & set(np.flatnonzero(old)))) / K
        by_task[meta["task"]].append({
            "task": meta["task"],
            "seed": int(meta["seed"]),
            "target_response": 0.5 * (new_gain + old_drop),
            "mask_response": 0.5 * (mask_gain + mask_drop),
            "correct_both": bool(new_gain > 0 and old_drop > 0),
            "synonym_top32_jaccard": jaccard(old_top, synonym_top),
            "synonym_attention_cosine": cosine(original, synonym),
        })
    all_rows = [x for group in by_task.values() for x in group]
    return {
        "layer_index_0based": layer,
        "layer_label": f"L{layer}",
        "target_response": float(np.mean([x["target_response"] for x in all_rows])),
        "mask_response": float(np.mean([x["mask_response"] for x in all_rows])),
        "correct_response_fraction": float(np.mean([x["correct_both"] for x in all_rows])),
        "positive_task_count": int(sum(np.mean([x["target_response"] for x in rows]) > 0 for rows in by_task.values())),
        "synonym_top32_jaccard": float(np.mean([x["synonym_top32_jaccard"] for x in all_rows])),
        "synonym_attention_cosine": float(np.mean([x["synonym_attention_cosine"] for x in all_rows])),
        "by_task": {task: {
            "target_response": float(np.mean([x["target_response"] for x in rows])),
            "mask_response": float(np.mean([x["mask_response"] for x in rows])),
            "correct_response_fraction": float(np.mean([x["correct_both"] for x in rows])),
            "synonym_top32_jaccard": float(np.mean([x["synonym_top32_jaccard"] for x in rows])),
        } for task, rows in by_task.items()},
        "state_rows": all_rows,
    }


def main() -> None:
    states = load_states()
    singles = [evaluate(states, layer) for layer in range(N_LAYERS)]
    target = [r["target_response"] for r in singles]
    mask = [r["mask_response"] for r in singles]
    correct = [r["correct_response_fraction"] for r in singles]
    synonym = [r["synonym_top32_jaccard"] for r in singles]
    for row in singles:
        row["selection_score"] = (
            0.55 * percentile(target, row["target_response"])
            + 0.20 * percentile(mask, row["mask_response"])
            + 0.15 * percentile(correct, row["correct_response_fraction"])
            + 0.10 * percentile(synonym, row["synonym_top32_jaccard"])
        )
    eligible = [r for r in singles if r["positive_task_count"] >= 2]
    ranked = sorted(eligible or singles, key=lambda r: (-r["selection_score"], r["layer_index_0based"]))
    all_ranked = sorted(singles, key=lambda r: (-r["selection_score"], r["layer_index_0based"]))
    for rank, row in enumerate(all_ranked, start=1):
        row["all_layer_rank"] = rank
        row["eligible_positive_on_two_or_more_tasks"] = row in eligible
    selected = ranked[0]

    # Small-sample robustness diagnostics: leave the same seed out across all
    # four tasks (so each fold tests four held-out states), and task-stratified
    # resampling of the six fixed seeds. These diagnose selection stability;
    # they are not independent confirmatory validation.
    # Reuse state-level metrics already computed above, avoiding repeated
    # attention-array top-k work inside the bootstrap.
    state_metrics = {}
    for layer, summary in enumerate(singles):
        state_metrics[layer] = {
            (row["task"], int(row["seed"])): np.asarray([
                row["target_response"], row["mask_response"],
                float(row["correct_both"]), row["synonym_top32_jaccard"],
            ], dtype=np.float64)
            for row in summary["state_rows"]
        }

    def select_metric_keys(keys):
        layer_means = []
        for layer in range(N_LAYERS):
            vals = np.stack([state_metrics[layer][key] for key in keys])
            positive_tasks = sum(
                np.mean([state_metrics[layer][(task, seed)][0] for task2, seed in keys if task2 == task]) > 0
                for task in TASKS
            )
            layer_means.append([*vals.mean(axis=0), positive_tasks])
        means = np.asarray(layer_means)
        components = [means[:, i] for i in range(4)]
        score = (
            0.55 * np.asarray([percentile(components[0], means[i, 0]) for i in range(N_LAYERS)])
            + 0.20 * np.asarray([percentile(components[1], means[i, 1]) for i in range(N_LAYERS)])
            + 0.15 * np.asarray([percentile(components[2], means[i, 2]) for i in range(N_LAYERS)])
            + 0.10 * np.asarray([percentile(components[3], means[i, 3]) for i in range(N_LAYERS)])
        )
        eligible_layers = [i for i in range(N_LAYERS) if means[i, 4] >= 2]
        choices = eligible_layers or list(range(N_LAYERS))
        chosen = sorted(choices, key=lambda i: (-score[i], i))[0]
        return chosen, score, means

    leave_seed_out = []
    for held_seed in SEEDS:
        train = [(meta["task"], int(meta["seed"])) for meta, _ in states if int(meta["seed"]) != held_seed]
        held = [(meta["task"], int(meta["seed"])) for meta, _ in states if int(meta["seed"]) == held_seed]
        chosen, scores, _ = select_metric_keys(train)
        leave_seed_out.append({
            "held_seed": held_seed,
            "selected_layer": chosen,
            "selected_layer_label": f"L{chosen}",
            "held_target_response": float(np.mean([state_metrics[chosen][key][0] for key in held])),
            "held_mask_response": float(np.mean([state_metrics[chosen][key][1] for key in held])),
        })
    rng = np.random.default_rng(20261001)
    task_states = {task: [item for item in states if item[0]["task"] == task] for task in TASKS}
    bootstrap_counts = np.zeros(N_LAYERS, dtype=np.int64)
    score_gaps = []
    for _ in range(2000):
        sample = []
        for task in TASKS:
            candidates = task_states[task]
            sample.extend((task, int(candidates[i][0]["seed"])) for i in rng.integers(0, len(candidates), len(candidates)))
        chosen, scores, _ = select_metric_keys(sample)
        bootstrap_counts[chosen] += 1
        score_gaps.append(float(scores[10] - scores[8]))
    robustness = {
        "leave_one_seed_out": leave_seed_out,
        "bootstrap_replicates": 2000,
        "bootstrap_resampling": "resample 6 seeds with replacement independently within each task",
        "selected_layer_counts": {f"L{i}": int(n) for i, n in enumerate(bootstrap_counts) if n},
        "selection_score_gap_L10_minus_L8_quantiles": np.quantile(score_gaps, [0.025, 0.5, 0.975]).tolist(),
        "warning": "These are small-sample stability diagnostics on the same 24-state pool, not held-out behavioral validation.",
    }
    result = {
        "protocol_id": "PI0_FRACTAL_GOOGLE_PROMPT_ATTENTION_LAYER_SCAN_V1",
        "checkpoint": "/home/leju-suzhou/zjt_ws/checkpoints/pi0/fractal_beta_step29576_2024-12-29_13-10_42.pt",
        "model": "open-pi-zero PyTorch PiZeroInference, Fractal-Beta fine-tune; PaliGemma language trunk",
        "candidate_layer_space": "0-based 18 language layers L0-L17; action expert excluded",
        "states": len(states), "tasks": list(TASKS), "seeds": list(SEEDS), "rank_budget": K,
        "selection_rule": "same diagnostic as previous SpatialVLA/Pi0 layer scans: 0.55 target-attention response percentile + 0.20 Top-32 target-mask response percentile + 0.15 correct-both response fraction percentile + 0.10 synonym Top-32 Jaccard percentile; eligible if positive mean target response on >=2/4 tasks",
        "selected_single_layer_index_0based": selected["layer_index_0based"],
        "selected_single_layer_label": selected["layer_label"],
        "selected_single_layer_metrics": {k: v for k, v in selected.items() if k != "state_rows"},
        "ranked_layers": [{k: v for k, v in r.items() if k != "state_rows"} for r in all_ranked],
        "selection_stability_diagnostics": robustness,
        "data_usage": "24 fixed states used for exploratory candidate selection; no independent held-out layer audit or closed-loop comparison. Input images are exact saved RGB; proprio was fixed at normalized zero raw state.",
        "interpretation": "The selected layer is the top offline candidate for Fractal/Google under this criterion. Do not call it closed-loop-optimal until validated against neighboring layers on paired held-out episodes.",
    }
    (OUT / "LAYER_SCAN_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "SELECTION_STABILITY.json").write_text(json.dumps(robustness, indent=2, sort_keys=True) + "\n")
    fields = ["layer_index_0based", "layer_label", "all_layer_rank", "eligible_positive_on_two_or_more_tasks", "selection_score", "target_response", "mask_response", "correct_response_fraction", "positive_task_count", "synonym_top32_jaccard", "synonym_attention_cosine"]
    with (OUT / "per_layer.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in all_ranked)
    x = np.arange(N_LAYERS)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, [r["target_response"] for r in singles], marker="o", label="target-attention response")
    ax.plot(x, [r["mask_response"] for r in singles], marker="s", label="Top-32 target-mask response")
    ax.plot(x, [r["synonym_top32_jaccard"] for r in singles], marker="^", label="synonym Top-32 Jaccard")
    ax.axvline(selected["layer_index_0based"], color="tab:red", linestyle="--", label=f"selected {selected['layer_label']}")
    ax.set_xlabel("0-based PaliGemma language layer")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "layer_scan_curve.png", dpi=180); plt.close(fig)
    print(json.dumps({"selected_layer": selected["layer_label"], "metrics": result["selected_single_layer_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
