"""Score the completed pi0-FAST all-layer offline attention extraction."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
OUT = ROOT / "token-cd/artifacts/pi0_fast_layer_selection_v1"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
SEEDS = (100, 110, 120, 130, 150, 160)
K = 32


def topk(x: np.ndarray, k: int = K) -> list[int]:
    return np.argsort(-x, kind="stable")[:k].tolist()


def jaccard(a, b) -> float:
    aa, bb = set(a), set(b)
    return len(aa & bb) / max(1, len(aa | bb))


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def percentile(values, x) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float((np.sum(v < x) + 0.5 * np.sum(v == x)) / len(v))


def load_states():
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            d = OUT / "states" / task / f"seed_{seed:03d}"
            meta_path, array_path = d / "meta.json", d / "layers.npz"
            if not meta_path.exists() or not array_path.exists():
                raise FileNotFoundError(f"missing scan output: {d}")
            rows.append((json.loads(meta_path.read_text()), dict(np.load(array_path))))
    return rows


def evaluate(states, layers):
    by_task: dict[str, list[dict]] = {}
    for meta, arrays in states:
        original = arrays["original"][list(layers)].mean(axis=0)
        synonym = arrays["synonym"][list(layers)].mean(axis=0)
        switched = arrays["target_switch"][list(layers)].mean(axis=0)
        # Compare spatial distributions on the queried primary view, so changes
        # in total cross-view attention mass do not masquerade as localization.
        original = original / max(1e-12, float(original.sum()))
        synonym = synonym / max(1e-12, float(synonym.sum()))
        switched = switched / max(1e-12, float(switched.sum()))
        old = arrays["old_target_mask"].astype(bool)
        new = arrays["new_target_mask"].astype(bool)
        new_gain = float(switched[new].sum() - original[new].sum())
        old_drop = float(original[old].sum() - switched[old].sum())
        omask, smask, tmask = topk(original), topk(synonym), topk(switched)
        old_set, new_set = set(omask), set(tmask)
        mask_gain = (len(new_set & set(new.nonzero()[0])) - len(old_set & set(new.nonzero()[0]))) / K
        mask_drop = (len(old_set & set(old.nonzero()[0])) - len(new_set & set(old.nonzero()[0]))) / K
        by_task.setdefault(meta["task"], []).append({
            "task": meta["task"], "seed": int(meta["seed"]),
            "target_response": 0.5 * (new_gain + old_drop),
            "mask_response": 0.5 * (mask_gain + mask_drop),
            "correct_both": bool(new_gain > 0 and old_drop > 0),
            "new_target_gain": new_gain, "old_target_drop": old_drop,
            "synonym_mask_jaccard": jaccard(omask, smask),
            "synonym_score_cosine": cosine(original, synonym),
        })
    all_rows = [x for values in by_task.values() for x in values]
    return {
        "layers": list(layers),
        "target_response": float(np.mean([x["target_response"] for x in all_rows])),
        "mask_response": float(np.mean([x["mask_response"] for x in all_rows])),
        "correct_response_fraction": float(np.mean([x["correct_both"] for x in all_rows])),
        "positive_task_count": int(sum(np.mean([x["target_response"] for x in v]) > 0 for v in by_task.values())),
        "synonym_mask_jaccard": float(np.mean([x["synonym_mask_jaccard"] for x in all_rows])),
        "synonym_score_cosine": float(np.mean([x["synonym_score_cosine"] for x in all_rows])),
        "by_task": {
            task: {
                "n_states": len(rows),
                "target_response": float(np.mean([x["target_response"] for x in rows])),
                "mask_response": float(np.mean([x["mask_response"] for x in rows])),
                "correct_response_fraction": float(np.mean([x["correct_both"] for x in rows])),
                "synonym_mask_jaccard": float(np.mean([x["synonym_mask_jaccard"] for x in rows])),
            }
            for task, rows in by_task.items()
        },
        "state_rows": all_rows,
    }


def main():
    states = load_states()
    n_layers = int(states[0][1]["original"].shape[0])
    singles = [evaluate(states, (i,)) for i in range(n_layers)]
    target_p = [r["target_response"] for r in singles]
    mask_p = [r["mask_response"] for r in singles]
    correct_p = [r["correct_response_fraction"] for r in singles]
    synonym_p = [r["synonym_mask_jaccard"] for r in singles]
    for row in singles:
        row["selection_score"] = (
            0.55 * percentile(target_p, row["target_response"])
            + 0.20 * percentile(mask_p, row["mask_response"])
            + 0.15 * percentile(correct_p, row["correct_response_fraction"])
            + 0.10 * percentile(synonym_p, row["synonym_mask_jaccard"])
        )
    eligible = [r for r in singles if r["positive_task_count"] >= 2]
    ranked_eligible = sorted(eligible or singles, key=lambda r: (-r["selection_score"], r["layers"][0]))
    ranked_all = sorted(singles, key=lambda r: (-r["selection_score"], r["layers"][0]))
    eligible_indices = {r["layers"][0] for r in eligible}
    for rank, row in enumerate(ranked_all, start=1):
        row["all_layer_rank"] = rank
        row["eligible_positive_on_two_or_more_tasks"] = row["layers"][0] in eligible_indices
    candidates = [r["layers"][0] for r in ranked_eligible[:4]]
    pairs = [evaluate(states, pair) for pair in itertools.combinations(candidates, 2)]
    for row in pairs:
        row["selection_score"] = (
            0.55 * percentile(target_p, row["target_response"])
            + 0.20 * percentile(mask_p, row["mask_response"])
            + 0.15 * percentile(correct_p, row["correct_response_fraction"])
            + 0.10 * percentile(synonym_p, row["synonym_mask_jaccard"])
        )
    selected = ranked_eligible[0]
    sparse = max(pairs, key=lambda r: (r["selection_score"], -sum(r["layers"]))) if pairs else None

    # Keep outputs compact; raw per-state values remain available in state_rows.
    summary = {
        "protocol": "PI0_FAST_LAYER_SELECTION_OFFLINE_V1",
        "checkpoint": str(ROOT / "checkpoints/openpi-assets/openpi-assets/checkpoints/pi0_fast_base"),
        "n_language_layers": n_layers,
        "states": len(states),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "rank_budget": K,
        "selection_rule": "same offline diagnostic weights as SpatialVLA scan: 0.55 target-attention response percentile + 0.20 top-32 target-mask response percentile + 0.15 correct-both response fraction percentile + 0.10 synonym Top-32 Jaccard percentile; candidate eligibility requires positive task-mean target response on >=2 tasks",
        "selected_single_layer_index_0based": selected["layers"][0],
        "selected_single_layer_label_same_as_existing_repo": f"L{selected['layers'][0]}",
        "selected_single_metrics": selected,
        "ranked_singles": [{k: v for k, v in row.items() if k != "state_rows"} for row in ranked_all],
        "top_single_pairs": [{k: v for k, v in row.items() if k != "state_rows"} for row in sorted(pairs, key=lambda r: -r["selection_score"])],
        "selected_sparse_pair": sparse,
        "caveat": "Offline attention-layer candidate screen only, not closed-loop validation. Uses six pre-existing fixed RGB states per Google Robot task. pi0-FAST FAST-prefix state text used a fixed zero state (same within each three-prompt comparison), transformed with local Fractal stats; task phrase query positions only. The scan therefore nominates a layer; it does not establish that it improves SIMPLER success.",
    }
    (OUT / "LAYER_SCAN_RESULTS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (OUT / "per_layer.csv").open("w", newline="") as f:
        keys = ["layer_index_0based", "all_layer_rank", "eligible_positive_on_two_or_more_tasks", "selection_score", "target_response", "mask_response", "correct_response_fraction", "positive_task_count", "synonym_mask_jaccard", "synonym_score_cosine"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in ranked_all:
            writer.writerow({**{key: row.get(key) for key in keys if key != "layer_index_0based"}, "layer_index_0based": row["layers"][0]})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_layers)
    ax.plot(x, [r["target_response"] for r in singles], marker="o", label="target attention response")
    ax.plot(x, [r["mask_response"] for r in singles], marker="s", label="Top-32 target-mask response")
    ax.plot(x, [r["synonym_mask_jaccard"] for r in singles], marker="^", label="synonym Top-32 Jaccard")
    ax.axvline(selected["layers"][0], color="tab:red", linestyle="--", label=f"selected L{selected['layers'][0]}")
    ax.set_xlabel("0-based PaliGemma language layer (same L-index convention as prior scans)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "layer_scan_curve.png", dpi=180); plt.close(fig)
    print(json.dumps({"selected_L": selected["layers"][0], "single_metrics": {k: selected[k] for k in ("selection_score", "target_response", "mask_response", "correct_response_fraction", "synonym_mask_jaccard")}, "sparse_pair": sparse["layers"] if sparse else None}, indent=2))


if __name__ == "__main__":
    main()
