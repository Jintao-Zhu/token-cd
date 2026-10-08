"""Analyze the standard Bridge-Beta Pi0 layer scan for WidowX carrot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
OUT = ROOT / "token-cd/artifacts/pi0_bridge_carrot_layer_selection_v1"
TASK = "widowx_carrot_on_plate"
SEEDS = (100, 110, 120, 130, 150, 160)
N_LAYERS = 18
K = 32


def topk(x):
    return np.argsort(-x, kind="stable")[:K].tolist()


def jaccard(a, b):
    aa, bb = set(a), set(b)
    return len(aa & bb) / max(1, len(aa | bb))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def percentile(values, value):
    values = np.asarray(values, dtype=np.float64)
    return float((np.sum(values < value) + .5 * np.sum(values == value)) / len(values))


def main():
    states = []
    for seed in SEEDS:
        d = OUT / "states" / TASK / f"seed_{seed:03d}"
        meta = json.loads((d / "meta.json").read_text())
        states.append((meta, dict(np.load(d / "layers.npz"))))
    singles = []
    for layer in range(N_LAYERS):
        rows = []
        for meta, arr in states:
            original, synonym, switched = [arr[k][layer].astype(np.float64) for k in ("original", "synonym", "target_switch")]
            original /= max(1e-12, original.sum())
            synonym /= max(1e-12, synonym.sum())
            switched /= max(1e-12, switched.sum())
            old, new = arr["old_target_mask"].astype(bool), arr["new_target_mask"].astype(bool)
            new_gain = float(switched[new].sum() - original[new].sum())
            old_drop = float(original[old].sum() - switched[old].sum())
            old_top, syn_top, sw_top = topk(original), topk(synonym), topk(switched)
            old_set, sw_set = set(old_top), set(sw_top)
            mask_gain = (len(sw_set & set(np.flatnonzero(new))) - len(old_set & set(np.flatnonzero(new)))) / K
            mask_drop = (len(old_set & set(np.flatnonzero(old))) - len(sw_set & set(np.flatnonzero(old)))) / K
            rows.append({"seed": int(meta["seed"]), "target_response": .5 * (new_gain + old_drop),
                         "mask_response": .5 * (mask_gain + mask_drop),
                         "correct_both": bool(new_gain > 0 and old_drop > 0),
                         "synonym_top32_jaccard": jaccard(old_top, syn_top),
                         "synonym_attention_cosine": cosine(original, synonym)})
        singles.append({
            "layer_index_0based": layer, "layer_label": f"L{layer}",
            "target_response": float(np.mean([r["target_response"] for r in rows])),
            "mask_response": float(np.mean([r["mask_response"] for r in rows])),
        "correct_response_fraction": float(np.mean([r["correct_both"] for r in rows])),
            "positive_state_count": int(sum(r["target_response"] > 0 for r in rows)),
            "synonym_top32_jaccard": float(np.mean([r["synonym_top32_jaccard"] for r in rows])),
            "synonym_attention_cosine": float(np.mean([r["synonym_attention_cosine"] for r in rows])),
            "state_rows": rows,
        })

    metrics = ["target_response", "mask_response", "correct_response_fraction", "synonym_top32_jaccard"]
    arrays = {k: np.asarray([x[k] for x in singles]) for k in metrics}
    for row in singles:
        row["selection_score"] = (
            .55 * percentile(arrays["target_response"], row["target_response"])
            + .20 * percentile(arrays["mask_response"], row["mask_response"])
            + .15 * percentile(arrays["correct_response_fraction"], row["correct_response_fraction"])
            + .10 * percentile(arrays["synonym_top32_jaccard"], row["synonym_top32_jaccard"])
        )
    ranked = sorted(singles, key=lambda r: (-r["selection_score"], r["layer_index_0based"]))
    for i, row in enumerate(ranked, 1):
        row["all_layer_rank"] = i

    rng = np.random.default_rng(20261001)
    bootstrap_counts = np.zeros(N_LAYERS, dtype=int)
    gaps = []
    for _ in range(2000):
        idx = rng.integers(0, len(states), len(states))
        means = []
        for layer in range(N_LAYERS):
            sampled = [singles[layer]["state_rows"][j] for j in idx]
            means.append([np.mean([r["correct_both" if k == "correct_response_fraction" else k] for r in sampled]) for k in metrics])
        means = np.asarray(means, dtype=np.float64)
        score = sum(weight * np.asarray([percentile(means[:, j], means[i, j]) for i in range(N_LAYERS)])
                    for j, weight in enumerate((.55, .20, .15, .10)))
        winner = int(np.argmax(score))
        bootstrap_counts[winner] += 1
        sorted_score = np.sort(score)
        gaps.append(float(sorted_score[-1] - sorted_score[-2]))

    selected = ranked[0]
    config = json.loads((OUT / "SCAN_CONFIG.json").read_text())
    result = {
        "protocol_id": "PI0_BRIDGE_CARROT_PROMPT_ATTENTION_LAYER_SCAN_V1",
        "checkpoint": config["checkpoint"], "model": config["model"],
        "candidate_layer_space": "0-based PiZero/PaliGemma language layers L0-L17; action expert excluded",
        "states": len(states), "task": TASK, "seeds": list(SEEDS), "rank_budget": K,
        "selection_rule": "0.55 target-attention response percentile + 0.20 Top-32 target-mask response percentile + 0.15 correct-both response fraction percentile + 0.10 synonym Top-32 Jaccard percentile; single-task eligibility requires positive mean target response",
        "selected_single_layer_index_0based": selected["layer_index_0based"],
        "selected_single_layer_label": selected["layer_label"],
        "selected_single_layer_metrics": {k: v for k, v in selected.items() if k != "state_rows"},
        "ranked_layers": [{k: v for k, v in r.items() if k != "state_rows"} for r in ranked],
        "selection_stability_diagnostics": {
            "bootstrap_replicates": 2000,
            "resampling": "resample the six fixed states with replacement",
            "selected_layer_counts": {f"L{i}": int(n) for i, n in enumerate(bootstrap_counts) if n},
            "top_score_gap_quantiles": np.quantile(gaps, [.025, .5, .975]).tolist(),
            "warning": "Six-state exploratory layer selection; bootstrap only diagnoses ranking stability and is not independent validation.",
        },
        "data_usage": "Six fixed saved states used for candidate selection. No independent held-out layer audit or closed-loop comparison; proprio is fixed at normalized zero raw state.",
        "interpretation": "The selected layer is the highest offline candidate for standard Bridge-Beta Pi0 on Carrot under this frozen diagnostic; it is not established as closed-loop optimal.",
    }
    (OUT / "LAYER_SCAN_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (OUT / "per_layer.csv").open("w", newline="") as f:
        fields = ["layer_index_0based", "layer_label", "all_layer_rank", "selection_score", "target_response", "mask_response", "correct_response_fraction", "positive_state_count", "synonym_top32_jaccard", "synonym_attention_cosine"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in ranked)
    x = np.arange(N_LAYERS)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in (("target_response", "target-attention response"), ("mask_response", "Top-32 target-mask response"), ("synonym_top32_jaccard", "synonym Top-32 Jaccard")):
        ax.plot(x, [r[key] for r in singles], marker="o", label=label)
    ax.axvline(selected["layer_index_0based"], color="tab:red", linestyle="--", label=f"selected {selected['layer_label']}")
    ax.set_xlabel("0-based PaliGemma language layer")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "layer_scan_curve.png", dpi=180); plt.close(fig)
    print(json.dumps({"selected_layer": selected["layer_label"], "top_layers": [{"layer": r["layer_label"], "score": r["selection_score"], "target_response": r["target_response"]} for r in ranked[:5]], "bootstrap_counts": result["selection_stability_diagnostics"]["selected_layer_counts"]}, indent=2))


if __name__ == "__main__":
    main()
