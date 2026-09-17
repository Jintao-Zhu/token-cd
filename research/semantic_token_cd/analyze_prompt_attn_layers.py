"""Select sparse attention layers on exploration data, then audit validation once."""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.semantic_token_cd.prompt_attn_shr_policy import stable_top_m
from research.semantic_token_cd.prompt_attn_state_diagnostics import components


LAYERS = tuple(range(32))
V1 = tuple(range(16, 32))


def load_states(artifact: Path, split: str) -> list[tuple[dict, dict[str, np.ndarray]]]:
    result = []
    for path in sorted((artifact / "states").glob("*/*/step_*.json")):
        row = json.loads(path.read_text())
        if row["split"] == split:
            result.append((row, dict(np.load(path.with_suffix(".npz")))))
    expected = 60 if split == "exploration" else 30
    if len(result) != expected:
        raise RuntimeError(f"expected {expected} {split} states, found {len(result)}")
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def jaccard(a: list[int], b: list[int]) -> float:
    x, y = set(a), set(b)
    return len(x & y) / max(1, len(x | y))


def layer_scores(arrays: dict[str, np.ndarray], layers: tuple[int, ...]) -> np.ndarray:
    return arrays["original"][list(layers)].mean(axis=0)


def evaluate_config(states, layers: tuple[int, ...]) -> dict:
    synonym_jaccard = []
    synonym_cosine = []
    components_all = []
    masks = []
    visual_mass = []
    controls = []
    task_response: dict[str, list[float]] = {}
    for row, arrays in states:
        m = int(row["m"])
        original = arrays["original"][list(layers)].mean(axis=0)
        synonym = arrays["synonym"][list(layers)].mean(axis=0)
        omask = stable_top_m(original, m)
        smask = stable_top_m(synonym, m)
        synonym_jaccard.append(jaccard(omask, smask))
        synonym_cosine.append(cosine(original, synonym))
        mask = np.isin(np.arange(256), omask).astype(np.uint8)
        components_all.append(components(mask))
        masks.append(mask)
        visual_mass.append(float(original.sum()))
        control = row.get("target_control")
        if control is None:
            continue
        target = arrays["target_switch"][list(layers)].mean(axis=0)
        tmask = stable_top_m(target, m)
        old_box = np.asarray(control["old_target"], dtype=int)
        new_box = np.asarray(control["new_target"], dtype=int)
        old_norm = original / max(1e-12, float(original.sum()))
        new_norm = target / max(1e-12, float(target.sum()))
        new_gain = float(new_norm[new_box].sum() - old_norm[new_box].sum())
        old_drop = float(old_norm[old_box].sum() - new_norm[old_box].sum())
        old_set, new_set = set(omask), set(tmask)
        new_mask_gain = (len(new_set & set(new_box)) - len(old_set & set(new_box))) / m
        old_mask_drop = (len(old_set & set(old_box)) - len(new_set & set(old_box))) / m
        response = 0.5 * (new_gain + old_drop)
        mask_response = 0.5 * (new_mask_gain + old_mask_drop)
        controls.append({
            "state_id": row["state_id"], "task": row["task"],
            "attention_response": response, "mask_response": mask_response,
            "correct_both": bool(new_gain > 0 and old_drop > 0),
            "new_target_gain": new_gain, "old_target_drop": old_drop,
            "target_mask_jaccard": jaccard(omask, tmask),
        })
        task_response.setdefault(row["task"], []).append(response)
    frequency = np.mean(masks, axis=0)
    return {
        "layers": list(layers),
        "target_response": float(np.mean([x["attention_response"] for x in controls])) if controls else None,
        "target_mask_response": float(np.mean([x["mask_response"] for x in controls])) if controls else None,
        "correct_response_fraction": float(np.mean([x["correct_both"] for x in controls])) if controls else None,
        "positive_task_count": int(sum(np.mean(v) > 0 for v in task_response.values())),
        "synonym_mask_jaccard": float(np.mean(synonym_jaccard)),
        "synonym_score_cosine": float(np.mean(synonym_cosine)),
        "mean_components": float(np.mean([x["num_components"] for x in components_all])),
        "mean_isolated_ratio": float(np.mean([x["isolated_token_ratio"] for x in components_all])),
        "mean_largest_component_ratio": float(np.mean([x["largest_component_ratio"] for x in components_all])),
        "mean_visual_attention_mass": float(np.mean(visual_mass)),
        "fixed_position_frequency_peak": float(frequency.max()),
        "target_controls": controls,
        "target_response_by_task": {k: float(np.mean(v)) for k, v in task_response.items()},
    }


def percentile(values: list[float], value: float) -> float:
    array = np.asarray(values)
    return float((np.sum(array < value) + 0.5 * np.sum(array == value)) / len(array))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "layers", "selection_score", "target_response", "target_mask_response",
        "correct_response_fraction", "positive_task_count", "synonym_mask_jaccard",
        "synonym_score_cosine", "mean_components", "mean_isolated_ratio",
        "mean_largest_component_ratio", "mean_visual_attention_mass",
        "fixed_position_frequency_peak",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if key == "layers" else row.get(key) for key in fields})


def plot_curve(path: Path, singles: list[dict], selected: set[int]) -> None:
    x = np.arange(32)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(x, [r["target_response"] for r in singles], marker="o", label="correct target attention response")
    axes[0].axhline(0, color="black", lw=.8); axes[0].legend()
    axes[1].plot(x, [r["correct_response_fraction"] for r in singles], marker="o", label="both-direction response fraction")
    axes[1].plot(x, [r["synonym_mask_jaccard"] for r in singles], marker="o", label="synonym Top-m Jaccard")
    axes[1].legend()
    axes[2].plot(x, [r["mean_components"] for r in singles], marker="o", label="mask components")
    axes[2].plot(x, [r["fixed_position_frequency_peak"] for r in singles], marker="o", label="fixed-position peak")
    axes[2].legend(); axes[2].set_xlabel("0-based language layer")
    for ax in axes:
        for layer in selected:
            ax.axvline(layer, color="tab:red", alpha=.25)
        ax.grid(alpha=.2)
    fig.suptitle("Prompt attention layer diagnosis — exploration episodes only")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def select(artifact: Path) -> None:
    states = load_states(artifact, "exploration")
    singles = [evaluate_config(states, (layer,)) for layer in LAYERS]
    target_values = [r["target_response"] for r in singles]
    mask_values = [r["target_mask_response"] for r in singles]
    synonym_values = [r["synonym_mask_jaccard"] for r in singles]
    correctness = [r["correct_response_fraction"] for r in singles]
    for row in singles:
        # Locked before validation: target-following dominates; synonym stability
        # prevents selecting a layer that merely reacts to wording.
        row["selection_score"] = (
            .55 * percentile(target_values, row["target_response"])
            + .20 * percentile(mask_values, row["target_mask_response"])
            + .15 * percentile(correctness, row["correct_response_fraction"])
            + .10 * percentile(synonym_values, row["synonym_mask_jaccard"])
        )
    eligible = [r for r in singles if r["positive_task_count"] >= 2]
    ranked = sorted(eligible or singles, key=lambda r: (-r["selection_score"], r["layers"][0]))
    candidate_layers = [row["layers"][0] for row in ranked[:4]]
    combos = [evaluate_config(states, pair) for pair in combinations(candidate_layers, 2)]
    for row in combos:
        row["selection_score"] = (
            .55 * percentile(target_values, row["target_response"])
            + .20 * percentile(mask_values, row["target_mask_response"])
            + .15 * percentile(correctness, row["correct_response_fraction"])
            + .10 * percentile(synonym_values, row["synonym_mask_jaccard"])
        )
    single = tuple(ranked[0]["layers"])
    sparse = tuple(max(combos, key=lambda r: (r["selection_score"], -sum(r["layers"]))) ["layers"]) if combos else None
    v1 = evaluate_config(states, V1); v1["selection_score"] = None
    lock = {
        "protocol_id": "PROMPT_ATTN_LAYER_SELECTION_V1",
        "selection_data": "exploration only (20 episodes / 60 states)",
        "validation_was_read": False,
        "candidate_single_layers": candidate_layers,
        "tested_pairs": [r["layers"] for r in combos],
        "prompt_single": list(single),
        "prompt_sparse": list(sparse) if sparse else None,
        "prompt_v1": list(V1),
        "selection_rule": "0.55 target attention response percentile + 0.20 target-mask response + 0.15 correct-both fraction + 0.10 synonym Top-m stability; require positive mean response on >=2 tasks when possible",
        "single_metrics": next(r for r in singles if r["layers"] == list(single)),
        "sparse_metrics": next((r for r in combos if r["layers"] == list(sparse)), None),
        "v1_metrics": v1,
    }
    path = artifact / "CANDIDATES_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError("candidate lock differs; refusing to reselect")
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    write_csv(artifact / "exploration_per_layer.csv", singles)
    write_csv(artifact / "exploration_candidate_pairs.csv", combos)
    plot_curve(artifact / "exploration_layer_response_curve.png", singles, set(candidate_layers))
    print(json.dumps({"candidate_layers": candidate_layers, "single": single, "sparse": sparse}))


def validate(artifact: Path) -> None:
    lock = json.loads((artifact / "CANDIDATES_LOCK.json").read_text())
    if lock.get("validation_was_read") is not False:
        raise RuntimeError("invalid candidate lock")
    states = load_states(artifact, "validation")
    configs = {
        "prompt_v1": tuple(lock["prompt_v1"]),
        "prompt_single": tuple(lock["prompt_single"]),
    }
    if lock.get("prompt_sparse"):
        configs["prompt_sparse"] = tuple(lock["prompt_sparse"])
    result = {name: evaluate_config(states, layers) for name, layers in configs.items()}
    (artifact / "VALIDATION_LAYER_RESULTS.json").write_text(json.dumps({
        "protocol_id": lock["protocol_id"],
        "selection_lock_sha256": __import__("hashlib").sha256((artifact / "CANDIDATES_LOCK.json").read_bytes()).hexdigest(),
        "validation_data": "10 held-out episodes / 30 states; evaluated once after candidate lock",
        "results": result,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {k: value for k, value in row.items() if k in (
        "layers", "target_response", "target_mask_response", "correct_response_fraction",
        "synonym_mask_jaccard", "mean_components")}
        for name, row in result.items()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--stage", choices=("select", "validate"), required=True)
    args = parser.parse_args()
    args.artifact.mkdir(parents=True, exist_ok=True)
    select(args.artifact) if args.stage == "select" else validate(args.artifact)


if __name__ == "__main__":
    main()
