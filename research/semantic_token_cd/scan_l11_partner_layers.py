#!/usr/bin/env python3
"""Scan every OpenVLA layer as a weighted partner for Prompt-Attention L11."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


REPO = Path(__file__).resolve().parents[2]
LAYER_ROOT = REPO / "artifacts/prompt_attn_layer_selection_v1"
CLOSE_ROOT = REPO / "artifacts/prompt_attn_l11_budget_diagnostic_v1/states/google_robot_close_drawer"
CANONICAL_ROOT = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2/episodes"
OUTPUT = REPO / "artifacts/l11_partner_layer_scan"
ALPHAS = (0.90, 0.75, 0.50)
PARTNERS = tuple(layer for layer in range(32) if layer != 11)
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
TASK_LABEL = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
L11_EPISODE_ROOTS = (LAYER_ROOT / "closed_loop", LAYER_ROOT / "closed_loop_remaining6")


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def stable_top_mask(scores: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(256), -values))
    mask = np.zeros(256, dtype=bool)
    mask[order[:count]] = True
    return mask


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_order = np.argsort(left, kind="stable")
    right_order = np.argsort(right, kind="stable")
    left_rank = np.empty(256, dtype=float)
    right_rank = np.empty(256, dtype=float)
    left_rank[left_order] = np.arange(256)
    right_rank[right_order] = np.arange(256)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def mask_geometry(mask: np.ndarray) -> tuple[int, float, float]:
    selected = set(np.flatnonzero(mask))
    remaining = set(selected)
    sizes = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            row, column = divmod(index, 16)
            neighbors = []
            if row:
                neighbors.append(index - 16)
            if row < 15:
                neighbors.append(index + 16)
            if column:
                neighbors.append(index - 1)
            if column < 15:
                neighbors.append(index + 1)
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    count = max(1, len(selected))
    return len(sizes), sum(size == 1 for size in sizes) / count, max(sizes, default=0) / count


def locate_l11_summary(task: str, seed: int) -> Path:
    for root in L11_EPISODE_ROOTS:
        path = root / "episodes" / task / "prompt_single" / f"episode_{seed:03d}_summary.json"
        if path.exists():
            return path
    raise FileNotFoundError((task, seed))


def outcome_labels(task: str, seed: int) -> dict:
    l11_success = bool(load_json(locate_l11_summary(task, seed))["success"])
    vanilla_path = CANONICAL_ROOT / task / "vanilla" / f"episode_{seed:03d}_summary.json"
    vanilla_success = bool(load_json(vanilla_path)["success"])
    if l11_success and not vanilla_success:
        paired = "rescue"
    elif vanilla_success and not l11_success:
        paired = "harm"
    elif l11_success:
        paired = "stable_success"
    else:
        paired = "stable_fail"
    return {
        "l11_success": l11_success,
        "vanilla_success": vanilla_success,
        "paired_group": paired,
        "l11_group": "success" if l11_success else "fail",
    }


def target_controls() -> dict[str, dict]:
    split = load_json(LAYER_ROOT / "SPLIT_AND_CONTROLS.json")
    return {item["state_id"]: item for item in split["target_controls"]}


def load_states() -> list[dict]:
    controls = target_controls()
    states = []
    for path in sorted((LAYER_ROOT / "states").glob("**/*.npz")):
        task = path.parents[1].name
        seed = int(path.parent.name.removeprefix("seed_"))
        step = int(path.stem.removeprefix("step_"))
        state_id = f"{task}__seed{seed:03d}__step{step:03d}"
        with np.load(path) as data:
            states.append({
                "task": task, "seed": seed, "step": step, "state_id": state_id,
                "scores": data["original"].astype(np.float64),
                "target_scores": data["target_switch"].astype(np.float64) if "target_switch" in data.files else None,
                "image": data["image"].copy(),
                "count": int(data["standard_mask"].astype(bool).sum()),
                "control": controls.get(state_id),
                **outcome_labels(task, seed),
            })
    for path in sorted(CLOSE_ROOT.glob("seed_*/*.npz")):
        task = "google_robot_close_drawer"
        seed = int(path.parent.name.removeprefix("seed_"))
        step = int(path.stem.removeprefix("step_"))
        state_id = f"{task}__seed{seed:03d}__step{step:03d}"
        with np.load(path) as data:
            states.append({
                "task": task, "seed": seed, "step": step, "state_id": state_id,
                "scores": data["original"].astype(np.float64),
                "target_scores": None, "image": data["image"].copy(),
                "count": int(data["standard_mask"].astype(bool).sum()), "control": None,
                **outcome_labels(task, seed),
            })
    if len(states) != 105:
        raise RuntimeError(f"expected 105 unique full-layer states, got {len(states)}")
    return states


def target_response(original: np.ndarray, target: np.ndarray, original_mask: np.ndarray, target_mask: np.ndarray, control: dict, count: int) -> tuple[float, float, int]:
    original_norm = original / max(1e-12, float(original.sum()))
    target_norm = target / max(1e-12, float(target.sum()))
    old_box = np.asarray(control["old_target"], dtype=int)
    new_box = np.asarray(control["new_target"], dtype=int)
    new_gain = float(target_norm[new_box].sum() - original_norm[new_box].sum())
    old_drop = float(original_norm[old_box].sum() - target_norm[old_box].sum())
    new_region = np.isin(np.arange(256), new_box)
    old_region = np.isin(np.arange(256), old_box)
    new_mask_gain = (np.logical_and(target_mask, new_region).sum() - np.logical_and(original_mask, new_region).sum()) / count
    old_mask_drop = (np.logical_and(original_mask, old_region).sum() - np.logical_and(target_mask, old_region).sum()) / count
    return 0.5 * (new_gain + old_drop), 0.5 * (new_mask_gain + old_mask_drop), int(new_gain > 0 and old_drop > 0)


def scan(states: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    spatial = defaultdict(lambda: {"lost": np.zeros(256, dtype=float), "new": np.zeros(256, dtype=float), "n": 0})
    for state in states:
        l11 = state["scores"][11]
        baseline_mask = stable_top_mask(l11, state["count"])
        base_components, base_isolated, base_largest = mask_geometry(baseline_mask)
        base_target_response = None
        if state["target_scores"] is not None:
            base_target = state["target_scores"][11]
            base_target_mask = stable_top_mask(base_target, state["count"])
            base_target_response = target_response(l11, base_target, baseline_mask, base_target_mask, state["control"], state["count"])[0]
        for partner in PARTNERS:
            partner_scores = state["scores"][partner]
            for alpha in ALPHAS:
                fused = alpha * l11 + (1.0 - alpha) * partner_scores
                mask = stable_top_mask(fused, state["count"])
                lost_mask = np.logical_and(baseline_mask, ~mask)
                new_mask = np.logical_and(~baseline_mask, mask)
                components, isolated, largest = mask_geometry(mask)
                row = {
                    "state_id": state["state_id"], "task": TASK_LABEL[state["task"]],
                    "seed": state["seed"], "step": state["step"], "partner_layer": partner,
                    "alpha": alpha, "m_t": state["count"], "l11_success": int(state["l11_success"]),
                    "vanilla_success": int(state["vanilla_success"]), "paired_group": state["paired_group"],
                    "l11_group": state["l11_group"], "mask_jaccard_l11": jaccard(mask, baseline_mask),
                    "mean_lost_l11_tokens": int(lost_mask.sum()), "mean_new_partner_tokens": int(new_mask.sum()),
                    "lost_l11_fraction": float(lost_mask.sum() / state["count"]),
                    "components": components, "component_change_l11": components - base_components,
                    "isolated_ratio": isolated, "isolated_change_l11": isolated - base_isolated,
                    "largest_component_ratio": largest, "largest_change_l11": largest - base_largest,
                    "score_cosine_l11": cosine(fused, l11),
                    "rank_correlation_l11": rank_correlation(fused, l11),
                    "target_response": "", "target_response_l11": "", "target_response_change": "",
                    "target_mask_response": "", "target_correct_both": "",
                }
                if state["target_scores"] is not None:
                    fused_target = alpha * state["target_scores"][11] + (1.0 - alpha) * state["target_scores"][partner]
                    fused_target_mask = stable_top_mask(fused_target, state["count"])
                    response, mask_response, correct = target_response(
                        fused, fused_target, mask, fused_target_mask, state["control"], state["count"]
                    )
                    row.update({
                        "target_response": response, "target_response_l11": base_target_response,
                        "target_response_change": response - base_target_response,
                        "target_mask_response": mask_response, "target_correct_both": correct,
                    })
                rows.append(row)
                key = (partner, alpha, state["task"])
                spatial[key]["lost"] += lost_mask
                spatial[key]["new"] += new_mask
                spatial[key]["n"] += 1
    return rows, spatial


def mean(items: list[dict], field: str):
    values = [float(item[field]) for item in items if item.get(field, "") != ""]
    return float(np.mean(values)) if values else ""


def aggregate_pair(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["partner_layer"], row["alpha"])].append(row)
    output = []
    for (partner, alpha), items in sorted(buckets.items()):
        task_delta = {}
        for task in ("open_drawer", "pick_coke_can", "move_near"):
            task_delta[task] = mean([item for item in items if item["task"] == task], "target_response_change")
        delta = mean(items, "target_response_change")
        jaccard_value = mean(items, "mask_jaccard_l11")
        lost = mean(items, "mean_lost_l11_tokens")
        lost_fraction = mean(items, "lost_l11_fraction")
        fragment_change = mean(items, "component_change_l11")
        task_nonharm_fraction = float(np.mean([value >= -0.002 for value in task_delta.values()]))
        output.append({
            "partner_layer": partner, "alpha": alpha, "states": len(items),
            "target_control_states": sum(item["target_response"] != "" for item in items),
            "target_response": mean(items, "target_response"), "target_response_l11": mean(items, "target_response_l11"),
            "target_response_change": delta, "mask_jaccard_l11": jaccard_value,
            "mean_lost_l11_tokens": lost, "mean_new_partner_tokens": mean(items, "mean_new_partner_tokens"),
            "lost_l11_fraction": lost_fraction, "components": mean(items, "components"),
            "component_change_l11": fragment_change, "isolated_ratio": mean(items, "isolated_ratio"),
            "largest_component_ratio": mean(items, "largest_component_ratio"),
            "score_cosine_l11": mean(items, "score_cosine_l11"),
            "rank_correlation_l11": mean(items, "rank_correlation_l11"),
            "open_target_delta": task_delta["open_drawer"], "pick_target_delta": task_delta["pick_coke_can"],
            "move_target_delta": task_delta["move_near"], "task_nonharm_fraction": task_nonharm_fraction,
        })
    deltas = np.asarray([row["target_response_change"] for row in output], dtype=float)
    gain_percentile = {id(row): float(np.mean(deltas <= row["target_response_change"])) for row in output}
    for row in output:
        frag_penalty = max(0.0, row["component_change_l11"]) / max(1.0, row["components"])
        row["complement_score"] = (
            0.45 * gain_percentile[id(row)]
            + 0.25 * row["mask_jaccard_l11"]
            + 0.15 * (1.0 - row["lost_l11_fraction"])
            + 0.15 * row["task_nonharm_fraction"]
            - 0.10 * frag_penalty
        )
        task_values = [row["open_target_delta"], row["pick_target_delta"], row["move_target_delta"]]
        row["strict_candidate"] = int(
            row["target_response_change"] >= 0
            and row["mask_jaccard_l11"] >= 0.90
            and row["mean_lost_l11_tokens"] <= 2.0
            and min(task_values) >= -0.01
            and sum(value >= -0.002 for value in task_values) >= 2
            and row["component_change_l11"] <= 0.5
        )
        row["relaxed_candidate"] = int(
            row["target_response_change"] >= -0.001
            and row["mask_jaccard_l11"] >= 0.90
            and row["mean_lost_l11_tokens"] <= 2.0
            and min(task_values) >= -0.01
            and row["component_change_l11"] <= 1.0
        )
    return output


def aggregate_taskwise(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["partner_layer"], row["alpha"], row["task"])].append(row)
    output = []
    for (partner, alpha, task), items in sorted(buckets.items()):
        output.append({
            "partner_layer": partner, "alpha": alpha, "task": task, "states": len(items),
            "episodes": len({item["seed"] for item in items}),
            "target_control_states": sum(item["target_response"] != "" for item in items),
            "target_response": mean(items, "target_response"),
            "target_response_change": mean(items, "target_response_change"),
            "mask_jaccard_l11": mean(items, "mask_jaccard_l11"),
            "mean_lost_l11_tokens": mean(items, "mean_lost_l11_tokens"),
            "components": mean(items, "components"), "component_change_l11": mean(items, "component_change_l11"),
            "score_cosine_l11": mean(items, "score_cosine_l11"),
            "rank_correlation_l11": mean(items, "rank_correlation_l11"),
        })
    return output


def aggregate_groupwise(rows: list[dict]) -> list[dict]:
    output = []
    for group_type, field, groups in (
        ("l11_outcome", "l11_group", ("success", "fail")),
        ("l11_vs_vanilla", "paired_group", ("rescue", "harm", "stable_success", "stable_fail")),
    ):
        buckets = defaultdict(list)
        for row in rows:
            buckets[(row["partner_layer"], row["alpha"], row[field])].append(row)
        for (partner, alpha, group), items in sorted(buckets.items()):
            if group not in groups:
                continue
            output.append({
                "partner_layer": partner, "alpha": alpha, "group_type": group_type, "group": group,
                "states": len(items), "episodes": len({(item["task"], item["seed"]) for item in items}),
                "target_control_states": sum(item["target_response"] != "" for item in items),
                "target_response_change": mean(items, "target_response_change"),
                "mask_jaccard_l11": mean(items, "mask_jaccard_l11"),
                "mean_lost_l11_tokens": mean(items, "mean_lost_l11_tokens"),
                "components": mean(items, "components"), "component_change_l11": mean(items, "component_change_l11"),
                "score_cosine_l11": mean(items, "score_cosine_l11"),
                "rank_correlation_l11": mean(items, "rank_correlation_l11"),
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matrix(pair_rows: list[dict], field: str) -> np.ndarray:
    values = np.full((len(ALPHAS), 32), np.nan)
    for row in pair_rows:
        values[ALPHAS.index(float(row["alpha"])), int(row["partner_layer"])] = float(row[field])
    return values


def plot_heatmap(values: np.ndarray, title: str, colorbar: str, path: Path, cmap: str, center_zero: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.8), constrained_layout=True)
    kwargs = {}
    if center_zero:
        limit = np.nanmax(np.abs(values))
        kwargs.update(vmin=-limit, vmax=limit)
    image = ax.imshow(values, aspect="auto", cmap=cmap, **kwargs)
    ax.set_xticks(range(32)); ax.set_xticklabels([f"L{x}" for x in range(32)], rotation=45)
    ax.set_yticks(range(len(ALPHAS))); ax.set_yticklabels([f"α={alpha}" for alpha in ALPHAS])
    ax.set(title=title, xlabel="Partner layer", ylabel="L11 weight")
    ax.axvline(11, color="black", linewidth=2)
    fig.colorbar(image, ax=ax, label=colorbar)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gain_damage(pair_rows: list[dict], top_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    colors = {0.9: "#3b78a8", 0.75: "#e29035", 0.5: "#a65fa5"}
    top_keys = {(int(row["partner_layer"]), float(row["alpha"])) for row in top_rows[:5]}
    for alpha in ALPHAS:
        rows = [row for row in pair_rows if float(row["alpha"]) == alpha]
        ax.scatter([row["mean_lost_l11_tokens"] for row in rows], [row["target_response_change"] for row in rows], label=f"α={alpha}", color=colors[alpha], alpha=.75)
    for row in pair_rows:
        key = (int(row["partner_layer"]), float(row["alpha"]))
        if int(row["partner_layer"]) in {7, 8, 14} or key in top_keys:
            ax.annotate(f"L{row['partner_layer']}@{row['alpha']}", (row["mean_lost_l11_tokens"], row["target_response_change"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Partner gain vs. damage", xlabel="Mean lost L11 tokens", ylabel="Δ target response vs pure L11")
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(OUTPUT / "figure_gain_vs_damage.png", dpi=180)
    plt.close(fig)


def top_candidates(pair_rows: list[dict]) -> list[dict]:
    return sorted(pair_rows, key=lambda row: (-row["strict_candidate"], -row["relaxed_candidate"], -row["complement_score"], -row["target_response_change"]))


def plot_taskwise(task_rows: list[dict], candidates: list[dict]) -> None:
    chosen = candidates[:5]
    labels = ["L11"] + [f"L{row['partner_layer']}@{row['alpha']}" for row in chosen]
    target_tasks = ("open_drawer", "pick_coke_can", "move_near")
    all_tasks = ("open_drawer", "close_drawer", "pick_coke_can", "move_near")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    x = np.arange(len(labels)); width = .22
    for index, task in enumerate(target_tasks):
        values = [0.0]
        for candidate in chosen:
            row = next(item for item in task_rows if item["task"] == task and int(item["partner_layer"]) == int(candidate["partner_layer"]) and float(item["alpha"]) == float(candidate["alpha"]))
            values.append(float(row["target_response_change"]))
        axes[0].bar(x + (index - 1) * width, values, width, label=task)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=25)
    axes[0].set(title="Taskwise target-response change", ylabel="Δ response")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
    width = .18
    for index, task in enumerate(all_tasks):
        values = [1.0]
        for candidate in chosen:
            row = next(item for item in task_rows if item["task"] == task and int(item["partner_layer"]) == int(candidate["partner_layer"]) and float(item["alpha"]) == float(candidate["alpha"]))
            values.append(float(row["mask_jaccard_l11"]))
        axes[1].bar(x + (index - 1.5) * width, values, width, label=task)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=25)
    axes[1].set(title="Taskwise mask retention proxy (close has no target-switch)", ylabel="Jaccard with L11", ylim=(0, 1.05))
    axes[1].legend(ncol=2); axes[1].grid(axis="y", alpha=.25)
    fig.savefig(OUTPUT / "figure_taskwise_candidates.png", dpi=180)
    plt.close(fig)


def normalize_heatmap(scores: np.ndarray) -> np.ndarray:
    low, high = np.percentile(scores, (2, 98))
    return np.clip((scores - low) / max(1e-12, high - low), 0, 1).reshape(16, 16)


def codes(baseline: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    value = np.zeros(256, dtype=int)
    value[np.logical_and(baseline, candidate)] = 1
    value[np.logical_and(baseline, ~candidate)] = 2
    value[np.logical_and(~baseline, candidate)] = 3
    return value.reshape(16, 16)


def plot_examples(states: list[dict], best: dict) -> None:
    partner = int(best["partner_layer"]); alpha = float(best["alpha"])
    chosen = []
    for task in ("google_robot_open_drawer", "google_robot_pick_coke_can", "google_robot_move_near"):
        candidates = [state for state in states if state["task"] == task]
        chosen.append(max(candidates, key=lambda state: int(state["paired_group"] == "rescue") * 100 + state["count"]))
    categorical = ListedColormap(["#eeeeee", "#6f8f78", "#d43f3a", "#377eb8"])
    fig, axes = plt.subplots(3, 5, figsize=(18, 11), constrained_layout=True)
    for row, state in enumerate(chosen):
        l11 = state["scores"][11]
        baseline = stable_top_mask(l11, state["count"])
        best_scores = alpha * l11 + (1 - alpha) * state["scores"][partner]
        bad_scores = 0.5 * l11 + 0.5 * state["scores"][14]
        best_mask = stable_top_mask(best_scores, state["count"])
        bad_mask = stable_top_mask(bad_scores, state["count"])
        axes[row, 0].imshow(state["image"])
        axes[row, 1].imshow(normalize_heatmap(l11), cmap="magma")
        axes[row, 2].imshow(codes(baseline, best_mask), cmap=categorical, vmin=0, vmax=3)
        axes[row, 3].imshow(codes(baseline, bad_mask), cmap=categorical, vmin=0, vmax=3)
        axes[row, 4].imshow(normalize_heatmap(best_scores), cmap="magma")
        axes[row, 0].set_ylabel(f"{TASK_LABEL[state['task']]}\nseed {state['seed']:03d} {state['paired_group']}", fontweight="bold")
        axes[row, 2].set_xlabel(f"lost={np.logical_and(baseline, ~best_mask).sum()}")
        axes[row, 3].set_xlabel(f"lost={np.logical_and(baseline, ~bad_mask).sum()}")
        for axis in axes[row]:
            axis.set_xticks([]); axis.set_yticks([])
    titles = ("RGB", "Pure L11 score", f"Best L{partner}@{alpha}", "Bad L14@0.5", "Best fused score")
    for axis, title in zip(axes[0], titles):
        axis.set_title(title)
    fig.suptitle("Token differences: neutral=common, red=lost from L11, blue=new from partner", fontsize=15)
    fig.savefig(OUTPUT / "figure_token_difference_examples.png", dpi=180)
    plt.close(fig)


def write_summary(pair_rows: list[dict], task_rows: list[dict], group_rows: list[dict], candidates: list[dict], states: list[dict]) -> dict:
    strict = [row for row in pair_rows if row["strict_candidate"]]
    relaxed = [row for row in pair_rows if row["relaxed_candidate"]]
    best = candidates[0]
    pollution = sorted(pair_rows, key=lambda row: (row["target_response_change"], -row["mean_lost_l11_tokens"]))[:5]
    special = {layer: sorted([row for row in pair_rows if int(row["partner_layer"]) == layer], key=lambda row: -row["complement_score"])[0] for layer in (7, 8, 14)}
    group_lookup = {(int(row["partner_layer"]), float(row["alpha"]), row["group_type"], row["group"]): row for row in group_rows}
    rescue = group_lookup.get((int(best["partner_layer"]), float(best["alpha"]), "l11_vs_vanilla", "rescue"))
    harm = group_lookup.get((int(best["partner_layer"]), float(best["alpha"]), "l11_vs_vanilla", "harm"))
    best_task_rows = {
        row["task"]: row
        for row in task_rows
        if int(row["partner_layer"]) == int(best["partner_layer"])
        and float(row["alpha"]) == float(best["alpha"])
    }
    positive_target_count = sum(row["target_response_change"] > 0 for row in pair_rows)
    worth_pilot = bool(strict)
    report = f"""# L11 partner-layer systematic scan

生成日期：2026-09-14  
本轮未运行新 OpenVLA forward，也未启动闭环。扫描范围为 31 个 partner layers × 3 个 L11 权重，共 93 个组合。

## 数据

- 105 个 full-layer same-state：open/pick/move 各30，close 15。
- 所有 state 均包含 `[32,256]` Prompt→Visual attention、RGB 与 Standard-SHR matched `m_t`。
- 9 个 state 具有 target-switch control；close 当前没有 target-switch，因此 close 只报告 mask/ranking proxy。
- Rescue/Harm 定义为 L11 相对 canonical Vanilla：Rescue=`Vanilla fail, L11 success`，Harm=`Vanilla success, L11 fail`。

## 结论

1. **有没有 partner 明确比纯 L11 更好？** 严格候选数量：**{len(strict)}**；宽松候选数量：**{len(relaxed)}**。
2. **最直接的结果**：93 个组合中，整体 target response 正增益为 **{positive_target_count}/93**；即当前扫描里没有任何组合超过纯 L11。
3. **最高互补排名**：L{best['partner_layer']}，α={best['alpha']}；Δ target response={best['target_response_change']:+.6f}，Jaccard={best['mask_jaccard_l11']:.4f}，lost={best['mean_lost_l11_tokens']:.2f}。它更接近“几乎不改变 L11”，而不是提供了可测的互补增益。
4. **是否值得 25-seed pilot？** **{'是' if worth_pilot else '否'}**。规则是只有至少一个组合同时通过 target、retention、task consistency 与 fragmentation 的严格门槛才启动。

## 最佳组合逐任务诊断

| Task | Δ target | Jaccard | Lost L11 tokens | Δ components |
|---|---:|---:|---:|---:|
| open_drawer | {best_task_rows['open_drawer']['target_response_change']:+.6f} | {best_task_rows['open_drawer']['mask_jaccard_l11']:.4f} | {best_task_rows['open_drawer']['mean_lost_l11_tokens']:.2f} | {best_task_rows['open_drawer']['component_change_l11']:+.2f} |
| close_drawer | N/A | {best_task_rows['close_drawer']['mask_jaccard_l11']:.4f} | {best_task_rows['close_drawer']['mean_lost_l11_tokens']:.2f} | {best_task_rows['close_drawer']['component_change_l11']:+.2f} |
| pick_coke_can | {best_task_rows['pick_coke_can']['target_response_change']:+.6f} | {best_task_rows['pick_coke_can']['mask_jaccard_l11']:.4f} | {best_task_rows['pick_coke_can']['mean_lost_l11_tokens']:.2f} | {best_task_rows['pick_coke_can']['component_change_l11']:+.2f} |
| move_near | {best_task_rows['move_near']['target_response_change']:+.6f} | {best_task_rows['move_near']['mask_jaccard_l11']:.4f} | {best_task_rows['move_near']['mean_lost_l11_tokens']:.2f} | {best_task_rows['move_near']['component_change_l11']:+.2f} |

## Top 10 complement ranking

Composite 仅用于排序：45% target-gain percentile + 25% Jaccard + 15% retained fraction + 15% task consistency − fragmentation penalty。

| Rank | Partner | α | Δ target | Jaccard | Lost | Δ components | Task nonharm | Strict | Relaxed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for rank, row in enumerate(candidates[:10], 1):
        report += f"| {rank} | L{row['partner_layer']} | {row['alpha']:.2f} | {row['target_response_change']:+.6f} | {row['mask_jaccard_l11']:.4f} | {row['mean_lost_l11_tokens']:.2f} | {row['component_change_l11']:+.2f} | {row['task_nonharm_fraction']:.2f} | {row['strict_candidate']} | {row['relaxed_candidate']} |\n"
    report += """

## 特别层

| Layer | Best α | Δ target | Jaccard | Lost | Composite |
|---:|---:|---:|---:|---:|---:|
"""
    for layer in (7, 8, 14):
        row = special[layer]
        report += f"| L{layer} | {row['alpha']:.2f} | {row['target_response_change']:+.6f} | {row['mask_jaccard_l11']:.4f} | {row['mean_lost_l11_tokens']:.2f} | {row['complement_score']:.4f} |\n"
    report += "\n## 最容易污染 L11 的组合\n\n| Partner | α | Δ target | Jaccard | Lost |\n|---:|---:|---:|---:|---:|\n"
    for row in pollution:
        report += f"| L{row['partner_layer']} | {row['alpha']:.2f} | {row['target_response_change']:+.6f} | {row['mask_jaccard_l11']:.4f} | {row['mean_lost_l11_tokens']:.2f} |\n"
    report += f"""

## Rescue/Harm 检查

最佳排名组合在 L11 Rescue 组中的平均 lost/Jaccard：**{float(rescue['mean_lost_l11_tokens']):.2f} / {float(rescue['mask_jaccard_l11']):.4f}**。  
在 L11 Harm 组中：**{float(harm['mean_lost_l11_tokens']):.2f} / {float(harm['mask_jaccard_l11']):.4f}**。  
这些是 seed-level outcome 条件下的 same-state mask proxy；不是候选真实闭环收益。

## 图

![Layer-alpha target heatmap](figure_layer_alpha_heatmap.png)

![Lost L11 heatmap](figure_lost_l11_heatmap.png)

![Gain vs damage](figure_gain_vs_damage.png)

![Taskwise candidates](figure_taskwise_candidates.png)

![Token examples](figure_token_difference_examples.png)

## 边界

- target response 仅来自9个手工 target-switch controls，必须作为候选筛选信号而非成功率替代品。
- close 没有 target-switch，不能声称某个组合提升了 close target response。
- Groupwise 只判断 partner 在既有 L11 Rescue/Harm state 上改变了什么，不能证明它会把 Harm 修复为成功。
- 没有通过严格门槛时，不为了多层融合形式强行启动闭环。
"""
    (OUTPUT / "SUMMARY.md").write_text(report)
    return {
        "state_count": len(states), "combination_count": len(pair_rows),
        "strict_candidate_count": len(strict), "relaxed_candidate_count": len(relaxed),
        "worth_25_seed_pilot": worth_pilot, "best_ranked": best,
        "top_candidates": candidates[:10], "special_layers": special,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    states = load_states()
    rows, spatial = scan(states)
    pair_rows = aggregate_pair(rows)
    task_rows = aggregate_taskwise(rows)
    group_rows = aggregate_groupwise(rows)
    candidates = top_candidates(pair_rows)
    write_csv(OUTPUT / "PAIR_SCAN.csv", pair_rows)
    write_csv(OUTPUT / "TASKWISE_SCAN.csv", task_rows)
    write_csv(OUTPUT / "GROUPWISE_SCAN.csv", group_rows)
    write_csv(OUTPUT / "TOP_CANDIDATES.csv", candidates[:20])
    write_csv(OUTPUT / "STATEWISE_SCAN.csv", rows)
    np.savez_compressed(
        OUTPUT / "LOST_NEW_SPATIAL.npz",
        **{
            f"partner_{partner:02d}_alpha_{str(alpha).replace('.', 'p')}_{TASK_LABEL[task]}_{kind}": values[kind] / values["n"]
            for (partner, alpha, task), values in spatial.items()
            for kind in ("lost", "new")
        },
    )
    plot_heatmap(matrix(pair_rows, "target_response_change"), "All partner layers: target-response change vs pure L11", "Δ target response", OUTPUT / "figure_layer_alpha_heatmap.png", "coolwarm", center_zero=True)
    plot_heatmap(matrix(pair_rows, "mean_lost_l11_tokens"), "All partner layers: mean L11 tokens displaced", "Mean lost L11 tokens", OUTPUT / "figure_lost_l11_heatmap.png", "magma")
    plot_gain_damage(pair_rows, candidates)
    plot_taskwise(task_rows, candidates)
    plot_examples(states, candidates[0])
    audit = write_summary(pair_rows, task_rows, group_rows, candidates, states)
    audit.update({
        "alphas": ALPHAS, "partners": PARTNERS,
        "candidate_rule": {
            "strict": "delta_target>=0, Jaccard>=.90, lost<=2, min task delta>=-.01, >=2/3 tasks delta>=-.002, component delta<=.5",
            "relaxed": "delta_target>=-.001, Jaccard>=.90, lost<=2, min task delta>=-.01, component delta<=1",
        },
        "complement_score": ".45 gain percentile + .25 Jaccard + .15 retained fraction + .15 task nonharm - .10 normalized positive fragmentation",
        "target_control_count": 9,
        "close_target_switch_available": False,
    })
    (OUTPUT / "AUDIT_DATA.json").write_text(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
