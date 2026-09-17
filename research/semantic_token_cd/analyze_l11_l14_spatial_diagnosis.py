#!/usr/bin/env python3
"""Diagnose whether L14 pollutes L11 ranking or needs spatial regularization."""

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
from scipy.ndimage import gaussian_filter


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "artifacts/prompt_attn_layer_selection_v1"
OUTPUT = REPO / "artifacts/l11_l14_spatial_diagnosis_v1"
ALPHAS = (1.0, 0.9, 0.75, 0.5)
SPATIAL_MODES = ("raw", "corners", "gaussian", "both")
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
ROOTS = (SOURCE / "closed_loop", SOURCE / "closed_loop_remaining6")


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def locate(task: str, arm: str, seed: int, suffix: str) -> Path:
    for root in ROOTS:
        path = root / "episodes" / task / arm / f"episode_{seed:03d}_{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError((task, arm, seed, suffix))


def top_mask(scores: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(256), -values))
    mask = np.zeros(256, dtype=bool)
    mask[order[:count]] = True
    return mask


def spatial(scores: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).copy()
    if mode in {"corners", "both"}:
        values[[0, 15, 240, 255]] = 0.0
    if mode in {"gaussian", "both"}:
        values = gaussian_filter(values.reshape(16, 16), 0.65, mode="reflect", truncate=4).ravel()
    return values


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def mask_geometry(mask: np.ndarray) -> tuple[int, float, float]:
    selected = set(np.flatnonzero(mask))
    remaining = set(selected)
    component_sizes = []
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
        component_sizes.append(size)
    count = max(1, len(selected))
    isolated = sum(size == 1 for size in component_sizes) / count
    largest = max(component_sizes, default=0) / count
    return len(component_sizes), isolated, largest


def outcome_group(l11_success: bool, sparse_success: bool) -> str:
    if l11_success and not sparse_success:
        return "l11_win"
    if sparse_success and not l11_success:
        return "sparse_win"
    if l11_success:
        return "both_success"
    return "both_fail"


def state_identity(path: Path) -> tuple[str, int, int, str]:
    task = path.parents[1].name
    seed = int(path.parent.name.removeprefix("seed_"))
    step = int(path.stem.removeprefix("step_"))
    state_id = f"{task}__seed{seed:03d}__step{step:03d}"
    return task, seed, step, state_id


def evaluate_mask(scores: np.ndarray, baseline: np.ndarray, count: int) -> dict:
    mask = top_mask(scores, count)
    common = np.logical_and(mask, baseline).sum()
    components, isolated, largest = mask_geometry(mask)
    return {
        "mask": mask,
        "jaccard_l11": jaccard(mask, baseline),
        "l11_retention": float(common / count),
        "lost_from_l11": int(count - common),
        "new_vs_l11": int(count - common),
        "components": components,
        "isolated_ratio": isolated,
        "largest_component_ratio": largest,
    }


def load_outcomes() -> dict[tuple[str, int], dict]:
    outcomes = {}
    for task in TASKS:
        for seed in range(100):
            single = load_json(locate(task, "prompt_single", seed, "summary.json"))
            sparse = load_json(locate(task, "prompt_sparse", seed, "summary.json"))
            l11_success = bool(single["success"])
            sparse_success = bool(sparse["success"])
            outcomes[(task, seed)] = {
                "l11_success": l11_success,
                "sparse_success": sparse_success,
                "outcome_group": outcome_group(l11_success, sparse_success),
            }
    return outcomes


def control_map() -> dict[str, dict]:
    split = load_json(SOURCE / "SPLIT_AND_CONTROLS.json")
    return {item["state_id"]: item for item in split["target_controls"]}


def analyze_cached_states(outcomes: dict, controls: dict) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    state_payload = {}
    for path in sorted((SOURCE / "states").glob("**/*.npz")):
        task, seed, step, state_id = state_identity(path)
        with np.load(path) as data:
            original = data["original"].astype(np.float64)
            synonym = data["synonym"].astype(np.float64)
            target = data["target_switch"].astype(np.float64) if "target_switch" in data.files else None
            image = data["image"].copy()
            count = int(data["standard_mask"].astype(bool).sum())
        raw_l11 = top_mask(original[11], count)
        state_payload[state_id] = {
            "task": task, "seed": seed, "step": step, "image": image,
            "a11": original[11], "a14": original[14], "count": count,
            "outcome_group": outcomes[(task, seed)]["outcome_group"],
        }
        for alpha in ALPHAS:
            original_mix = alpha * original[11] + (1.0 - alpha) * original[14]
            synonym_mix = alpha * synonym[11] + (1.0 - alpha) * synonym[14]
            target_mix = None if target is None else alpha * target[11] + (1.0 - alpha) * target[14]
            for mode in SPATIAL_MODES:
                processed = spatial(original_mix, mode)
                result = evaluate_mask(processed, raw_l11, count)
                synonym_mask = top_mask(spatial(synonym_mix, mode), count)
                row = {
                    "dataset": "cached_90", "task": TASK_LABEL[task], "seed": seed, "step": step,
                    "state_id": state_id, "outcome_group": outcomes[(task, seed)]["outcome_group"],
                    "alpha": alpha, "spatial_mode": mode, "m_t": count,
                    "jaccard_l11": result["jaccard_l11"], "l11_retention": result["l11_retention"],
                    "lost_from_l11": result["lost_from_l11"], "new_vs_l11": result["new_vs_l11"],
                    "components": result["components"], "isolated_ratio": result["isolated_ratio"],
                    "largest_component_ratio": result["largest_component_ratio"],
                    "synonym_mask_jaccard": jaccard(result["mask"], synonym_mask),
                    "target_response": "", "target_mask_response": "", "target_correct_both": "",
                }
                if target_mix is not None:
                    control = controls[state_id]
                    target_processed = spatial(target_mix, mode)
                    target_mask = top_mask(target_processed, count)
                    original_norm = processed / max(1e-12, float(processed.sum()))
                    target_norm = target_processed / max(1e-12, float(target_processed.sum()))
                    old_box = np.asarray(control["old_target"], dtype=int)
                    new_box = np.asarray(control["new_target"], dtype=int)
                    new_gain = float(target_norm[new_box].sum() - original_norm[new_box].sum())
                    old_drop = float(original_norm[old_box].sum() - target_norm[old_box].sum())
                    new_mask_gain = (
                        np.logical_and(target_mask, np.isin(np.arange(256), new_box)).sum()
                        - np.logical_and(result["mask"], np.isin(np.arange(256), new_box)).sum()
                    ) / count
                    old_mask_drop = (
                        np.logical_and(result["mask"], np.isin(np.arange(256), old_box)).sum()
                        - np.logical_and(target_mask, np.isin(np.arange(256), old_box)).sum()
                    ) / count
                    row.update({
                        "target_response": 0.5 * (new_gain + old_drop),
                        "target_mask_response": 0.5 * (new_mask_gain + old_mask_drop),
                        "target_correct_both": int(new_gain > 0 and old_drop > 0),
                    })
                rows.append(row)
    return rows, state_payload


def analyze_initial_states(outcomes: dict) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    rows = []
    payload = {}
    for task in TASKS:
        for seed in range(100):
            with np.load(locate(task, "prompt_single", seed, "arrays.npz")) as single, np.load(
                locate(task, "prompt_sparse", seed, "arrays.npz")
            ) as sparse:
                a11 = single["prompt_attention"][0].astype(np.float64)
                average = sparse["prompt_attention"][0].astype(np.float64)
                a14 = 2.0 * average - a11
                count = int(single["selected_mask"][0].sum())
                if count != int(sparse["selected_mask"][0].sum()):
                    raise RuntimeError("initial matched coverage differs")
                baseline = top_mask(a11, count)
                if not np.array_equal(baseline, single["selected_mask"][0].astype(bool)):
                    raise RuntimeError("saved L11 mask is not score Top-m")
                if not np.array_equal(top_mask(average, count), sparse["selected_mask"][0].astype(bool)):
                    raise RuntimeError("saved L11+L14 mask is not fused-score Top-m")
            payload[(task, seed)] = {
                "a11": a11, "a14": a14, "average": average, "count": count,
                **outcomes[(task, seed)],
            }
            for alpha in ALPHAS:
                mixed = alpha * a11 + (1.0 - alpha) * a14
                for mode in SPATIAL_MODES:
                    result = evaluate_mask(spatial(mixed, mode), baseline, count)
                    rows.append({
                        "dataset": "initial_400", "task": TASK_LABEL[task], "seed": seed, "step": 0,
                        "state_id": f"{task}__seed{seed:03d}__initial", "outcome_group": outcomes[(task, seed)]["outcome_group"],
                        "l11_success": int(outcomes[(task, seed)]["l11_success"]),
                        "sparse_success": int(outcomes[(task, seed)]["sparse_success"]),
                        "alpha": alpha, "spatial_mode": mode, "m_t": count,
                        "jaccard_l11": result["jaccard_l11"], "l11_retention": result["l11_retention"],
                        "lost_from_l11": result["lost_from_l11"], "new_vs_l11": result["new_vs_l11"],
                        "components": result["components"], "isolated_ratio": result["isolated_ratio"],
                        "largest_component_ratio": result["largest_component_ratio"],
                    })
    return rows, payload


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in group_fields)].append(row)
    output = []
    metrics = (
        "jaccard_l11", "l11_retention", "lost_from_l11", "new_vs_l11", "components",
        "isolated_ratio", "largest_component_ratio", "synonym_mask_jaccard",
        "target_response", "target_mask_response", "target_correct_both",
    )
    for key, items in sorted(buckets.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row = dict(zip(group_fields, key))
        row["n"] = len(items)
        for metric in metrics:
            values = [float(item[metric]) for item in items if metric in item and item[metric] != ""]
            row[metric] = float(np.mean(values)) if values else ""
        output.append(row)
    return output


def normalize_heatmap(scores: np.ndarray) -> np.ndarray:
    low, high = np.percentile(scores, (2, 98))
    return np.clip((scores - low) / max(1e-12, high - low), 0, 1).reshape(16, 16)


def replacement_codes(baseline: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    codes = np.zeros(256, dtype=int)
    codes[np.logical_and(baseline, candidate)] = 1
    codes[np.logical_and(baseline, ~candidate)] = 2
    codes[np.logical_and(~baseline, candidate)] = 3
    return codes.reshape(16, 16)


def choose_cached_representatives(payload: dict[str, dict]) -> dict[str, dict]:
    selected = {}
    for task in ("google_robot_open_drawer", "google_robot_pick_coke_can", "google_robot_move_near"):
        candidates = []
        for state_id, item in payload.items():
            if item["task"] != task:
                continue
            baseline = top_mask(item["a11"], item["count"])
            candidate = top_mask(0.5 * item["a11"] + 0.5 * item["a14"], item["count"])
            lost = int(np.logical_and(baseline, ~candidate).sum())
            priority = int(item["outcome_group"] == "l11_win")
            candidates.append((priority, lost, state_id, item, baseline, candidate))
        _, _, state_id, item, baseline, candidate = max(candidates, key=lambda value: (value[0], value[1]))
        selected[task] = {"state_id": state_id, "item": item, "baseline": baseline, "candidate": candidate}
    return selected


def choose_close_representative(payload: dict[tuple[str, int], dict]) -> dict:
    candidates = []
    task = "google_robot_close_drawer"
    for seed in range(100):
        item = payload[(task, seed)]
        baseline = top_mask(item["a11"], item["count"])
        candidate = top_mask(item["average"], item["count"])
        lost = int(np.logical_and(baseline, ~candidate).sum())
        priority = int(item["outcome_group"] == "l11_win")
        candidates.append((priority, lost, seed, item, baseline, candidate))
    _, _, seed, item, baseline, candidate = max(candidates, key=lambda value: (value[0], value[1]))
    return {"seed": seed, "item": item, "baseline": baseline, "candidate": candidate}


def figure_replacements(cached_payload: dict[str, dict], initial_payload: dict[tuple[str, int], dict]) -> None:
    reps = choose_cached_representatives(cached_payload)
    close = choose_close_representative(initial_payload)
    fig, axes = plt.subplots(4, 4, figsize=(16, 15), constrained_layout=True)
    categorical = ListedColormap(["#eeeeee", "#3ca370", "#d43f3a", "#377eb8"])
    for row, task in enumerate(TASKS):
        if task == "google_robot_close_drawer":
            item = close["item"]
            baseline, candidate = close["baseline"], close["candidate"]
            axes[row, 0].text(.5, .55, "RGB not present\nin existing close cache", ha="center", va="center", fontsize=12)
            axes[row, 0].text(.5, .25, f"seed {close['seed']:03d}\n{item['outcome_group']}", ha="center", va="center")
            axes[row, 0].set_facecolor("#f4f4f4")
            title_id = f"seed {close['seed']:03d} initial"
        else:
            rep = reps[task]
            item = rep["item"]
            baseline, candidate = rep["baseline"], rep["candidate"]
            axes[row, 0].imshow(item["image"])
            title_id = rep["state_id"].split("__", 1)[1]
        axes[row, 1].imshow(normalize_heatmap(item["a11"]), cmap="magma", interpolation="nearest")
        axes[row, 2].imshow(normalize_heatmap(0.5 * item["a11"] + 0.5 * item["a14"]), cmap="magma", interpolation="nearest")
        axes[row, 3].imshow(replacement_codes(baseline, candidate), cmap=categorical, vmin=0, vmax=3, interpolation="nearest")
        lost = int(np.logical_and(baseline, ~candidate).sum())
        common = int(np.logical_and(baseline, candidate).sum())
        for column in range(4):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"{TASK_LABEL[task]}\n{title_id}", fontsize=10, fontweight="bold")
        axes[row, 3].set_xlabel(f"common={common}, lost={lost}, new={lost}")
    for column, title in enumerate(("RGB / state", "L11 score", "50/50 L11+L14 score", "Mask replacement")):
        axes[0, column].set_title(title, fontsize=12)
    fig.suptitle("What L14 changes: green=retained L11, red=lost from L11, blue=new from L14", fontsize=16)
    fig.savefig(OUTPUT / "figure_token_replacement.png", dpi=180)
    plt.close(fig)


def metric_lookup(rows: list[dict], alpha: float, mode: str, metric: str) -> float:
    match = next(row for row in rows if float(row["alpha"]) == alpha and row["spatial_mode"] == mode)
    return float(match[metric])


def figure_scan(cached_agg: list[dict], initial_agg: list[dict]) -> None:
    cached_overall = [row for row in cached_agg if row["task"] == "ALL" and row["outcome_group"] == "ALL"]
    initial_overall = [row for row in initial_agg if row["task"] == "ALL" and row["outcome_group"] == "ALL"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = {"raw": "#333333", "corners": "#d28b32", "gaussian": "#3976a8", "both": "#a55c9e"}
    labels = {"raw": "raw", "corners": "corner only", "gaussian": "Gaussian only", "both": "corner + Gaussian"}
    for mode in SPATIAL_MODES:
        axes[0, 0].plot(ALPHAS, [metric_lookup(initial_overall, alpha, mode, "jaccard_l11") for alpha in ALPHAS], marker="o", color=colors[mode], label=labels[mode])
        axes[0, 1].plot(ALPHAS, [metric_lookup(cached_overall, alpha, mode, "target_response") for alpha in ALPHAS], marker="o", color=colors[mode], label=labels[mode])
        axes[1, 0].plot(ALPHAS, [metric_lookup(cached_overall, alpha, mode, "components") for alpha in ALPHAS], marker="o", color=colors[mode], label=labels[mode])
        axes[1, 1].plot(ALPHAS, [metric_lookup(initial_overall, alpha, mode, "lost_from_l11") for alpha in ALPHAS], marker="o", color=colors[mode], label=labels[mode])
    for ax in axes.flat:
        ax.set_xticks(ALPHAS)
        ax.invert_xaxis()
        ax.grid(alpha=.25)
        ax.set_xlabel("L11 weight alpha (right = more L14)")
    axes[0, 0].set(title="400 initial states: mask Jaccard with raw L11", ylabel="Mean Jaccard")
    axes[0, 1].set(title="90 cached states: target-switch response", ylabel="Mean response")
    axes[1, 0].set(title="90 cached states: mask fragmentation", ylabel="Mean connected components")
    axes[1, 1].set(title="400 initial states: L11 tokens displaced", ylabel="Mean lost tokens")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Weighted L11/L14 fusion and DTP-style spatial post-processing", fontsize=16)
    fig.savefig(OUTPUT / "figure_weight_spatial_scan.png", dpi=180)
    plt.close(fig)


def figure_outcome(initial_group_agg: list[dict]) -> None:
    groups = ("l11_win", "sparse_win", "both_success", "both_fail")
    labels = ("L11 wins", "L11+L14 wins", "Both success", "Both fail")
    configs = ((1.0, "raw", "L11 raw"), (0.5, "raw", "50/50 raw"), (1.0, "both", "L11 spatial"), (0.5, "both", "50/50 spatial"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    x = np.arange(len(groups)); width = .19
    for index, (alpha, mode, label) in enumerate(configs):
        values = []
        components = []
        for group in groups:
            row = next(item for item in initial_group_agg if item["task"] == "ALL" and item["outcome_group"] == group and float(item["alpha"]) == alpha and item["spatial_mode"] == mode)
            values.append(float(row["lost_from_l11"]))
            components.append(float(row["components"]))
        axes[0].bar(x + (index - 1.5) * width, values, width, label=label)
        axes[1].bar(x + (index - 1.5) * width, components, width, label=label)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12)
        ax.grid(axis="y", alpha=.25)
    axes[0].set(title="Token displacement by real closed-loop outcome", ylabel="Mean lost-from-L11 tokens")
    axes[1].set(title="Fragmentation by real closed-loop outcome", ylabel="Mean connected components")
    axes[0].legend(ncol=2)
    fig.savefig(OUTPUT / "figure_outcome_conditioned.png", dpi=180)
    plt.close(fig)


def add_all_groups(rows: list[dict]) -> list[dict]:
    expanded = []
    for row in rows:
        expanded.append(row)
        all_task = dict(row); all_task["task"] = "ALL"; expanded.append(all_task)
        all_outcome = dict(row); all_outcome["outcome_group"] = "ALL"; expanded.append(all_outcome)
        both = dict(row); both["task"] = "ALL"; both["outcome_group"] = "ALL"; expanded.append(both)
    return expanded


def summarize(cached_agg: list[dict], initial_agg: list[dict], outcome_counts: dict) -> dict:
    cached = [row for row in cached_agg if row["task"] == "ALL" and row["outcome_group"] == "ALL"]
    initial = [row for row in initial_agg if row["task"] == "ALL" and row["outcome_group"] == "ALL"]
    def get(rows, alpha, mode):
        return next(row for row in rows if float(row["alpha"]) == alpha and row["spatial_mode"] == mode)
    summary = {
        "raw_weight_scan": {},
        "spatial_at_l11": {},
        "spatial_at_equal_fusion": {},
        "raw_target_response_by_task": {},
        "outcome_counts": outcome_counts,
    }
    for alpha in ALPHAS:
        c, i = get(cached, alpha, "raw"), get(initial, alpha, "raw")
        summary["raw_weight_scan"][str(alpha)] = {
            "initial_jaccard_l11": i["jaccard_l11"],
            "initial_l11_retention": i["l11_retention"],
            "initial_lost_tokens": i["lost_from_l11"],
            "cached_target_response": c["target_response"],
            "cached_components": c["components"],
        }
    for mode in SPATIAL_MODES:
        for alpha, key in ((1.0, "spatial_at_l11"), (0.5, "spatial_at_equal_fusion")):
            c, i = get(cached, alpha, mode), get(initial, alpha, mode)
            summary[key][mode] = {
                "initial_jaccard_l11": i["jaccard_l11"],
                "initial_lost_tokens": i["lost_from_l11"],
                "cached_target_response": c["target_response"],
                "cached_components": c["components"],
                "cached_isolated_ratio": c["isolated_ratio"],
                "cached_largest_component_ratio": c["largest_component_ratio"],
            }
    for task in ("open_drawer", "pick_coke_can", "move_near"):
        task_rows = [row for row in cached_agg if row["task"] == task and row["outcome_group"] == "ALL"]
        summary["raw_target_response_by_task"][task] = {
            str(alpha): get(task_rows, alpha, "raw")["target_response"] for alpha in ALPHAS
        }
    return summary


def write_report(results: dict) -> None:
    raw = results["raw_weight_scan"]
    l11 = results["spatial_at_l11"]
    equal = results["spatial_at_equal_fusion"]
    counts = results["outcome_counts"]
    raw_target = [raw[str(alpha)]["cached_target_response"] for alpha in ALPHAS]
    monotonic_target = all(raw_target[index] >= raw_target[index + 1] for index in range(len(raw_target) - 1))
    best_equal_mode = max(SPATIAL_MODES, key=lambda mode: equal[mode]["cached_target_response"])
    raw_drop = 1.0 - raw["0.5"]["cached_target_response"] / raw["1.0"]["cached_target_response"]
    gaussian_equal_drop = 1.0 - equal["gaussian"]["cached_target_response"] / equal["raw"]["cached_target_response"]
    gaussian_component_reduction = 1.0 - equal["gaussian"]["cached_components"] / equal["raw"]["cached_components"]
    report = f"""# L11 + L14 ranking pollution vs spatial post-processing

生成日期：2026-09-14  
本轮只做离线重建，没有启动新闭环或模型 forward。

## 数据与一致性

- 90 个完整 same-state 缓存：open/pick/move 各 30 states，均有 L0–L31 attention、RGB、matched mask；其中 9 states 有 target-switch attention。
- 400 个共同初始状态：open/close/pick/move 各 100 seeds。L11 与 L11+L14 的 summary/arrays 均完整。
- 400/400 seed 的两臂初始 matched token 数一致；800/800 保存 mask 均精确等于对应 score map 的稳定 Top-m。
- 初始状态的 L14 由 `A14 = 2 * A(L11+L14) - A11` 恢复，仅用于同一初始 state 的离线加权扫描。
- 空间处理采用仓库已有 calibration：四角 score 置零；Gaussian `sigma=0.65`、reflect padding、truncate=4。

## 真实闭环输赢分组

| Group | Episodes |
|---|---:|
| L11 success / L11+L14 fail | {counts['l11_win']} |
| L11 fail / L11+L14 success | {counts['sparse_win']} |
| Both success | {counts['both_success']} |
| Both fail | {counts['both_fail']} |

## 核心结果

### 1. L14 权重增加时，L11 ranking 是否持续被替换

| L11 weight α | Jaccard with L11 | L11 retention | Lost tokens | Target response | Components |
|---:|---:|---:|---:|---:|---:|
"""
    for alpha in ALPHAS:
        row = raw[str(alpha)]
        report += f"| {alpha:.2f} | {row['initial_jaccard_l11']:.4f} | {row['initial_l11_retention']:.4f} | {row['initial_lost_tokens']:.2f} | {row['cached_target_response']:.5f} | {row['cached_components']:.2f} |\n"
    report += f"""

Raw target response 随 L14 权重增加是否单调下降：**{'是' if monotonic_target else '否'}**。

该趋势在三个有 target-control 的任务上分别成立：

| Task | α=1.0 | α=.9 | α=.75 | α=.5 |
|---|---:|---:|---:|---:|
"""
    for task in ("open_drawer", "pick_coke_can", "move_near"):
        values = results["raw_target_response_by_task"][task]
        report += f"| {task} | {values['1.0']:.5f} | {values['0.9']:.5f} | {values['0.75']:.5f} | {values['0.5']:.5f} |\n"
    report += f"""

从 α=1.0 到 α=.5，aggregate target response 下降 **{raw_drop:.1%}**，初始 mask 平均挤掉 **{raw['0.5']['initial_lost_tokens']:.2f}** 个 L11 token。

### 2. DTP-style spatial 是否能救 50/50 融合

| Spatial | 50/50 Jaccard | Lost tokens | Target response | Components | Isolated ratio | Largest component |
|---|---:|---:|---:|---:|---:|---:|
"""
    for mode in SPATIAL_MODES:
        row = equal[mode]
        report += f"| {mode} | {row['initial_jaccard_l11']:.4f} | {row['initial_lost_tokens']:.2f} | {row['cached_target_response']:.5f} | {row['cached_components']:.2f} | {row['cached_isolated_ratio']:.4f} | {row['cached_largest_component_ratio']:.4f} |\n"
    report += f"""

按 target response，50/50 融合中最佳 spatial mode 是 **{best_equal_mode}**。是否值得进入小闭环，必须同时检查它是否只是通过大幅改变 L11 mask 换来更平滑外观。

Gaussian 确实把 components 从 **{equal['raw']['cached_components']:.2f}** 降到 **{equal['gaussian']['cached_components']:.2f}**（减少 **{gaussian_component_reduction:.1%}**），但 target response 同时从 **{equal['raw']['cached_target_response']:.5f}** 降到 **{equal['gaussian']['cached_target_response']:.5f}**（再下降 **{gaussian_equal_drop:.1%}**），相对 raw L11 平均挤掉的 token 也从 **{equal['raw']['initial_lost_tokens']:.2f}** 增加到 **{equal['gaussian']['initial_lost_tokens']:.2f}**。它改善的是空间连通性，不是目标相关 ranking。

### 3. Spatial 对纯 L11 的影响

| Spatial | Jaccard with raw L11 | Lost tokens | Target response | Components |
|---|---:|---:|---:|---:|
"""
    for mode in SPATIAL_MODES:
        row = l11[mode]
        report += f"| {mode} | {row['initial_jaccard_l11']:.4f} | {row['initial_lost_tokens']:.2f} | {row['cached_target_response']:.5f} | {row['cached_components']:.2f} |\n"
    report += f"""

Corner-only 对纯 L11 几乎是恒等变换：400 个初始状态平均只替换 **{l11['corners']['initial_lost_tokens']:.2f}** 个 token，target response 仅从 **{l11['raw']['cached_target_response']:.5f}** 变为 **{l11['corners']['cached_target_response']:.5f}**。

## 当前判定

本轮证据更支持：

> **L14 主要在持续扰动 L11 的 ranking；缺失 Gaussian/corner 不是 50/50 组合失败的主要原因。**

理由：L14 权重增加时，三个任务的 target response 均单调下降；50/50 仅保留 **{raw['0.5']['initial_l11_retention']:.1%}** 的 L11 token；Gaussian 虽让 mask 更连通，却进一步降低 target response并替换更多 L11 token；corner-only 的影响又太小。另一方面，L11-win 与 sparse-win 组的替换数量很接近，说明真正重要的是被替换 token 的位置和语义，而不是数量本身。

**Go/No-Go：当前对 weighted L11+L14 + Gaussian/corner 的 25-seed 闭环判定为 No-Go。** 离线没有出现能够同时保住 L11 target response、保留 L11 ranking、并利用空间正则化改善组合的候选。唯一勉强可测的是 `L11 + corner-only`，但它与 raw L11 几乎相同，预期效应很小。

## 图

![Token replacement](figure_token_replacement.png)

![Weight and spatial scan](figure_weight_spatial_scan.png)

![Outcome conditioned](figure_outcome_conditioned.png)

## 解释边界

- 90-state target response 只有 9 个手工 target-switch controls，因此它是机制指标，不是成功率替代品。
- 400-state outcome grouping 使用共同初始 state，避免了两条闭环轨迹在第一步后状态分叉的问题。
- close_drawer 没有现成 RGB full-layer cache，因此 replacement 图展示其初始 16×16 score/mask，而非 RGB overlay。
- Gaussian/corner 的具体实现是本仓库的 paper-based calibrated implementation；原论文没有完全指定所有空间处理细节。
- 本轮不根据单一指标自动启动闭环。是否跑 25-seed pilot 应结合 target response、L11 retention、fragmentation 和 outcome-conditioned replacement 一起判断。

## 输出文件

- `CACHED_90_STATE_ROWS.csv`
- `INITIAL_400_STATE_ROWS.csv`
- `CACHED_90_AGGREGATE.csv`
- `INITIAL_400_AGGREGATE.csv`
- `INITIAL_400_OUTCOME_AGGREGATE.csv`
- `RESULTS.json`
- `figure_token_replacement.png`
- `figure_weight_spatial_scan.png`
- `figure_outcome_conditioned.png`
"""
    (OUTPUT / "SUMMARY.md").write_text(report)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outcomes = load_outcomes()
    controls = control_map()
    cached_rows, cached_payload = analyze_cached_states(outcomes, controls)
    initial_rows, initial_payload = analyze_initial_states(outcomes)
    cached_agg = aggregate(add_all_groups(cached_rows), ("task", "outcome_group", "alpha", "spatial_mode"))
    initial_agg = aggregate(add_all_groups(initial_rows), ("task", "outcome_group", "alpha", "spatial_mode"))
    initial_outcome_agg = [row for row in initial_agg if row["task"] == "ALL" and row["outcome_group"] != "ALL"]
    outcome_counts = defaultdict(int)
    for item in outcomes.values():
        outcome_counts[item["outcome_group"]] += 1
    results = summarize(cached_agg, initial_agg, dict(outcome_counts))
    write_csv(OUTPUT / "CACHED_90_STATE_ROWS.csv", cached_rows)
    write_csv(OUTPUT / "INITIAL_400_STATE_ROWS.csv", initial_rows)
    write_csv(OUTPUT / "CACHED_90_AGGREGATE.csv", cached_agg)
    write_csv(OUTPUT / "INITIAL_400_AGGREGATE.csv", initial_agg)
    write_csv(OUTPUT / "INITIAL_400_OUTCOME_AGGREGATE.csv", initial_outcome_agg)
    (OUTPUT / "RESULTS.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    figure_replacements(cached_payload, initial_payload)
    figure_scan(cached_agg, initial_agg)
    figure_outcome(initial_outcome_agg)
    write_report(results)


if __name__ == "__main__":
    main()
