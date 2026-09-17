#!/usr/bin/env python3
"""Build a source-grounded audit of the Prompt-Attention L11 layer choice."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


REPO = Path(__file__).resolve().parents[2]
LAYER_ARTIFACT = REPO / "artifacts/prompt_attn_layer_selection_v1"
PROMPT_V1_ARTIFACT = REPO / "artifacts/prompt_attn_shr_v1"
FAILURE_ARTIFACT = REPO / "artifacts/prompt_attn_shr_failure_diagnosis_v1"
WEEKLY_REPORT = REPO / "reports/weekly_report_2026-09-05_09-11.md"
OUTPUT = REPO / "artifacts/l11_layer_selection_audit"

TASK_LABELS = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
    "google_robot_place_apple_in_closed_top_drawer": "place_apple",
    "widowx_carrot_on_plate": "carrot_on_plate",
    "widowx_put_eggplant_in_basket": "put_eggplant",
    "widowx_spoon_on_towel": "spoon_on_towel",
    "widowx_stack_cube": "stack_cube",
}

CORE_TASKS = [
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
]

CANONICAL_0_99 = {
    "vanilla": {
        "google_robot_open_drawer": 27,
        "google_robot_close_drawer": 56,
        "google_robot_pick_coke_can": 23,
        "google_robot_move_near": 62,
        "google_robot_place_apple_in_closed_top_drawer": 0,
        "widowx_carrot_on_plate": 5,
        "widowx_put_eggplant_in_basket": 0,
        "widowx_spoon_on_towel": 0,
        "widowx_stack_cube": 0,
    },
    "original_shr": {
        "google_robot_open_drawer": 46,
        "google_robot_close_drawer": 78,
        "google_robot_pick_coke_can": 41,
        "google_robot_move_near": 67,
        "google_robot_place_apple_in_closed_top_drawer": 0,
        "widowx_carrot_on_plate": 3,
        "widowx_put_eggplant_in_basket": 0,
        "widowx_spoon_on_towel": 0,
        "widowx_stack_cube": 0,
    },
}


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def layers_label(value: str | list[int]) -> str:
    layers = ast.literal_eval(value) if isinstance(value, str) else value
    if len(layers) == 1:
        return f"L{layers[0]}"
    if layers == list(range(16, 32)):
        return "Avg L16-31"
    return "+".join(f"L{layer}" for layer in layers)


def summary_successes(task: str, arm: str) -> int | None:
    roots = [LAYER_ARTIFACT / "closed_loop", LAYER_ARTIFACT / "closed_loop_remaining6"]
    files = []
    for root in roots:
        files.extend((root / "episodes" / task / arm).glob("episode_*_summary.json"))
    by_seed = {}
    for path in files:
        item = load_json(path)
        seed = int(item["seed"])
        if 0 <= seed <= 99:
            by_seed[seed] = bool(item["success"])
    if len(by_seed) != 100:
        return None
    return sum(by_seed.values())


def stable_top_mask(scores: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[order[:count]] = True
    return mask


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def state_files() -> list[Path]:
    return sorted((LAYER_ARTIFACT / "states").glob("**/*.npz"))


def compute_overlap() -> tuple[np.ndarray, np.ndarray, float, float, int]:
    totals = np.zeros((32, 32), dtype=float)
    l11_v1 = []
    l11_pair = []
    files = state_files()
    for path in files:
        with np.load(path) as data:
            attention = data["original"].astype(float)
            count = int(np.asarray(data["standard_mask"]).astype(bool).sum())
        masks = np.stack([stable_top_mask(attention[layer], count) for layer in range(32)])
        for left in range(32):
            for right in range(left, 32):
                value = jaccard(masks[left], masks[right])
                totals[left, right] += value
                totals[right, left] += value if right != left else 0.0
        v1_mask = stable_top_mask(attention[16:32].mean(axis=0), count)
        pair_mask = stable_top_mask(attention[[11, 14]].mean(axis=0), count)
        l11_v1.append(jaccard(masks[11], v1_mask))
        l11_pair.append(jaccard(masks[11], pair_mask))
    return totals / len(files), totals[11] / len(files), float(np.mean(l11_v1)), float(np.mean(l11_pair)), len(files)


def closed_loop_counts() -> dict[str, dict[str, int]]:
    counts = {"prompt_v1": {}, "prompt_single": {}, "prompt_sparse": {}}
    formal = load_json(LAYER_ARTIFACT / "closed_loop/FINAL_RESULTS.json")["summaries"]
    for arm in counts:
        for task in TASK_LABELS:
            if task in formal and arm in formal[task]:
                value = int(formal[task][arm]["successes"])
            else:
                value = summary_successes(task, arm)
            if value is not None:
                counts[arm][task] = value
    return counts


def make_evidence_table(
    singles: list[dict],
    pairs: list[dict],
    validation: dict,
    counts: dict[str, dict[str, int]],
) -> None:
    columns = [
        "evidence_stage", "scope", "strategy", "layers", "task", "successes", "episodes",
        "success_rate", "selection_score", "target_response", "target_mask_response",
        "correct_response_fraction", "positive_task_count", "synonym_mask_jaccard",
        "mean_components", "mean_isolated_ratio", "mean_largest_component_ratio",
        "source_path", "notes",
    ]
    rows = []
    metric_columns = [
        "selection_score", "target_response", "target_mask_response", "correct_response_fraction",
        "positive_task_count", "synonym_mask_jaccard", "mean_components", "mean_isolated_ratio",
        "mean_largest_component_ratio",
    ]
    for stage, source_rows, source in [
        ("offline_exploration_single", singles, "artifacts/prompt_attn_layer_selection_v1/exploration_per_layer.csv"),
        ("offline_exploration_pair", pairs, "artifacts/prompt_attn_layer_selection_v1/exploration_candidate_pairs.csv"),
    ]:
        for item in source_rows:
            row = {column: "" for column in columns}
            row.update({"evidence_stage": stage, "scope": "20 episodes / 60 states", "strategy": layers_label(item["layers"]), "layers": item["layers"], "source_path": source})
            for column in metric_columns:
                row[column] = item.get(column, "")
            rows.append(row)
    for strategy, item in validation["results"].items():
        row = {column: "" for column in columns}
        row.update({
            "evidence_stage": "offline_frozen_validation",
            "scope": "10 held-out episodes / 30 states",
            "strategy": {"prompt_single": "L11", "prompt_sparse": "L11+L14", "prompt_v1": "Avg L16-31"}[strategy],
            "layers": json.dumps(item["layers"]),
            "source_path": "artifacts/prompt_attn_layer_selection_v1/VALIDATION_LAYER_RESULTS.json",
            "notes": "Candidates and score rule were frozen before this split was read.",
        })
        for column in metric_columns:
            row[column] = item.get(column, "")
        rows.append(row)
    strategies = {
        "vanilla": ("Vanilla", CANONICAL_0_99["vanilla"], "reports/weekly_report_2026-09-05_09-11.md", "Canonical prior 9-task baseline, seeds 0-99."),
        "original_shr": ("Original SHR", CANONICAL_0_99["original_shr"], "reports/weekly_report_2026-09-05_09-11.md", "Canonical prior 9-task SHR baseline, seeds 0-99."),
        "prompt_v1": ("Avg L16-31", counts["prompt_v1"], "artifacts/prompt_attn_layer_selection_v1/closed_loop*/episodes", "Matched Prompt-Attn arm; lambda=0.5."),
        "prompt_single": ("L11", counts["prompt_single"], "artifacts/prompt_attn_layer_selection_v1/closed_loop*/episodes", "Matched Prompt-Attn arm; lambda=0.5."),
        "prompt_sparse": ("L11+L14", counts["prompt_sparse"], "artifacts/prompt_attn_layer_selection_v1/closed_loop*/episodes", "Matched Prompt-Attn arm; lambda=0.5."),
    }
    for _, (label, task_counts, source, notes) in strategies.items():
        for task, successes in task_counts.items():
            row = {column: "" for column in columns}
            row.update({
                "evidence_stage": "closed_loop_seed_0_99", "scope": "same task-seed; 100 episodes",
                "strategy": label, "task": TASK_LABELS[task], "successes": successes, "episodes": 100,
                "success_rate": successes / 100.0, "source_path": source, "notes": notes,
            })
            rows.append(row)
        if task_counts:
            row = {column: "" for column in columns}
            successes = sum(task_counts.values())
            episodes = 100 * len(task_counts)
            row.update({
                "evidence_stage": "closed_loop_seed_0_99", "scope": f"{len(task_counts)} tasks",
                "strategy": label, "task": "OVERALL", "successes": successes, "episodes": episodes,
                "success_rate": successes / episodes, "source_path": source, "notes": notes,
            })
            rows.append(row)
    with (OUTPUT / "EVIDENCE_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def figure_layer_comparison(singles: list[dict], validation: dict, counts: dict[str, dict[str, int]]) -> None:
    layers = np.array([ast.literal_eval(item["layers"])[0] for item in singles])
    scores = np.array([float(item["selection_score"]) for item in singles])
    responses = np.array([float(item["target_response"]) for item in singles])
    components = np.array([float(item["mean_components"]) for item in singles])
    candidates = {7, 8, 11, 14}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(layers, scores, color="#315b7d", marker="o", markersize=3)
    for layer in candidates:
        ax.scatter(layer, scores[layer], s=80, color="#e39d25" if layer != 11 else "#c93434", zorder=3)
        ax.annotate(f"L{layer}", (layer, scores[layer]), xytext=(3, 7), textcoords="offset points")
    ax.set(title="Offline single-layer selection score (all 32 layers)", xlabel="Layer index", ylabel="Selection score", xticks=range(0, 32, 2))
    ax.grid(alpha=.25)

    ax = axes[0, 1]
    ax.plot(layers, responses, color="#498c6f", marker="o", markersize=3, label="target response")
    ax2 = ax.twinx()
    ax2.plot(layers, components, color="#8b6bb1", alpha=.65, label="mask components")
    ax.axvline(11, color="#c93434", linestyle="--", linewidth=1.5)
    ax.set(title="Target response vs. mask fragmentation", xlabel="Layer index", ylabel="Target response", xticks=range(0, 32, 2))
    ax2.set_ylabel("Connected components (lower is less fragmented)")
    ax.grid(alpha=.25)

    ax = axes[1, 0]
    labels = ["Avg L16-31", "L11", "L11+L14"]
    keys = ["prompt_v1", "prompt_single", "prompt_sparse"]
    vals = [validation["results"][key] for key in keys]
    x = np.arange(3)
    ax.bar(x - .2, [value["target_response"] for value in vals], width=.4, label="target response", color="#3c8d70")
    ax.set_ylabel("Target response")
    ax2 = ax.twinx()
    ax2.bar(x + .2, [value["mean_components"] for value in vals], width=.4, label="components", color="#a585c4")
    ax2.set_ylabel("Mask components")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Frozen validation: mechanism diagnostics")
    handles = [ax.patches[0], ax2.patches[0]]
    ax.legend(handles, ["target response", "components"], loc="upper left")

    ax = axes[1, 1]
    core3 = ["google_robot_open_drawer", "google_robot_pick_coke_can", "google_robot_move_near"]
    labels = ["Vanilla", "Standard SHR", "Avg L16-31", "L11", "L11+L14"]
    values = [112, 148, sum(counts["prompt_v1"][task] for task in core3), sum(counts["prompt_single"][task] for task in core3), sum(counts["prompt_sparse"][task] for task in core3)]
    colors = ["#969696", "#4b78a8", "#d18b47", "#c93434", "#8a65aa"]
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value}/300", ha="center", va="bottom", fontsize=9)
    ax.set(title="Closed-loop decision evidence (3 core tasks, seeds 0-99)", ylabel="Successful episodes", ylim=(0, 180))
    ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Why L11 was selected: exhaustive offline scan, then closed-loop confirmation", fontsize=16)
    fig.savefig(OUTPUT / "figure_layer_comparison.png", dpi=180)
    plt.close(fig)


def normalized_heatmap(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1).reshape(16, 16)


def figure_attention_heatmaps() -> None:
    representative = [
        ("open", LAYER_ARTIFACT / "states/google_robot_open_drawer/seed_000/step_011.npz"),
        ("pick", LAYER_ARTIFACT / "states/google_robot_pick_coke_can/seed_003/step_008.npz"),
        ("move", LAYER_ARTIFACT / "states/google_robot_move_near/seed_000/step_008.npz"),
    ]
    with np.load(representative[0][1]) as data:
        full_attention = data["original"].astype(float)
    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    grid = GridSpec(7, 8, figure=fig, height_ratios=[1, 1, 1, 1, 1.35, 1.35, 1.35])
    for layer in range(32):
        row, column = divmod(layer, 8)
        ax = fig.add_subplot(grid[row, column])
        ax.imshow(normalized_heatmap(full_attention[layer]), cmap="magma", interpolation="nearest")
        ax.set_title(f"L{layer}", fontsize=9, color="#c93434" if layer == 11 else "black", fontweight="bold" if layer == 11 else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        if layer == 11:
            for spine in ax.spines.values():
                spine.set_color("#c93434"); spine.set_linewidth(3)
    columns = [(None, "RGB"), (3, "L3"), (7, "L7"), (11, "L11"), (14, "L14"), (20, "L20"), (31, "L31"), ("avg", "Avg L16-31")]
    for row, (task, path) in enumerate(representative, start=4):
        with np.load(path) as data:
            attention = data["original"].astype(float)
            image = data["image"]
        for column, (layer, title) in enumerate(columns):
            ax = fig.add_subplot(grid[row, column])
            if layer is None:
                ax.imshow(image)
            else:
                values = attention[16:32].mean(axis=0) if layer == "avg" else attention[layer]
                ax.imshow(normalized_heatmap(values), cmap="magma", interpolation="nearest")
            if row == 4:
                ax.set_title(title, fontsize=10, color="#c93434" if layer == 11 else "black", fontweight="bold" if layer == 11 else "normal")
            if column == 0:
                ax.set_ylabel(task, fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            if layer == 11:
                for spine in ax.spines.values():
                    spine.set_color("#c93434"); spine.set_linewidth(3)
    fig.suptitle("Prompt-to-visual attention evidence: all layers plus representative target-switch states\n(each heatmap is independently percentile-normalized; appearance is diagnostic, not the final selection criterion)", fontsize=16)
    fig.savefig(OUTPUT / "figure_attention_heatmaps.png", dpi=170)
    plt.close(fig)


def figure_overlap_heatmap(overlap: np.ndarray, l11_overlap: np.ndarray, l11_v1: float, l11_pair: float, state_count: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1]})
    im = axes[0].imshow(overlap, vmin=0, vmax=1, cmap="viridis", origin="lower")
    axes[0].axhline(11, color="white", linewidth=1)
    axes[0].axvline(11, color="white", linewidth=1)
    axes[0].set(title=f"Mean top-m mask Jaccard across {state_count} cached states", xlabel="Layer", ylabel="Layer", xticks=range(0, 32, 2), yticks=range(0, 32, 2))
    fig.colorbar(im, ax=axes[0], fraction=.046)
    colors = ["#c93434" if layer == 11 else "#5586a4" for layer in range(32)]
    axes[1].bar(range(32), l11_overlap, color=colors)
    axes[1].axhline(l11_v1, color="#d18b47", linestyle="--", label=f"L11 vs Avg L16-31 mask: {l11_v1:.3f}")
    axes[1].axhline(l11_pair, color="#8a65aa", linestyle=":", linewidth=2, label=f"L11 vs L11+L14 mask: {l11_pair:.3f}")
    axes[1].set(title="L11 overlap with every single-layer mask", xlabel="Other layer", ylabel="Mean Jaccard", xticks=range(0, 32, 2), ylim=(0, 1.03))
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Is L11 mask unique? It is related to neighboring layers, but not equivalent to the late-layer average", fontsize=15)
    fig.savefig(OUTPUT / "figure_overlap_heatmap.png", dpi=180)
    plt.close(fig)


def figure_taskwise(counts: dict[str, dict[str, int]]) -> None:
    labels = [TASK_LABELS[task] for task in CORE_TASKS]
    series = {
        "Vanilla": [CANONICAL_0_99["vanilla"][task] for task in CORE_TASKS],
        "Original SHR": [CANONICAL_0_99["original_shr"][task] for task in CORE_TASKS],
        "Avg L16-31": [counts["prompt_v1"][task] for task in CORE_TASKS],
        "L11": [counts["prompt_single"][task] for task in CORE_TASKS],
        "L11+L14": [counts["prompt_sparse"][task] for task in CORE_TASKS],
    }
    colors = ["#969696", "#4b78a8", "#d18b47", "#c93434", "#8a65aa"]
    x = np.arange(len(labels)); width = .16
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    for index, ((name, values), color) in enumerate(zip(series.items(), colors)):
        bars = ax.bar(x + (index - 2) * width, values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, str(value), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Successes / 100 seeds")
    ax.set_ylim(0, 92)
    ax.set_title("Taskwise closed-loop comparison on the four core tasks (seeds 0-99)")
    ax.legend(ncol=5, loc="upper center")
    ax.grid(axis="y", alpha=.25)
    fig.savefig(OUTPUT / "figure_taskwise_comparison.png", dpi=180)
    plt.close(fig)


def write_summary(singles: list[dict], pairs: list[dict], validation: dict, counts: dict[str, dict[str, int]], l11_v1: float, l11_pair: float, state_count: int) -> None:
    single_by_layer = {ast.literal_eval(item["layers"])[0]: item for item in singles}
    l11 = single_by_layer[11]
    pair_1114 = next(item for item in pairs if ast.literal_eval(item["layers"]) == [11, 14])
    core3 = ["google_robot_open_drawer", "google_robot_pick_coke_can", "google_robot_move_near"]
    core_totals = {arm: sum(counts[arm][task] for task in core3) for arm in counts}
    all9_totals = {arm: sum(values.values()) for arm, values in counts.items()}
    summary = f"""# OpenVLA Prompt-Attention L11 选层审计

生成日期：2026-09-14  
审计范围：当前代码仓库、`artifacts/`、周报、冻结配置、离线缓存和闭环 episode summary。所有图表均由现存数据重建；未找到的数据明确标注为未找到。

## 一句话结论

L11 **不是先看一张热力图拍脑袋决定的，也不是把 L0–L31 全部逐层跑了闭环后取最大值**。实际链条是：先用 **L16–L31 简单平均**做 Prompt-v1，但三任务闭环只有 **109/300**，弱于 Standard SHR 的 **148/300**；随后从缓存的 90 个 state 中对 **L0–L31 全部做离线逐层诊断**，按“目标切换响应 + mask 响应 + 正确方向比例 + 同义指令稳定性”冻结出 L11、L7、L14、L8，并把 **L11**与离线分数最高的组合 **L11+L14**送入闭环；最终 L11 得到 **156/300**，高于 L11+L14 的 **136/300**和 Avg L16–31 的 **109/300**，因此固定为 L11。

换句话说：**attention/目标切换诊断负责提出候选，matched closed-loop success 负责最终定案。** 热力图只是机制证据，正式报告明确写道最终结论不由“看起来更好”决定。

![层选择总览](figure_layer_comparison.png)

## 证据链

### 1. 起点：后半层平均 Prompt-v1 先失败

- 配置：`artifacts/prompt_attn_shr_v1/CONFIG_LOCK.json`。使用 L16–L31 等权平均、完整 instruction query、与 SHR 相同的 selected-token 数量、相同 harmonic reconstruction、`lambda=0.5`。
- 代码实现：`research/semantic_token_cd/prompt_attn_shr_policy.py:430`提取全部 32 层 attention；`research/semantic_token_cd/prompt_attn_shr_policy.py:482`对配置指定层做算术平均。
- 三任务闭环：Vanilla **112/300**，Standard SHR **148/300**，Prompt-v1 **109/300**，Random-SHR **125/300**。Prompt-v1 相对 SHR 为 Rescue 25、Harm 64、净变化 **-39**。
- 失败诊断：`artifacts/prompt_attn_shr_failure_diagnosis_v1/AGGREGATE_DIAGNOSTICS.json`显示 Avg L16–31 mask 平均 **12.978**个连通分量、孤立 token 比例 **0.256**、最大连通块比例 **0.283**；Standard SHR 分别为 **3.056 / 0.031 / 0.753**。两者动作 residual cosine 只有 **0.297**。
- 当时周报原文：**“Attention 不是不能用，而是把很多层直接平均，会把少数有用层的任务信号冲淡。”** 来源：`reports/weekly_report_2026-09-05_09-11.md:151`。

### 2. 逐层离线初筛：L0–L31 都测了，但测的是缓存 state，不是闭环

- `research/semantic_token_cd/analyze_prompt_attn_layers.py:20`定义 `LAYERS=tuple(range(32))`，确实逐一评估了全部 32 层。
- 数据划分：`artifacts/prompt_attn_layer_selection_v1/SPLIT_AND_CONTROLS.json`固定 30 个 episode、每个 3 个 state，共 90 states；20 episodes/60 states 用于 exploration，10 episodes/30 states 完全留作 frozen validation。
- 评分规则：`research/semantic_token_cd/analyze_prompt_attn_layers.py:161`：55% target attention response percentile、20% target-mask response、15% correct-both fraction、10% synonym top-m Jaccard；且至少两个任务 target response 为正。
- L11 是符合资格的单层第一名：selection score **{float(l11['selection_score']):.6f}**；target response **{float(l11['target_response']):.6f}**；target-mask response **{float(l11['target_mask_response']):.6f}**；同义指令 mask Jaccard **{float(l11['synonym_mask_jaccard']):.3f}**。
- 冻结出的单层候选顺序是 **L11、L7、L14、L8**，见 `artifacts/prompt_attn_layer_selection_v1/CANDIDATES_LOCK.json`。
- 邻近层并非完全没信号：L10 score **{float(single_by_layer[10]['selection_score']):.3f}**，L12 score **{float(single_by_layer[12]['selection_score']):.3f}**；但都低于 L11 的 **{float(l11['selection_score']):.3f}**。L14 是 shortlist 第三，而不是单层冠军。

### 3. 多层组合也测了：离线组合甚至一度胜过 L11

- `research/semantic_token_cd/analyze_prompt_attn_layers.py:168`先取前四个单层候选，然后评估它们的全部 6 个两两组合：L11+L7、L11+L14、L11+L8、L7+L14、L7+L8、L14+L8。
- exploration 上 L11+L14 score 为 **{float(pair_1114['selection_score']):.6f}**，实际上高于 L11 的 **{float(l11['selection_score']):.6f}**，所以它被诚实地选为 `prompt_sparse` 候选。
- 没有找到 L11+L12 的正式实验；也没有找到 L7、L8、L14 单独跑 100-seed 闭环的结果。不能把“全 32 层离线扫描”写成“全 32 层闭环扫描”。

### 4. 冻结验证：L11/L11+L14 都能响应目标切换，晚层平均明显被稀释

- 候选与评分规则先写入 `CANDIDATES_LOCK.json`，再读取 held-out validation。文件时间也符合协议：候选锁 2026-09-06 09:30:52，validation 结果 09:31:09。
- L11 validation：target response **{validation['results']['prompt_single']['target_response']:.4f}**，正确响应比例 **100%**，mask components **{validation['results']['prompt_single']['mean_components']:.2f}**，最大连通块 **{validation['results']['prompt_single']['mean_largest_component_ratio']:.3f}**。
- L11+L14：target response **{validation['results']['prompt_sparse']['target_response']:.4f}**，正确响应比例 **100%**，components **{validation['results']['prompt_sparse']['mean_components']:.2f}**。
- Avg L16–31：target response 仅 **{validation['results']['prompt_v1']['target_response']:.4f}**，正确响应比例 **66.7%**，components **{validation['results']['prompt_v1']['mean_components']:.2f}**，最大连通块仅 **{validation['results']['prompt_v1']['mean_largest_component_ratio']:.3f}**。
- 因此 validation 中 L11 的 target response 约为 Avg L16–31 的 **{validation['results']['prompt_single']['target_response']/validation['results']['prompt_v1']['target_response']:.1f} 倍**，且 mask 更集中。这里支持“平均稀释信号”，但还不是最终定案。

![代表性 attention 热力图](figure_attention_heatmaps.png)

### 5. 最终定案：matched closed-loop 明确选择 L11，而不是离线分更高的 L11+L14

- 闭环配置：`artifacts/prompt_attn_layer_selection_v1/closed_loop/CONFIG_LOCK.json`；L11 与 L11+L14 均保持 `lambda=0.5`、相同 matched token coverage、相同重建方式和 canonical snapshot。
- 三任务 seeds 0–99：Avg L16–31 **{core_totals['prompt_v1']}/300**，L11 **{core_totals['prompt_single']}/300**，L11+L14 **{core_totals['prompt_sparse']}/300**。L11 比平均多 **{core_totals['prompt_single']-core_totals['prompt_v1']}** 次成功，比组合多 **{core_totals['prompt_single']-core_totals['prompt_sparse']}** 次成功。
- 配对结果见 `artifacts/prompt_attn_layer_selection_v1/closed_loop/report.md`：L11 vs Avg L16–31 为 Rescue 76 / Harm 29 / 净 **+47**；L11+L14 vs Avg 为 Rescue 61 / Harm 34 / 净 **+27**；L11 vs Standard SHR 为 Rescue 51 / Harm 43 / 净 **+8**。L11+L14 vs SHR 反而净 **-12**。
- 这一步推翻了“离线 selection score 最高就直接采用 L11+L14”的可能性。最终选择 L11 的最关键证据就是 **同 seed matched 闭环成功率与 Rescue/Harm 配对结果**。

### 6. 扩展到 9 任务：L11 的优势不是只存在于三任务筛选

- 周报统一对齐 seeds 0–99、900/900 初始 state/RGB hash 一致：Vanilla **173/900**，原始 SHR **235/900**，L11 **243/900**。
- L11 vs Vanilla：Rescue 99 / Harm 29 / 净 **+70**；L11 vs 原始 SHR：Rescue 65 / Harm 57 / 净 **+8**。
- 主要提升来自 open、close、pick；move_near 比原始 SHR 少 5，其余低成功率任务判别力有限。因此“L11 最好”是当时这套 9-task aggregate 下的选择，不代表每个任务都由 L11 单独支配。
- 当前 raw summaries 还能恢复 Avg L16–31、L11、L11+L14 的 9-task totals，分别为 **{all9_totals['prompt_v1']}/900、{all9_totals['prompt_single']}/900、{all9_totals['prompt_sparse']}/900**。其中 L11 仍最高。

![逐任务闭环对比](figure_taskwise_comparison.png)

## L11 是否“特殊”

对缓存的 {state_count} 个 state，按照每个 state 的 matched token 数量重建所有单层 top-m mask。L11 与 Avg L16–31 mask 的平均 Jaccard 为 **{l11_v1:.3f}**，与 L11+L14 为 **{l11_pair:.3f}**。这说明 L11 不是和所有层完全无关的孤岛，但它也绝不等价于晚层平均；加入 L14 会实质改变选区，并在闭环中损失 20/300 成功。

![层间 mask overlap](figure_overlap_heatmap.png)

## 明确回答六个问题

1. **是不是做过逐层实验？** 是，但准确说是 L0–L31 的离线 state-level attention/mask 诊断；没有证据表明 32 个单层都各跑过完整闭环。
2. **是不是做过多层平均 vs 单层？** 是。正式比较了 Avg L16–31、L11，以及 L11+L14；Prompt-v1 的平均策略先在闭环失败，L11 后来显著胜出。
3. **是不是做过 L11+其他层？** 是。离线做过前四候选的 6 个两层组合；正式闭环做过 L11+L14。没有找到 L11+L12 闭环证据。
4. **最终选 L11 最关键证据是什么？** 三任务 100 seeds 的 matched closed-loop：L11 156/300，L11+L14 136/300，Avg L16–31 109/300；同时有同 seed Rescue/Harm 配对支持。
5. **候选来自可视化还是闭环直接选优？** 两者共同决定，但职责不同：目标切换/attention/mask 诊断提出 L11 候选，闭环成功率最终定案。不是只看热力图。
6. **有没有“平均多层稀释信号”的证据？** 有。Avg L16–31 的 validation target response 只有 {validation['results']['prompt_v1']['target_response']:.4f}，L11 为 {validation['results']['prompt_single']['target_response']:.4f}；平均 mask 更碎（12.63 vs 7.17 components），闭环更差（109/300 vs 156/300）。加入 L14 同样从 156/300 降到 136/300。不过应写成“在已测试的简单等权平均中出现稀释”，不能推广为所有多层融合必然失败。

## 重要边界与缺失数据

- 没有找到“每个 L0–L31 单层各 100 seeds 闭环”的证据；全层曲线是离线评分曲线。
- 没有找到 all-layer average（L0–L31）正式结果；找到的是 later-layer average（L16–L31）。
- 没有找到 L11+L12 正式结果。
- L7/L8/L14 是离线 shortlist，但未找到它们的独立 100-seed 闭环结果。
- first-3-task 筛选报告里的 `Standard SHR` 是当轮重跑参考（open 47、pick 41、move 60）；9-task 周报里的“原始 SHR”来自此前 canonical 9×300 数据切片（open 46、pick 41、move 67）。本报告不混用这两套基线：三任务定案引用当轮 matched report，四任务/九任务图引用周报 canonical baseline。
- attention 热力图逐图独立归一化，只能解释空间结构，不能用于跨层比较绝对 attention mass。

## 可复现文件

- 汇总表：`EVIDENCE_TABLE.csv`
- 作图/重建脚本：`research/semantic_token_cd/build_l11_layer_selection_audit.py`
- 原始逐层表：`artifacts/prompt_attn_layer_selection_v1/exploration_per_layer.csv`
- 原始组合表：`artifacts/prompt_attn_layer_selection_v1/exploration_candidate_pairs.csv`
- 候选冻结：`artifacts/prompt_attn_layer_selection_v1/CANDIDATES_LOCK.json`
- 冻结验证：`artifacts/prompt_attn_layer_selection_v1/VALIDATION_LAYER_RESULTS.json`
- 闭环报告：`artifacts/prompt_attn_layer_selection_v1/closed_loop/report.md`
- 周报总结：`reports/weekly_report_2026-09-05_09-11.md`
"""
    (OUTPUT / "SUMMARY.md").write_text(summary)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    singles = load_csv(LAYER_ARTIFACT / "exploration_per_layer.csv")
    pairs = load_csv(LAYER_ARTIFACT / "exploration_candidate_pairs.csv")
    validation = load_json(LAYER_ARTIFACT / "VALIDATION_LAYER_RESULTS.json")
    counts = closed_loop_counts()
    overlap, l11_overlap, l11_v1, l11_pair, state_count = compute_overlap()
    make_evidence_table(singles, pairs, validation, counts)
    figure_layer_comparison(singles, validation, counts)
    figure_attention_heatmaps()
    figure_overlap_heatmap(overlap, l11_overlap, l11_v1, l11_pair, state_count)
    figure_taskwise(counts)
    write_summary(singles, pairs, validation, counts, l11_v1, l11_pair, state_count)
    audit = {
        "generated": "2026-09-14",
        "state_count_for_overlap": state_count,
        "l11_vs_avg_l16_31_mask_jaccard": l11_v1,
        "l11_vs_l11_l14_mask_jaccard": l11_pair,
        "closed_loop_successes_seed_0_99": counts,
    }
    (OUTPUT / "AUDIT_DATA.json").write_text(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
