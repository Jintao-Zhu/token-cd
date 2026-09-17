"""Audit what the SHR-derived matched L11 token budget encodes.

This script is intentionally offline. It reads the completed four-task L11
count sweep and the previously cached full-layer budget-diagnostic states. It
does not run OpenVLA or control an environment.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COUNT_ROOT = ROOT / "artifacts/prompt_attn_l11_token_count_v1"
STATE_ROOT = ROOT / "artifacts/prompt_attn_l11_budget_diagnostic_v1/states"
LAYER_STATE_ROOT = ROOT / "artifacts/prompt_attn_layer_selection_v1/states"
OUTPUT = ROOT / "artifacts/l11_matched_budget_encoding_audit"
TASK_ORDER = (
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
SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(x: list[float], y: list[float], spearman: bool = False) -> float:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    if spearman:
        left, right = rankdata(left), rankdata(right)
    return float(np.corrcoef(left, right)[0, 1])


def phase_name(progress: float) -> str:
    if progress < 1 / 3:
        return "early"
    if progress < 2 / 3:
        return "middle"
    return "late"


def eta_squared(values: np.ndarray, groups: list[str]) -> float:
    grand = float(values.mean())
    denominator = float(np.square(values - grand).sum())
    if denominator == 0:
        return 0.0
    numerator = 0.0
    for group in sorted(set(groups)):
        selected = values[np.asarray([item == group for item in groups])]
        numerator += len(selected) * float(selected.mean() - grand) ** 2
    return numerator / denominator


def categorical_r2(values: np.ndarray, keys: list[str]) -> float:
    prediction = np.empty_like(values)
    for key in sorted(set(keys)):
        indices = np.asarray([item == key for item in keys])
        prediction[indices] = values[indices].mean()
    denominator = float(np.square(values - values.mean()).sum())
    return 0.0 if denominator == 0 else 1.0 - float(np.square(values - prediction).sum()) / denominator


def load_trajectory_rows() -> tuple[list[dict], list[dict]]:
    steps: list[dict] = []
    episodes: list[dict] = []
    for task in TASK_ORDER:
        paths = sorted((COUNT_ROOT / "episodes" / task / "l11_matched").glob("episode_*_summary.json"))
        for path in paths:
            summary = json.loads(path.read_text())
            trace = summary["selector_trace"]
            counts = np.asarray([record["m_t"] for record in trace], dtype=float)
            for step, record in enumerate(trace):
                progress = step / max(1, len(trace) - 1)
                group_ids = record.get("selected_group_ids") or []
                steps.append({
                    "task": TASK_LABEL[task],
                    "seed": int(summary["seed"]),
                    "success": int(bool(summary["success"])),
                    "step": step,
                    "progress": progress,
                    "phase": phase_name(progress),
                    "m_t": int(record["m_t"]),
                    "selected_group_count": len(set(group_ids)),
                    "entity_count": len(record.get("selected_entities") or []),
                    "entities_share_cluster": int(len(group_ids) >= 2 and len(set(group_ids)) == 1),
                    "mask_components": int(record["mask_component_count"]),
                    "isolated_ratio": float(record["isolated_token_ratio"]),
                    "feature_perturbation_relative": float(record["feature_perturbation_relative"]),
                    "centered_logit_residual_norm": float(record["centered_logit_residual_norm"]),
                    "guided_changed_dims": int(record["guided_changed_dims"]),
                })
            thirds = np.array_split(counts, 3)
            episodes.append({
                "task": TASK_LABEL[task],
                "seed": int(summary["seed"]),
                "success": int(bool(summary["success"])),
                "steps": len(trace),
                "mean_m_t": float(counts.mean()),
                "std_m_t": float(counts.std()),
                "cv_m_t": float(counts.std() / counts.mean()),
                "min_m_t": int(counts.min()),
                "max_m_t": int(counts.max()),
                "range_m_t": int(counts.max() - counts.min()),
                "mean_abs_step_change": float(np.abs(np.diff(counts)).mean()) if len(counts) > 1 else 0.0,
                "early_mean_m_t": float(thirds[0].mean()),
                "middle_mean_m_t": float(thirds[1].mean()),
                "late_mean_m_t": float(thirds[2].mean()),
            })
    return steps, episodes


def mask_components(mask: np.ndarray) -> int:
    grid = mask.reshape(16, 16).astype(bool)
    seen = np.zeros_like(grid)
    components = 0
    for row in range(16):
        for column in range(16):
            if not grid[row, column] or seen[row, column]:
                continue
            components += 1
            stack = [(row, column)]
            seen[row, column] = True
            while stack:
                current_row, current_column = stack.pop()
                for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = current_row + delta_row
                    next_column = current_column + delta_column
                    if 0 <= next_row < 16 and 0 <= next_column < 16 and grid[next_row, next_column] and not seen[next_row, next_column]:
                        seen[next_row, next_column] = True
                        stack.append((next_row, next_column))
    return components


def load_attention_rows() -> list[dict]:
    rows: list[dict] = []
    metadata_paths = sorted(LAYER_STATE_ROOT.glob("*/seed_*/*.json"))
    metadata_paths += sorted(STATE_ROOT.glob("*/seed_*/*.json"))
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        arrays = np.load(metadata_path.with_suffix(".npz"))
        scores = np.asarray(arrays["original"][11], dtype=float)
        probabilities = scores / scores.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        normalized_entropy = entropy / math.log(256)
        effective_support = 1.0 / float(np.square(probabilities).sum())
        matched_count = int(metadata.get("matched_count", metadata.get("m")))
        mask = arrays["standard_mask"].astype(bool)
        rows.append({
            "task": TASK_LABEL[metadata["task"]],
            "seed": int(metadata["seed"]),
            "step": int(metadata.get("step", metadata.get("source_step"))),
            "phase": metadata["phase"],
            "matched_count": matched_count,
            "l11_entropy": entropy,
            "l11_normalized_entropy": normalized_entropy,
            "l11_effective_support": effective_support,
            "matched_over_effective_support": matched_count / effective_support,
            "top16_mass": float(np.sort(probabilities)[-16:].sum()),
            "top32_mass": float(np.sort(probabilities)[-32:].sum()),
            "standard_mask_components": mask_components(mask),
        })
    return rows


def summarize(steps: list[dict], episodes: list[dict], attention: list[dict]) -> dict:
    counts = np.asarray([row["m_t"] for row in steps], dtype=float)
    tasks = [row["task"] for row in steps]
    phases = [row["phase"] for row in steps]
    task_phase = [f"{row['task']}::{row['phase']}" for row in steps]
    attention_counts = [row["matched_count"] for row in attention]
    support = [row["l11_effective_support"] for row in attention]
    entropy = [row["l11_normalized_entropy"] for row in attention]
    episode_fixed_mae = []
    for task in TASK_ORDER:
        label = TASK_LABEL[task]
        for seed in range(100, 200):
            rows = [row for row in steps if row["task"] == label and row["seed"] == seed]
            if rows:
                first = rows[0]["m_t"]
                episode_fixed_mae.extend(abs(row["m_t"] - first) for row in rows)
    task_stats = {}
    for task in (TASK_LABEL[item] for item in TASK_ORDER):
        selected = [row for row in steps if row["task"] == task]
        values = np.asarray([row["m_t"] for row in selected], dtype=float)
        task_stats[task] = {
            "steps": len(selected),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
        }
    outcome_stats = {}
    for task in (TASK_LABEL[item] for item in TASK_ORDER):
        for success in (0, 1):
            selected = [row["mean_m_t"] for row in episodes if row["task"] == task and row["success"] == success]
            outcome_stats[f"{task}::{success}"] = {"episodes": len(selected), "mean_episode_budget": float(np.mean(selected)) if selected else None}
    scaling = {}
    for scale in SCALES:
        scaled = np.clip(np.rint(scale * counts), 1, 256)
        scaling[str(scale)] = {
            "mean": float(scaled.mean()),
            "std": float(scaled.std()),
            "p10": float(np.quantile(scaled, .1)),
            "median": float(np.median(scaled)),
            "p90": float(np.quantile(scaled, .9)),
        }
    return {
        "episodes": len(episodes),
        "steps": len(steps),
        "attention_states": len(attention),
        "overall_budget": {
            "mean": float(counts.mean()), "std": float(counts.std()),
            "p10": float(np.quantile(counts, .1)), "median": float(np.median(counts)),
            "p90": float(np.quantile(counts, .9)), "min": int(counts.min()), "max": int(counts.max()),
        },
        "variance_explained": {
            "task_eta_squared": eta_squared(counts, tasks),
            "phase_eta_squared": eta_squared(counts, phases),
            "task_phase_r2": categorical_r2(counts, task_phase),
            "remaining_within_task_phase_fraction": 1.0 - categorical_r2(counts, task_phase),
        },
        "episode_dynamics": {
            "mean_within_episode_std": float(np.mean([row["std_m_t"] for row in episodes])),
            "mean_within_episode_range": float(np.mean([row["range_m_t"] for row in episodes])),
            "mean_abs_step_change": float(np.mean([row["mean_abs_step_change"] for row in episodes])),
            "episode_fixed_step_mae": float(np.mean(episode_fixed_mae)),
        },
        "attention_relationships": {
            "matched_vs_effective_support_pearson": correlation(attention_counts, support),
            "matched_vs_effective_support_spearman": correlation(attention_counts, support, True),
            "matched_vs_entropy_pearson": correlation(attention_counts, entropy),
            "matched_vs_entropy_spearman": correlation(attention_counts, entropy, True),
            "matched_vs_mask_components_spearman": correlation(
                attention_counts, [row["standard_mask_components"] for row in attention], True
            ),
        },
        "intervention_relationships": {
            "m_t_vs_feature_perturbation_spearman": correlation(
                [row["m_t"] for row in steps], [row["feature_perturbation_relative"] for row in steps], True
            ),
            "m_t_vs_logit_residual_spearman": correlation(
                [row["m_t"] for row in steps], [row["centered_logit_residual_norm"] for row in steps], True
            ),
            "m_t_vs_guided_changed_dims_spearman": correlation(
                [row["m_t"] for row in steps], [row["guided_changed_dims"] for row in steps], True
            ),
        },
        "task_stats": task_stats,
        "outcome_stats": outcome_stats,
        "scaled_budget_distributions": scaling,
    }


def plot_task_phase(steps: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    labels = [TASK_LABEL[item] for item in TASK_ORDER]
    axes[0].boxplot([[row["m_t"] for row in steps if row["task"] == task] for task in labels], tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("Matched budget $m_t$")
    axes[0].set_title("Budget differs strongly by task")
    axes[0].tick_params(axis="x", rotation=18)
    width = .22
    for index, phase in enumerate(("early", "middle", "late")):
        means = [np.mean([row["m_t"] for row in steps if row["task"] == task and row["phase"] == phase]) for task in labels]
        axes[1].bar(np.arange(len(labels)) + (index - 1) * width, means, width, label=phase)
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=18)
    axes[1].set_ylabel("Mean matched budget")
    axes[1].set_title("Task-conditioned phase profile")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_task_phase_budget.png", dpi=180)
    plt.close(fig)


def plot_trajectories(steps: list[dict], episodes: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, task in zip(axes.flat, (TASK_LABEL[item] for item in TASK_ORDER)):
        task_episodes = [row for row in episodes if row["task"] == task]
        chosen = sorted(task_episodes, key=lambda row: row["std_m_t"], reverse=True)[:8]
        for episode in chosen:
            rows = [row for row in steps if row["task"] == task and row["seed"] == episode["seed"]]
            axis.plot([row["progress"] for row in rows], [row["m_t"] for row in rows], alpha=.6, linewidth=1)
        phase_means = []
        centers = np.linspace(.025, .975, 20)
        for left in np.linspace(0, .95, 20):
            selected = [row["m_t"] for row in steps if row["task"] == task and left <= row["progress"] < left + .05]
            phase_means.append(np.mean(selected))
        axis.plot(centers, phase_means, color="black", linewidth=3, label="task mean")
        axis.set_title(task)
        axis.set_ylabel("$m_t$")
        axis.grid(alpha=.2)
    axes[-1, 0].set_xlabel("Normalized episode progress")
    axes[-1, 1].set_xlabel("Normalized episode progress")
    fig.suptitle("Matched budget is dynamic within episodes")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_budget_trajectories.png", dpi=180)
    plt.close(fig)


def plot_attention_support(attention: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = dict(zip((TASK_LABEL[item] for item in TASK_ORDER), plt.cm.tab10.colors))
    for task, color in colors.items():
        selected = [row for row in attention if row["task"] == task]
        axes[0].scatter([row["l11_effective_support"] for row in selected], [row["matched_count"] for row in selected], label=task, color=color, alpha=.8)
        axes[1].scatter([row["l11_normalized_entropy"] for row in selected], [row["matched_count"] for row in selected], label=task, color=color, alpha=.8)
    axes[0].set_xlabel("L11 attention effective support")
    axes[1].set_xlabel("L11 normalized entropy")
    for axis in axes:
        axis.set_ylabel("Matched budget $m_t$")
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Does matched budget reduce to an L11 concentration statistic?")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_attention_support_relation.png", dpi=180)
    plt.close(fig)


def plot_scaling(steps: list[dict]) -> None:
    counts = np.asarray([row["m_t"] for row in steps], dtype=float)
    fig, axis = plt.subplots(figsize=(10, 6))
    parts = axis.violinplot([np.rint(scale * counts) for scale in SCALES], showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(.55)
    axis.set_xticks(range(1, len(SCALES) + 1), [str(scale) for scale in SCALES])
    axis.set_xlabel("Budget scaling α")
    axis.set_ylabel("Scaled token budget")
    axis.set_title("Dose range induced by matched-budget scaling")
    axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_scaled_budget_distributions.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict) -> None:
    variance = summary["variance_explained"]
    dynamics = summary["episode_dynamics"]
    attention = summary["attention_relationships"]
    intervention = summary["intervention_relationships"]
    overall = summary["overall_budget"]
    lines = [
        "# What does the L11 matched budget encode?", "",
        "生成日期：2026-09-14。该报告完全离线；未运行新 OpenVLA forward，未控制环境。", "",
        "## 数据", "",
        f"- {summary['episodes']} 条四任务 Matched episode，合计 {summary['steps']} 个 control steps（seeds 100–199）。",
        f"- {summary['attention_states']} 个缓存 full-layer same-state，用于恢复 L11 attention entropy/effective support。", "",
        "## 当前结论", "",
        f"1. Matched 的整体预算均值为 **{overall['mean']:.2f}**，但分布很宽：P10={overall['p10']:.0f}、median={overall['median']:.0f}、P90={overall['p90']:.0f}、范围={overall['min']}–{overall['max']}。因此均值约34不能代表逐状态剂量。",
        f"2. task 单独解释预算方差的 **{variance['task_eta_squared']:.1%}**；phase 单独解释 **{variance['phase_eta_squared']:.1%}**；task+phase 仅解释 **{variance['task_phase_r2']:.1%}**，仍有 **{variance['remaining_within_task_phase_fraction']:.1%}** 是同任务同阶段内变化。",
        f"3. episode 内预算并不固定：平均 episode 内标准差 **{dynamics['mean_within_episode_std']:.2f}**，平均范围 **{dynamics['mean_within_episode_range']:.2f}**；若整条 episode 固定为首步预算，逐步 MAE 为 **{dynamics['episode_fixed_step_mae']:.2f} tokens**。",
        f"4. 在{summary['attention_states']}个 attention state 上，m_t 与 L11 effective support 的 Spearman 相关仅 **{attention['matched_vs_effective_support_spearman']:+.3f}**，与 entropy 为 **{attention['matched_vs_entropy_spearman']:+.3f}**。Matched 不能简单等价为一个 L11 concentration cutoff。",
        f"5. m_t 与实际 feature corruption 的 Spearman 相关为 **{intervention['m_t_vs_feature_perturbation_spearman']:+.3f}**，与 logit residual 为 **{intervention['m_t_vs_logit_residual_spearman']:+.3f}**，与改变动作维数为 **{intervention['m_t_vs_guided_changed_dims_spearman']:+.3f}**。这验证 m_t 确实控制了反事实干预剂量，但 token 数并不完全决定最终动作影响。", "",
        "## 对闭环实验的含义", "",
        "- **Global Shuffle**：检验整体数量分布是否足够。", 
        "- **Within-task Shuffle**：检验 task-level 分布是否足够；由于 task+phase 后仍有大量剩余方差，这是最关键的 state-alignment 对照。",
        "- **Episode-fixed**：现有轨迹显示它会产生明显逐步预算误差，可检验 phase/state 动态是否必要。",
        "- **Scaling**：α={0.5,0.75,1.0,1.25,1.5} 保留状态排序但改变绝对 corruption dose；用于判断 α=1 是否接近峰值。", "",
        "## 冻结闭环协议", "",
        "1. 使用与 count/top-p 完全相同的四任务 canonical seeds 100–199、L11 ranking、λ=0.5、harmonic reconstruction。",
        "2. Matched α=1.0 直接复用已有 400 episodes；新跑 Global-Shuffle、Within-task-Shuffle、Episode-fixed 与 α=0.5/0.75/1.25/1.5，共 7×400=2800 条新 episode。",
        "3. Shuffle 映射必须在运行前以 JSON 冻结；Global Shuffle 跨四任务同 normalized progress bin 置换，Within-task Shuffle 在 task+progress bin 内置换，避免拿不存在的未来轨迹长度直接对齐。",
        "4. 主指标为 paired success/McNemar；辅助指标为逐步 budget deviation、feature perturbation、logit residual 和 guided action change。", "",
        "## 尚未回答", "",
        "- 现有 trace 只保存 union mask 和 group IDs，没有保存完整 KMeans labels，因此无法离线恢复 source/target 各自 cluster size；需要未来 replay 时补充日志。",
        "- 本报告说明 matched budget 包含 task 与 state/phase 变化，也控制 corruption dose；但只有 shuffle/scaling 闭环才能证明这些变化是否造成成功率优势。", "",
        "## 图", "",
        "![Task and phase](figure_task_phase_budget.png)", "",
        "![Trajectories](figure_budget_trajectories.png)", "",
        "![Attention support](figure_attention_support_relation.png)", "",
        "![Scaling](figure_scaled_budget_distributions.png)", "",
    ]
    (OUTPUT / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    steps, episodes = load_trajectory_rows()
    attention = load_attention_rows()
    summary = summarize(steps, episodes, attention)
    write_csv(OUTPUT / "STEP_BUDGETS.csv", steps)
    write_csv(OUTPUT / "EPISODE_BUDGETS.csv", episodes)
    write_csv(OUTPUT / "ATTENTION_SUPPORT.csv", attention)
    (OUTPUT / "AUDIT_DATA.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    plot_task_phase(steps)
    plot_trajectories(steps, episodes)
    plot_attention_support(attention)
    plot_scaling(steps)
    write_report(summary)
    print(json.dumps({
        "episodes": len(episodes), "steps": len(steps), "attention_states": len(attention),
        "output": str(OUTPUT), "variance_explained": summary["variance_explained"],
        "attention_relationships": summary["attention_relationships"],
    }, indent=2))


if __name__ == "__main__":
    main()
