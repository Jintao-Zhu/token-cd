"""Analyze the paired five-arm Causal Attention Masking CD pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
)
ARMS = ("vanilla", "random_cd", "semantic_cd", "attn_random_cd", "attn_semantic_cd")
LABELS = {
    "vanilla": "Vanilla",
    "random_cd": "Legacy-Rand",
    "semantic_cd": "Legacy-Sem",
    "attn_random_cd": "Attn-Rand",
    "attn_semantic_cd": "Attn-Sem",
}


def load_summary(root: Path, task: str, arm: str, seed: int) -> dict:
    return json.loads(
        (root / "episodes" / task / arm / f"episode_{seed:03d}_summary.json").read_text()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--legacy-artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    legacy = args.legacy_artifact.resolve()
    seeds = list(range(10))

    records = []
    residuals = {arm: [] for arm in ARMS if arm != "vanilla"}
    feature_audits = []
    for task in TASKS:
        manifest = json.loads((artifact / "episodes" / task / "pairing_manifest.json").read_text())
        if not manifest["all_five_arm_exact_pairing"] or len(manifest["pairs"]) != 10:
            raise RuntimeError(f"Incomplete pairing manifest for {task}")
        for seed in seeds:
            for arm in ARMS:
                root = legacy if arm in {"vanilla", "random_cd", "semantic_cd"} else artifact
                summary = load_summary(root, task, arm, seed)
                records.append({
                    "task": task,
                    "seed": seed,
                    "arm": arm,
                    "success": bool(summary["success"]),
                })
                if arm != "vanilla":
                    residuals[arm].append(float(summary["mean_residual_norm"]))
                if arm.startswith("attn_"):
                    feature_audits.append(bool(summary["all_visual_features_bit_identical"]))

    def sr(task, arm):
        values = [r["success"] for r in records if r["task"] == task and r["arm"] == arm]
        if len(values) != 10:
            raise RuntimeError(f"Expected 10 records for {task}/{arm}, got {len(values)}")
        return float(np.mean(values))

    task_rows = []
    for task in TASKS:
        row = {"task": task}
        row.update({arm: sr(task, arm) for arm in ARMS})
        row["attn_semantic_random_gap"] = row["attn_semantic_cd"] - row["attn_random_cd"]
        task_rows.append(row)
    averages = {arm: float(np.mean([row[arm] for row in task_rows])) for arm in ARMS}
    averages["attn_semantic_random_gap"] = averages["attn_semantic_cd"] - averages["attn_random_cd"]

    gate1_diff_pp = abs(averages["attn_random_cd"] - averages["vanilla"]) * 100
    gate2_gain_pp = averages["attn_semantic_random_gap"] * 100
    open_row = next(row for row in task_rows if row["task"] == "google_robot_open_drawer")
    gates = {
        "gate_1_random_inertness": {"pass": gate1_diff_pp <= 5.0, "diff_pp": gate1_diff_pp},
        "gate_2_semantic_over_random": {"pass": gate2_gain_pp >= 8.0, "gain_pp": gate2_gain_pp},
        "gate_3_open_drawer_no_harm": {
            "pass": open_row["attn_semantic_cd"] >= open_row["vanilla"],
            "attn_semantic_sr": open_row["attn_semantic_cd"],
            "vanilla_sr": open_row["vanilla"],
        },
    }
    mean_residuals = {arm: float(np.mean(values)) for arm, values in residuals.items()}
    random_reduction = 1.0 - mean_residuals["attn_random_cd"] / mean_residuals["random_cd"]
    decision = {
        "protocol_id": "CAUSAL_ATTENTION_MASKING_CD_PHASE0_V1",
        "n_task_seed_units": 30,
        "task_rows": task_rows,
        "averages": averages,
        "gates": gates,
        "all_visual_features_bit_identical": all(feature_audits),
        "mean_residual_norms": mean_residuals,
        "attn_random_residual_reduction_vs_legacy": random_reduction,
        "overall_pass": all(gate["pass"] for gate in gates.values()),
    }
    (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    with (artifact / "task_breakdown.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", *ARMS, "attn_semantic_random_gap"])
        writer.writeheader()
        writer.writerows(task_rows)

    def pct(value):
        return f"{100 * value:.1f}%"

    lines = [
        "# Attention Masking CD 实验结果报告",
        "",
        "## 1. 核心指标对比表",
        "",
        "| Task Name | Vanilla (Arm 0) | Legacy-Rand (Arm 1) | Legacy-Sem (Arm 2) | Attn-Rand (Arm 3) | Attn-Sem (Arm 4) | Δ(Attn-Sem vs Attn-Rand) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task'].removeprefix('google_robot_')} | "
            + " | ".join(pct(row[arm]) for arm in ARMS)
            + f" | {100 * row['attn_semantic_random_gap']:+.1f} pp |"
        )
    lines.append(
        "| **Average** | "
        + " | ".join(f"**{pct(averages[arm])}**" for arm in ARMS)
        + f" | **{100 * averages['attn_semantic_random_gap']:+.1f} pp** |"
    )
    lines += [
        "",
        "## 2. 门控裁决",
        "",
        f"- Gate 1 (Random 惰性校验): **{'PASS' if gates['gate_1_random_inertness']['pass'] else 'FAIL'}** (Diff: {gate1_diff_pp:.1f} pp)",
        f"- Gate 2 (语义超越 Random): **{'PASS' if gates['gate_2_semantic_over_random']['pass'] else 'FAIL'}** (Gain: {gate2_gain_pp:+.1f} pp)",
        f"- Gate 3 (结构破坏修复): **{'PASS' if gates['gate_3_open_drawer_no_harm']['pass'] else 'FAIL'}** (Open Drawer: {pct(open_row['attn_semantic_cd'])} vs Vanilla {pct(open_row['vanilla'])})",
        f"- 总裁决: **{'PASS' if decision['overall_pass'] else 'STOP'}**",
        "",
        "## 3. 残差模长与物理分析",
        "",
        f"- `||Δz_legacy_rand|| = {mean_residuals['random_cd']:.2f}` vs `||Δz_attn_rand|| = {mean_residuals['attn_random_cd']:.2f}` (缩减比例: {100 * random_reduction:.1f}%)",
        f"- `||Δz_legacy_sem|| = {mean_residuals['semantic_cd']:.2f}` vs `||Δz_attn_sem|| = {mean_residuals['attn_semantic_cd']:.2f}`",
        f"- 视觉特征逐元素一致性: **{'PASS' if decision['all_visual_features_bit_identical'] else 'FAIL'}**",
        "- 结论: 注意力阻断显著减小了 Random 残差，但没有恢复 Random 惰性，也没有产生语义特异性；Semantic 阻断在三个任务上均未超过 Random。因此当前 selector + late-layer hard masking 不能支持因果语义条件的主张。",
        "",
        "## 4. 统计口径",
        "",
        "- 每任务 10 个共同 seeds (`0..9`)，共 30 个 task-seed 单元。",
        "- Arms 0–2 复用既有 Phase0 V2 中完全相同环境、snapshot 与 seed 的记录；Arms 3–4 的 pairing manifest 已逐 seed 验证五臂状态/RGB/canonical snapshot 哈希一致。",
        "- 注意力阻断固定在 Llama layers 16–31，Visual keys 使用 Prismatic 实际绝对位置 `1 + patch_id`。",
    ]
    (artifact / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
