"""Full analysis for Spatial Grid vs Random Attention-CD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.spatial_grid_rollout import ARMS, PROTOCOL, TASKS


LABELS = {
    "vanilla": "Vanilla",
    "attn_rand_64": "Rand-64",
    "attn_grid_strided": "Grid-Strided",
    "attn_grid_checkerboard": "Checkerboard",
    "attn_sem_hard": "Sem-Hard",
    "attn_sem_sparse": "Sem-Sparse",
}
SHORT_TASKS = {
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_move_near": "move_near",
    "google_robot_place_apple_in_closed_top_drawer": "apple_in_drawer",
}


def paired_bootstrap_ci(values: np.ndarray, seed: int = 20260825) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(10000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    records = []
    all_feature_equal = True
    for task in TASKS:
        manifest = json.loads((artifact / "episodes" / task / "pairing_manifest.json").read_text())
        if not manifest["all_six_arm_exact_pairing"] or manifest["seeds"] != list(range(30)):
            raise RuntimeError(f"Incomplete pairing manifest for {task}")
        for seed in range(30):
            for arm in ARMS:
                path = artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                summary = json.loads(path.read_text())
                if summary["task"] != task or summary["seed"] != seed or summary["arm"] != arm:
                    raise RuntimeError(f"Identity mismatch in {path}")
                all_feature_equal &= bool(summary["all_visual_features_bit_identical"])
                records.append({
                    "task": task,
                    "seed": seed,
                    "arm": arm,
                    "success": bool(summary["success"]),
                    "jitter": float(summary["action_jitter_index"]),
                    "first_residual": float(summary["first_step_residual_norm"]),
                })
    if len(records) != 900:
        raise RuntimeError(f"Expected 900 episode records, got {len(records)}")

    def values(task: str | None, arm: str, field: str) -> np.ndarray:
        return np.asarray([
            record[field] for record in records
            if record["arm"] == arm and (task is None or record["task"] == task)
        ], dtype=np.float64)

    task_rows = []
    for task in TASKS:
        row = {"task": task}
        for arm in ARMS:
            row[arm] = float(values(task, arm, "success").mean())
        best = max(row[arm] for arm in ARMS)
        row["best_arms"] = [arm for arm in ARMS if row[arm] == best]
        task_rows.append(row)
    averages = {arm: float(values(None, arm, "success").mean()) for arm in ARMS}

    grid_random_pp = 100 * (averages["attn_grid_strided"] - averages["attn_rand_64"])
    grid_vanilla_pp = 100 * (averages["attn_grid_strided"] - averages["vanilla"])
    grid_task_wins = sum(
        row["attn_grid_strided"] > row["attn_rand_64"] for row in task_rows
    )
    gates = {
        "G1_grid_vs_random": {"pass": grid_random_pp >= 5.0, "delta_pp": grid_random_pp},
        "G2_grid_vs_vanilla": {"pass": grid_vanilla_pp >= 20.0, "delta_pp": grid_vanilla_pp},
        "G3_task_coverage": {"pass": grid_task_wins >= 4, "wins": grid_task_wins, "total": 5, "comparison": "Grid-Strided > Rand-64"},
        "G4_sparse_semantic": {
            "pass": averages["attn_sem_sparse"] > averages["attn_sem_hard"],
            "sparse_sr": averages["attn_sem_sparse"],
            "hard_sr": averages["attn_sem_hard"],
        },
    }

    diagnostics = {}
    for arm in ARMS:
        jitter = values(None, arm, "jitter")
        residual = values(None, arm, "first_residual")
        diagnostics[arm] = {
            "jitter_mean": float(jitter.mean()),
            "jitter_variance": float(jitter.var(ddof=1)),
            "first_residual_mean": float(residual.mean()),
            "first_residual_variance": float(residual.var(ddof=1)),
        }
    jitter_difference = (
        values(None, "attn_grid_strided", "jitter")
        - values(None, "attn_rand_64", "jitter")
    )
    residual_difference = (
        values(None, "attn_grid_strided", "first_residual")
        - values(None, "attn_rand_64", "first_residual")
    )
    diagnostics["paired_grid_minus_random"] = {
        "jitter_mean_difference": float(jitter_difference.mean()),
        "jitter_bootstrap_95ci": paired_bootstrap_ci(jitter_difference),
        "first_residual_mean_difference": float(residual_difference.mean()),
        "first_residual_bootstrap_95ci": paired_bootstrap_ci(residual_difference, seed=20260826),
    }
    decision = {
        "protocol_id": PROTOCOL,
        "n_episodes": len(records),
        "n_paired_task_seed_units": 150,
        "task_rows": task_rows,
        "average_success_rates": averages,
        "gates": gates,
        "diagnostics": diagnostics,
        "all_visual_features_bit_identical": all_feature_equal,
        "overall_pass": all(gate["pass"] for gate in gates.values()),
    }
    (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    with (artifact / "task_breakdown.csv").open("w", newline="") as handle:
        fieldnames = ["task", *ARMS, "best_arms"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in task_rows:
            writer.writerow({**row, "best_arms": ";".join(row["best_arms"])})

    def pct(value):
        return f"{100 * value:.1f}%"

    lines = [
        "# Spatial Grid vs. Random Attention-CD 实验全量报告",
        "",
        "## 1. 全任务成功率汇总表 (5 Tasks × 30 Seeds = 900 Runs)",
        "",
        "| Task Name | Vanilla (Arm 0) | Rand-64 (Arm 1) | Grid-Strided (Arm 2) | Checkerboard (Arm 3) | Sem-Hard (Arm 4) | Sem-Sparse (Arm 5) | Best Arm |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in task_rows:
        best = ", ".join(LABELS[arm] for arm in row["best_arms"])
        lines.append(
            f"| {SHORT_TASKS[row['task']]} | "
            + " | ".join(pct(row[arm]) for arm in ARMS)
            + f" | {best} |"
        )
    lines.append(
        "| **通盘平均 SR** | "
        + " | ".join(f"**{pct(averages[arm])}**" for arm in ARMS)
        + " | - |"
    )
    lines += [
        "",
        "## 2. 自动化门控裁决",
        "",
        f"- Gate G1 (Grid vs. Random): **{'PASS' if gates['G1_grid_vs_random']['pass'] else 'FAIL'}** (Δ = {grid_random_pp:+.1f} pp)",
        f"- Gate G2 (Grid vs. Vanilla): **{'PASS' if gates['G2_grid_vs_vanilla']['pass'] else 'FAIL'}** (Δ = {grid_vanilla_pp:+.1f} pp)",
        f"- Gate G3 (任务普适性覆盖): **{'PASS' if gates['G3_task_coverage']['pass'] else 'FAIL'}** ({grid_task_wins} / 5 胜出)",
        f"- Gate G4 (稀疏语义改善): **{'PASS' if gates['G4_sparse_semantic']['pass'] else 'FAIL'}** (Sparse {pct(averages['attn_sem_sparse'])} vs Hard {pct(averages['attn_sem_hard'])})",
        f"- 总裁决: **{'PASS' if decision['overall_pass'] else 'STOP'}**",
        "",
        "## 3. 轨迹平滑度与首步残差",
        "",
        "| Arm | Jitter Mean | Jitter Variance | First-step Residual Mean | First-step Residual Variance |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        item = diagnostics[arm]
        lines.append(
            f"| {LABELS[arm]} | {item['jitter_mean']:.5f} | {item['jitter_variance']:.5f} | {item['first_residual_mean']:.3f} | {item['first_residual_variance']:.3f} |"
        )
    paired = diagnostics["paired_grid_minus_random"]
    jitter_ci = paired["jitter_bootstrap_95ci"]
    residual_ci = paired["first_residual_bootstrap_95ci"]
    lines += [
        "",
        f"- Grid−Random jitter 配对均值差: `{paired['jitter_mean_difference']:+.5f}`，bootstrap 95% CI `[{jitter_ci[0]:+.5f}, {jitter_ci[1]:+.5f}]`。",
        f"- Grid−Random 首步残差配对均值差: `{paired['first_residual_mean_difference']:+.3f}`，bootstrap 95% CI `[{residual_ci[0]:+.3f}, {residual_ci[1]:+.3f}]`。",
        f"- 视觉特征逐元素一致性: **{'PASS' if all_feature_equal else 'FAIL'}**。",
        "",
        "## 4. 实验口径",
        "",
        "- 所有六臂均重新运行，使用相同的 150 个 task-seed 初始 snapshot；旧 `Attn-Random=50%` 是动态 token 数对照，不作为本实验 Rand-64 数据。",
        "- Rand-64 与 Sem-Sparse 均按 episode seed 和控制步确定性采样；Grid-Strided 固定选择偶数行×偶数列的 64 个 token。",
        "- Attention mask 请求值为 `-1e4`，bfloat16 实际值为 `-9984`；阻断层固定为 16–31。",
        "- Checkerboard 使用 128 tokens，因此它同时改变几何与密度，只作为诊断臂，不能单独归因于规则性。",
    ]
    (artifact / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
