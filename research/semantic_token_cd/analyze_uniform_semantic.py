"""Analyze the 50-seed, three-arm Uniform vs Semantic experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.uniform_semantic_rollout import PROTOCOL, TASKS


ARMS = ("vanilla", "attn_uniform", "attn_semantic")
LABELS = {
    "vanilla": "Vanilla",
    "attn_uniform": "Attn-Uniform",
    "attn_semantic": "Attn-Semantic",
}
SHORT_TASKS = {
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
}


def bootstrap_gap(success_a: np.ndarray, success_b: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = success_a - success_b
    means = np.empty(10000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    records = []
    feature_equal = True
    source_counts: dict[str, int] = {}

    for task in TASKS:
        manifest_path = artifact / "episodes" / task / "pairing_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            not manifest["all_three_arm_exact_pairing"]
            or manifest["seeds"] != list(range(50))
            or len(manifest["pairs"]) != 50
        ):
            raise RuntimeError(f"Incomplete pairing manifest: {manifest_path}")
        for pair in manifest["pairs"]:
            seed = int(pair["seed"])
            canonical = pair["canonical_snapshot_sha256"]
            for arm in ARMS:
                path = Path(pair["sources"][arm])
                summary = json.loads(path.read_text())
                if (
                    summary["task"] != task
                    or int(summary["seed"]) != seed
                    or summary["canonical_snapshot_sha256"] != canonical
                ):
                    raise RuntimeError(f"Identity or snapshot mismatch: {path}")
                source = str(path.parents[3])
                source_counts[source] = source_counts.get(source, 0) + 1
                if arm != "vanilla":
                    feature_equal &= bool(summary["all_visual_features_bit_identical"])
                records.append({
                    "task": task,
                    "seed": seed,
                    "arm": arm,
                    "success": int(bool(summary["success"])),
                    "jitter": summary.get("action_jitter_index"),
                    "first_residual": summary.get("first_step_residual_norm"),
                })
    if len(records) != 450:
        raise RuntimeError(f"Expected 450 records, got {len(records)}")

    def values(task: str | None, arm: str, field: str) -> np.ndarray:
        return np.asarray([
            record[field] for record in records
            if record["arm"] == arm and (task is None or record["task"] == task)
        ], dtype=np.float64)

    task_rows = []
    for task_index, task in enumerate(TASKS):
        row = {"task": task}
        for arm in ARMS:
            successes = values(task, arm, "success")
            row[f"{arm}_successes"] = int(successes.sum())
            row[arm] = float(successes.mean())
            row[f"{arm}_variance"] = float(successes.var(ddof=1))
        row["uniform_vanilla_gap"] = row["attn_uniform"] - row["vanilla"]
        row["uniform_semantic_gap"] = row["attn_uniform"] - row["attn_semantic"]
        row["uniform_vanilla_ci"] = bootstrap_gap(
            values(task, "attn_uniform", "success"),
            values(task, "vanilla", "success"),
            20260830 + task_index,
        )
        row["uniform_semantic_ci"] = bootstrap_gap(
            values(task, "attn_uniform", "success"),
            values(task, "attn_semantic", "success"),
            20260910 + task_index,
        )
        task_rows.append(row)

    averages = {arm: float(values(None, arm, "success").mean()) for arm in ARMS}
    counts = {arm: int(values(None, arm, "success").sum()) for arm in ARMS}
    uniform_vanilla_gap = averages["attn_uniform"] - averages["vanilla"]
    uniform_semantic_gap = averages["attn_uniform"] - averages["attn_semantic"]
    uniform_task_wins = sum(row["uniform_semantic_gap"] > 0 for row in task_rows)
    semantic_task_wins = sum(row["uniform_semantic_gap"] < 0 for row in task_rows)
    overall_uniform_vanilla_ci = bootstrap_gap(
        values(None, "attn_uniform", "success"), values(None, "vanilla", "success"), 20260920
    )
    overall_uniform_semantic_ci = bootstrap_gap(
        values(None, "attn_uniform", "success"), values(None, "attn_semantic", "success"), 20260921
    )

    if uniform_semantic_gap > 0 and uniform_task_wins >= 2:
        gate2_decision = "uniform_wins"
    elif uniform_semantic_gap < 0:
        gate2_decision = "semantic_wins"
    else:
        gate2_decision = "inconclusive"
    gates = {
        "gate_1_uniform_vs_vanilla": {
            "pass": uniform_vanilla_gap >= 0.15,
            "delta_pp": 100 * uniform_vanilla_gap,
        },
        "gate_2_uniform_vs_semantic": {
            "decision": gate2_decision,
            "delta_pp": 100 * uniform_semantic_gap,
            "uniform_task_wins": uniform_task_wins,
            "semantic_task_wins": semantic_task_wins,
        },
    }

    diagnostics = {}
    for arm in ("attn_uniform", "attn_semantic"):
        jitter = values(None, arm, "jitter")
        residual = values(None, arm, "first_residual")
        diagnostics[arm] = {
            "jitter_mean": float(jitter.mean()),
            "jitter_variance": float(jitter.var(ddof=1)),
            "first_residual_mean": float(residual.mean()),
            "first_residual_variance": float(residual.var(ddof=1)),
        }
    decision = {
        "protocol_id": PROTOCOL,
        "n_episodes": len(records),
        "n_task_seed_units": 150,
        "task_rows": task_rows,
        "average_success_rates": averages,
        "success_counts": counts,
        "uniform_vanilla_gap_bootstrap_95ci": overall_uniform_vanilla_ci,
        "uniform_semantic_gap_bootstrap_95ci": overall_uniform_semantic_ci,
        "gates": gates,
        "diagnostics": diagnostics,
        "all_attention_visual_features_bit_identical": feature_equal,
        "source_counts": source_counts,
    }
    (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    with (artifact / "task_breakdown.csv").open("w", newline="") as handle:
        fieldnames = [
            "task", "vanilla_successes", "vanilla", "attn_uniform_successes", "attn_uniform",
            "attn_semantic_successes", "attn_semantic", "uniform_vanilla_gap", "uniform_semantic_gap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in task_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    def rate(value: float, successes: int, total: int) -> str:
        return f"{100 * value:.1f}% ({successes}/{total})"

    lines = [
        "# 450 Episodes 三臂对照实验报告 (50 Seeds/Arm)",
        "",
        "## 1. 成功率汇总表",
        "",
        "| 任务名称 | Vanilla (Arm 0) | Attn-Uniform (Arm 1) | Attn-Semantic (Arm 2) | Δ(Uniform - Vanilla) | Δ(Uniform - Sem) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in task_rows:
        lines.append(
            f"| {SHORT_TASKS[row['task']]} (50 seeds) | "
            f"{rate(row['vanilla'], row['vanilla_successes'], 50)} | "
            f"{rate(row['attn_uniform'], row['attn_uniform_successes'], 50)} | "
            f"{rate(row['attn_semantic'], row['attn_semantic_successes'], 50)} | "
            f"{100 * row['uniform_vanilla_gap']:+.1f} pp | {100 * row['uniform_semantic_gap']:+.1f} pp |"
        )
    lines.append(
        "| **平均成功率 (150 runs/arm)** | "
        f"**{rate(averages['vanilla'], counts['vanilla'], 150)}** | "
        f"**{rate(averages['attn_uniform'], counts['attn_uniform'], 150)}** | "
        f"**{rate(averages['attn_semantic'], counts['attn_semantic'], 150)}** | "
        f"**{100 * uniform_vanilla_gap:+.1f} pp** | **{100 * uniform_semantic_gap:+.1f} pp** |"
    )
    gate1 = gates["gate_1_uniform_vs_vanilla"]
    gate2 = gates["gate_2_uniform_vs_semantic"]
    lines += [
        "",
        "## 2. 门控判定",
        "",
        f"- Gate 1 (Uniform vs. Vanilla): **{'PASS' if gate1['pass'] else 'FAIL'}** (Δ = {gate1['delta_pp']:+.1f} pp)",
        f"- Gate 2 (Uniform vs. Semantic): **{gate2['decision']}** (Δ = {gate2['delta_pp']:+.1f} pp; Uniform task wins {uniform_task_wins}/3)",
        f"- Uniform−Vanilla 配对 bootstrap 95% CI: `[{100 * overall_uniform_vanilla_ci[0]:+.1f}, {100 * overall_uniform_vanilla_ci[1]:+.1f}] pp`",
        f"- Uniform−Semantic 配对 bootstrap 95% CI: `[{100 * overall_uniform_semantic_ci[0]:+.1f}, {100 * overall_uniform_semantic_ci[1]:+.1f}] pp`",
        "",
        "## 3. 统计学与物理机制结论",
        "",
        f"- Attn-Uniform jitter: mean `{diagnostics['attn_uniform']['jitter_mean']:.5f}`, variance `{diagnostics['attn_uniform']['jitter_variance']:.5f}`。",
        f"- Attn-Semantic jitter: mean `{diagnostics['attn_semantic']['jitter_mean']:.5f}`, variance `{diagnostics['attn_semantic']['jitter_variance']:.5f}`。",
        f"- Attn-Uniform first-step residual: mean `{diagnostics['attn_uniform']['first_residual_mean']:.3f}`, variance `{diagnostics['attn_uniform']['first_residual_variance']:.3f}`。",
        f"- Attn-Semantic first-step residual: mean `{diagnostics['attn_semantic']['first_residual_mean']:.3f}`, variance `{diagnostics['attn_semantic']['first_residual_variance']:.3f}`。",
        f"- 视觉特征逐元素一致性: **{'PASS' if feature_equal else 'FAIL'}**。",
        "- Vanilla 复用既有 50-seed 审计结果；注意力臂复用参数完全一致的已完成前缀，并补跑缺失 seeds。所有三臂均逐 seed 校验初始 state、RGB 与 canonical snapshot 哈希。",
    ]
    (artifact / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
