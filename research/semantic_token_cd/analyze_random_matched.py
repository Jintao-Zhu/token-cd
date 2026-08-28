"""Analyze semantic specificity against a strictly size-matched random arm."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.random_matched_rollout import ARM, PROTOCOL, TASKS


REFERENCE_ARMS = ("vanilla", "attn_uniform", "attn_semantic")


def bootstrap_gap(a: np.ndarray, b: np.ndarray, seed: int) -> tuple[float, float]:
    differences = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(10000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def rescue_harm(vanilla: np.ndarray, method: np.ndarray) -> dict:
    rescue = int(np.sum((vanilla == 0) & (method == 1)))
    harm = int(np.sum((vanilla == 1) & (method == 0)))
    ratio = float("inf") if harm == 0 and rescue > 0 else (
        0.0 if harm == 0 else rescue / harm
    )
    return {"rescue": rescue, "harm": harm, "ratio": ratio}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    reference = args.reference_artifact.resolve()
    records = []
    feature_equal = True
    token_count_checks = 0

    for task in TASKS:
        reference_manifest = json.loads(
            (reference / "episodes" / task / "pairing_manifest.json").read_text()
        )
        random_manifest = json.loads(
            (artifact / "episodes" / task / "pairing_manifest.json").read_text()
        )
        expected_seeds = (
            list(range(50))
            if task == "google_robot_pick_coke_can"
            else [seed for seed in range(50) if seed != 4]
        )
        if (
            not reference_manifest["all_three_arm_exact_pairing"]
            or not random_manifest["all_four_arm_exact_pairing"]
            or random_manifest["seeds"] != expected_seeds
        ):
            raise RuntimeError(f"Incomplete manifest for {task}")
        reference_pairs = {int(pair["seed"]): pair for pair in reference_manifest["pairs"]}
        random_pairs = {int(pair["seed"]): pair for pair in random_manifest["pairs"]}
        for seed in expected_seeds:
            reference_pair = reference_pairs[seed]
            random_pair = random_pairs[seed]
            canonical = reference_pair["canonical_snapshot_sha256"]
            if random_pair["canonical_snapshot_sha256"] != canonical:
                raise RuntimeError(f"Four-arm snapshot mismatch for {task} seed {seed}")
            for arm in REFERENCE_ARMS:
                path = Path(reference_pair["sources"][arm])
                summary = json.loads(path.read_text())
                records.append({
                    "task": task,
                    "seed": seed,
                    "arm": arm,
                    "success": int(bool(summary["success"])),
                })
            random_path = Path(random_pair["summary"])
            random_summary = json.loads(random_path.read_text())
            feature_equal &= bool(random_summary["all_visual_features_bit_identical"])
            for step in random_summary["selector_trace"]:
                if step["num_tokens"] != len(step["reference_semantic_token_ids"]):
                    raise RuntimeError(f"Token-count mismatch in {random_path}")
                token_count_checks += 1
            records.append({
                "task": task,
                "seed": seed,
                "arm": ARM,
                "success": int(bool(random_summary["success"])),
            })
    if len(records) != 592:
        raise RuntimeError(f"Expected 592 logical records, got {len(records)}")

    def values(task: str | None, arm: str) -> np.ndarray:
        return np.asarray([
            record["success"] for record in records
            if record["arm"] == arm and (task is None or record["task"] == task)
        ], dtype=np.int64)

    task_rows = []
    for task_index, task in enumerate(TASKS):
        vanilla = values(task, "vanilla")
        semantic = values(task, "attn_semantic")
        random = values(task, ARM)
        row = {
            "task": task,
            "vanilla": float(vanilla.mean()),
            "attn_uniform": float(values(task, "attn_uniform").mean()),
            "attn_semantic": float(semantic.mean()),
            ARM: float(random.mean()),
            "semantic_random_gap": float(semantic.mean() - random.mean()),
            "semantic_random_ci": bootstrap_gap(semantic, random, 20261001 + task_index),
            "semantic_rescue_harm": rescue_harm(vanilla, semantic),
            "random_rescue_harm": rescue_harm(vanilla, random),
        }
        task_rows.append(row)

    averages = {
        arm: float(values(None, arm).mean())
        for arm in (*REFERENCE_ARMS, ARM)
    }
    vanilla = values(None, "vanilla")
    semantic = values(None, "attn_semantic")
    random = values(None, ARM)
    semantic_gap = averages["attn_semantic"] - averages[ARM]
    task_wins = sum(row["semantic_random_gap"] > 0 for row in task_rows)
    semantic_rh = rescue_harm(vanilla, semantic)
    random_rh = rescue_harm(vanilla, random)
    gates = {
        "S1_semantic_minus_random_matched": {
            "pass": semantic_gap >= 0.05,
            "delta_pp": 100 * semantic_gap,
        },
        "S2_semantic_task_wins": {
            "pass": task_wins >= 2,
            "wins": task_wins,
            "total": 3,
        },
        "S3_semantic_rescue_harm": {
            "pass": semantic_rh["ratio"] > random_rh["ratio"],
            "semantic": semantic_rh,
            "random_matched": random_rh,
        },
    }
    decision = {
        "protocol_id": PROTOCOL,
        "n_new_episodes": 148,
        "n_logical_four_arm_episodes": 592,
        "technical_exclusions": {
            "google_robot_open_drawer": [4],
            "google_robot_close_drawer": [4],
        },
        "average_success_rates": averages,
        "task_rows": task_rows,
        "semantic_random_gap_bootstrap_95ci": bootstrap_gap(semantic, random, 20261010),
        "gates": gates,
        "overall_pass": all(gate["pass"] for gate in gates.values()),
        "all_random_visual_features_bit_identical": feature_equal,
        "matched_token_count_step_checks": token_count_checks,
    }
    (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    with (artifact / "task_breakdown.csv").open("w", newline="") as handle:
        fieldnames = ["task", "vanilla", "attn_uniform", "attn_semantic", ARM, "semantic_random_gap"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in task_rows)

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    lines = [
        "# Semantic vs Random-Matched Attention-CD 报告",
        "",
        "## 1. 成功率",
        "",
        "| Task | Vanilla | Uniform-64 | Semantic | Random-Matched | Semantic−Random |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task'].removeprefix('google_robot_')} | {pct(row['vanilla'])} | "
            f"{pct(row['attn_uniform'])} | {pct(row['attn_semantic'])} | "
            f"{pct(row[ARM])} | {100 * row['semantic_random_gap']:+.1f} pp |"
        )
    lines.append(
        f"| **Average** | **{pct(averages['vanilla'])}** | **{pct(averages['attn_uniform'])}** | "
        f"**{pct(averages['attn_semantic'])}** | **{pct(averages[ARM])}** | **{100 * semantic_gap:+.1f} pp** |"
    )
    ci = decision["semantic_random_gap_bootstrap_95ci"]
    lines += [
        "",
        "## 2. 预注册门控",
        "",
        f"- S1 Semantic−Random ≥ +5pp: **{'PASS' if gates['S1_semantic_minus_random_matched']['pass'] else 'FAIL'}** ({100 * semantic_gap:+.1f} pp)",
        f"- S2 Semantic task wins ≥ 2/3: **{'PASS' if gates['S2_semantic_task_wins']['pass'] else 'FAIL'}** ({task_wins}/3)",
        f"- S3 Semantic Rescue/Harm > Random: **{'PASS' if gates['S3_semantic_rescue_harm']['pass'] else 'FAIL'}** (Semantic {semantic_rh['rescue']}/{semantic_rh['harm']}; Random {random_rh['rescue']}/{random_rh['harm']})",
        f"- Semantic−Random 配对 bootstrap 95% CI: `[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp`",
        f"- 总裁决: **{'PASS' if decision['overall_pass'] else 'STOP'}**",
        "",
        "## 3. 完整性",
        "",
        f"- Random 负分支视觉特征逐元素一致性: **{'PASS' if feature_equal else 'FAIL'}**。",
        f"- 已逐控制步验证 Random 与参考 Semantic token 数相同: `{token_count_checks}` 次。",
        "- Random 通过置换 KMeans membership 保留 selected group ID、group 数、每组大小和总 Token 数；唯一移除的是 Token membership 的语义结构。",
    ]
    (artifact / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
