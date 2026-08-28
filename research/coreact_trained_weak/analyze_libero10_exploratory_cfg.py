from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ARTIFACT = Path("artifacts/coreact_trained_weak_libero10_quality_screen_v1_20260812_212541")
ARMS = ("Strong_15k", "Weak_10k", "Midpoint", "CFG")


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_stats(rows: list[dict], left: str, right: str, rng: np.random.Generator) -> dict:
    by_task = np.asarray(
        [
            [int(rows[(task * 10) + init][left]) - int(rows[(task * 10) + init][right]) for init in range(10)]
            for task in range(10)
        ],
        dtype=np.int8,
    )
    flat = by_task.reshape(-1)
    bootstrap = np.empty(100_000)
    stratified = np.empty(100_000)
    for start in range(0, 100_000, 5_000):
        indices = rng.integers(0, 100, (5_000, 100))
        bootstrap[start : start + 5_000] = flat[indices].mean(axis=1) * 100
        task_indices = rng.integers(0, 10, (5_000, 10, 10))
        sampled = np.take_along_axis(by_task[None, :, :], task_indices, axis=2)
        stratified[start : start + 5_000] = sampled.mean(axis=(1, 2)) * 100
    rescued = int((flat == 1).sum())
    harmed = int((flat == -1).sum())
    discordant = rescued + harmed
    return {
        "effect_pp": float(flat.mean() * 100),
        "paired_bootstrap_95_ci_pp": interval(bootstrap),
        "task_stratified_bootstrap_95_ci_pp": interval(stratified),
        "mcnemar_exact_p": float(binomtest(min(rescued, harmed), discordant).pvalue) if discordant else 1.0,
        "rescued": rescued,
        "harmed": harmed,
    }


def main() -> None:
    episode_paths = sorted((ARTIFACT / "episodes").glob("*.json"))
    if len(episode_paths) != 400:
        raise RuntimeError(f"expected 400 episode files, got {len(episode_paths)}")
    episodes = [json.loads(path.read_text()) for path in episode_paths]
    units: dict[tuple[int, int], dict[str, dict]] = {}
    for episode in episodes:
        if episode.get("status") != "complete" or episode.get("all_actions_finite") is not True:
            raise RuntimeError(f"incomplete/nonfinite episode: {episode['episode_id']}")
        key = (int(episode["task_id"]), int(episode["init_state_id"]))
        units.setdefault(key, {})[episode["arm"]] = episode
    if sorted(units) != [(task, init) for task in range(10) for init in range(10)]:
        raise RuntimeError("missing or unexpected task/init units")

    audit_failures = []
    rows = []
    for key in sorted(units):
        group = units[key]
        if set(group) != set(ARMS):
            audit_failures.append(f"{key}: arms={sorted(group)}")
            continue
        for field in ("initial_sim_state_sha256", "initial_prepared_input_sha256"):
            if len({group[arm][field] for arm in ARMS}) != 1:
                audit_failures.append(f"{key}: {field} mismatch")
        if len({group[arm]["noise_sha256_by_replan"][0] for arm in ARMS}) != 1:
            audit_failures.append(f"{key}: first noise mismatch")
        rows.append({"task_id": key[0], "init_state_id": key[1], **{arm: bool(group[arm]["success"]) for arm in ARMS}})
    if audit_failures:
        raise RuntimeError("; ".join(audit_failures[:10]))

    rates = {arm: sum(row[arm] for row in rows) / 100 for arm in ARMS}
    task_rows = []
    for task in range(10):
        subset = [row for row in rows if row["task_id"] == task]
        counts = {arm: sum(row[arm] for row in subset) for arm in ARMS}
        task_rows.append({
            "task_id": task,
            **counts,
            "CFG_minus_Strong_pp": float((counts["CFG"] - counts["Strong_15k"]) * 10),
        })
    rng = np.random.default_rng(1729)
    comparisons = {
        "CFG_minus_Strong": paired_stats(rows, "CFG", "Strong_15k", rng),
        "CFG_minus_Midpoint": paired_stats(rows, "CFG", "Midpoint", rng),
        "Weak_minus_Strong": paired_stats(rows, "Weak_10k", "Strong_15k", rng),
        "Midpoint_minus_Strong": paired_stats(rows, "Midpoint", "Strong_15k", rng),
    }
    deltas = [row["CFG_minus_Strong_pp"] for row in task_rows]
    positive, ties, negative = sum(x > 0 for x in deltas), sum(x == 0 for x in deltas), sum(x < 0 for x in deltas)
    catastrophic = any(x < -10 for x in deltas)
    effect = comparisons["CFG_minus_Strong"]["effect_pp"]
    observed = effect >= 5 and comparisons["CFG_minus_Strong"]["paired_bootstrap_95_ci_pp"][0] > 0
    decision = "LIBERO10_EXPLORATORY_CFG_BENEFIT_OBSERVED" if observed else "LIBERO10_EXPLORATORY_CFG_NO_BENEFIT"
    analysis = {
        "decision": decision,
        "interpretation": "exploratory_after_failed_quality_gap_gate",
        "episodes": 400,
        "matched_units": 100,
        "success_rates": rates,
        "comparisons": comparisons,
        "tasks_CFG_positive_tie_negative_vs_Strong": [positive, ties, negative],
        "catastrophic_task_harm_gt_10pp": catastrophic,
        "lambda": 0.5,
        "trust_region_kappa": 0.25,
        "posthoc_tuning": False,
        "changes_prior_spatial_no_go": False,
    }
    (ARTIFACT / "exploratory_four_arm_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (ARTIFACT / "exploratory_four_arm_decision.json").write_text(
        json.dumps({"decision": decision, "confirmatory": False, "analysis": analysis}, indent=2) + "\n"
    )
    integrity = {
        "status": "PASS",
        "episode_files": 400,
        "matched_units": 100,
        "required_arms_per_unit": list(ARMS),
        "initial_sim_state_hash_match": "100/100",
        "initial_prepared_input_hash_match": "100/100",
        "first_noise_hash_match": "100/100",
        "complete_and_finite": "400/400",
    }
    (ARTIFACT / "exploratory_four_arm_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    with (ARTIFACT / "exploratory_four_arm_task_success.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_rows[0].keys())
        writer.writeheader()
        writer.writerows(task_rows)

    primary = comparisons["CFG_minus_Strong"]
    table = "\n".join(
        f"| {row['task_id']} | {row['Strong_15k']} | {row['Weak_10k']} | {row['Midpoint']} | {row['CFG']} | {row['CFG_minus_Strong_pp']:+.0f}pp |"
        for row in task_rows
    )
    report = f"""# LIBERO-10 Trained-Weak CFG 探索性四臂报告

## 结论

`{decision}`

本实验原本因 Strong/Weak 质量差仅 5pp（低于 8pp gate）而停止，后经用户明确授权继续。以下结果是探索性证据，不是确认性结论，也不改变此前 LIBERO-Spatial 上 trained-weak CFG `+1pp` 的正式 no-go。

| Arm | 成功率 |
|---|---:|
| Strong 15k | {rates['Strong_15k']:.0%} |
| Weak 10k | {rates['Weak_10k']:.0%} |
| Midpoint | {rates['Midpoint']:.0%} |
| CFG, lambda=0.5 | {rates['CFG']:.0%} |

- CFG - Strong: {effect:+.1f}pp
- paired bootstrap 95% CI: [{primary['paired_bootstrap_95_ci_pp'][0]:+.1f}, {primary['paired_bootstrap_95_ci_pp'][1]:+.1f}]pp
- task-stratified bootstrap 95% CI: [{primary['task_stratified_bootstrap_95_ci_pp'][0]:+.1f}, {primary['task_stratified_bootstrap_95_ci_pp'][1]:+.1f}]pp
- McNemar exact p: {primary['mcnemar_exact_p']:.6g}
- CFG rescue / harm: {primary['rescued']} / {primary['harmed']}
- CFG 正向 / 持平 / 负向任务: {positive} / {ties} / {negative}
- CFG - Midpoint: {comparisons['CFG_minus_Midpoint']['effect_pp']:+.1f}pp
- Weak - Strong: {comparisons['Weak_minus_Strong']['effect_pp']:+.1f}pp
- Midpoint - Strong: {comparisons['Midpoint_minus_Strong']['effect_pp']:+.1f}pp

## 逐任务成功数（/10）

| Task | Strong | Weak | Midpoint | CFG | CFG-Strong |
|---:|---:|---:|---:|---:|---:|
{table}

## 解释边界

固定 Strong=15k、Weak=10k、lambda=0.5、trust region=0.25；没有重训、换 pair、调 lambda、挑任务或按结果补跑。由于进入四臂前的 quality-gap gate 已失败，即使观察到正差，也只能作为后续独立确认的依据，不能包装为已建立的一般收益。

## 完整性

400/400 episodes 完成且动作有限；100/100 单元均包含四臂，初始 simulator state、prepared input 和首个 flow noise hash 严格一致。
"""
    (ARTIFACT / "exploratory_four_arm_report.md").write_text(report)
    status = {
        "status": "complete",
        "phase": "exploratory_four_arm_after_user_override",
        "decision": decision,
        "episodes_complete": 400,
        "episodes_planned": 400,
        "integrity": "PASS",
    }
    (ARTIFACT / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    checksum_paths = sorted(path for path in ARTIFACT.rglob("*") if path.is_file() and path.name != "final_artifacts.sha256")
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ARTIFACT)}" for path in checksum_paths]
    (ARTIFACT / "final_artifacts.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
