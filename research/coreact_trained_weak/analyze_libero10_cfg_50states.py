from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ARTIFACT = Path("artifacts/coreact_trained_weak_libero10_cfg_50states_v1_20260812_230341")
ARMS = ("Strong_15k", "Weak_10k", "Midpoint", "CFG")


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def comparison(rows: list[dict], left: str, right: str, rng: np.random.Generator) -> dict:
    by_task = np.asarray([
        [int(rows[task * 50 + init][left]) - int(rows[task * 50 + init][right]) for init in range(50)]
        for task in range(10)
    ], dtype=np.int8)
    flat = by_task.reshape(-1)
    paired, stratified = np.empty(100_000), np.empty(100_000)
    for start in range(0, 100_000, 2_000):
        paired_idx = rng.integers(0, 500, (2_000, 500))
        paired[start:start + 2_000] = flat[paired_idx].mean(axis=1) * 100
        task_idx = rng.integers(0, 50, (2_000, 10, 50))
        sampled = np.take_along_axis(by_task[None, :, :], task_idx, axis=2)
        stratified[start:start + 2_000] = sampled.mean(axis=(1, 2)) * 100
    rescued, harmed = int((flat == 1).sum()), int((flat == -1).sum())
    discordant = rescued + harmed
    return {
        "effect_pp": float(flat.mean() * 100),
        "paired_bootstrap_95_ci_pp": ci(paired),
        "task_stratified_bootstrap_95_ci_pp": ci(stratified),
        "mcnemar_exact_p": float(binomtest(min(rescued, harmed), discordant).pvalue) if discordant else 1.0,
        "rescued": rescued,
        "harmed": harmed,
    }


def main() -> None:
    paths = sorted((ARTIFACT / "episodes").glob("*.json"))
    if len(paths) != 2_000:
        raise RuntimeError(f"expected 2000 episodes, got {len(paths)}")
    episodes = [json.loads(path.read_text()) for path in paths]
    units: dict[tuple[int, int], dict[str, dict]] = {}
    for episode in episodes:
        if episode.get("status") != "complete" or episode.get("all_actions_finite") is not True:
            raise RuntimeError(f"invalid episode: {episode.get('episode_id')}")
        key = (int(episode["task_id"]), int(episode["init_state_id"]))
        if episode["arm"] in units.setdefault(key, {}):
            raise RuntimeError(f"duplicate arm for {key}: {episode['arm']}")
        units[key][episode["arm"]] = episode
    expected = [(task, init) for task in range(10) for init in range(50)]
    if sorted(units) != expected:
        raise RuntimeError("task/init coverage mismatch")

    rows, failures = [], []
    for key in expected:
        group = units[key]
        if set(group) != set(ARMS):
            failures.append(f"{key}: arms")
            continue
        for field in ("initial_sim_state_sha256", "initial_prepared_input_sha256"):
            if len({group[arm][field] for arm in ARMS}) != 1:
                failures.append(f"{key}: {field}")
        if len({group[arm]["noise_sha256_by_replan"][0] for arm in ARMS}) != 1:
            failures.append(f"{key}: first_noise")
        rows.append({"task_id": key[0], "init_state_id": key[1], **{arm: bool(group[arm]["success"]) for arm in ARMS}})
    if failures:
        raise RuntimeError("paired audit failed: " + ", ".join(failures[:10]))

    rates = {arm: sum(row[arm] for row in rows) / 500 for arm in ARMS}
    task_rows = []
    for task in range(10):
        subset = [row for row in rows if row["task_id"] == task]
        counts = {arm: sum(row[arm] for row in subset) for arm in ARMS}
        task_rows.append({"task_id": task, **counts, "CFG_minus_Strong_pp": (counts["CFG"] - counts["Strong_15k"]) * 2.0})
    rng = np.random.default_rng(1729)
    comparisons = {
        "CFG_minus_Strong": comparison(rows, "CFG", "Strong_15k", rng),
        "CFG_minus_Midpoint": comparison(rows, "CFG", "Midpoint", rng),
        "Weak_minus_Strong": comparison(rows, "Weak_10k", "Strong_15k", rng),
        "Midpoint_minus_Strong": comparison(rows, "Midpoint", "Strong_15k", rng),
    }
    deltas = [row["CFG_minus_Strong_pp"] for row in task_rows]
    signs = [sum(x > 0 for x in deltas), sum(x == 0 for x in deltas), sum(x < 0 for x in deltas)]
    primary = comparisons["CFG_minus_Strong"]
    benefit = primary["effect_pp"] >= 5 and primary["paired_bootstrap_95_ci_pp"][0] > 0
    decision = "LIBERO10_50STATE_CFG_BENEFIT" if benefit else "LIBERO10_50STATE_CFG_NO_RELIABLE_BENEFIT"
    analysis = {
        "decision": decision,
        "interpretation": "exploratory_50_state_extension_reusing_first_10_states",
        "episodes": 2_000,
        "matched_units": 500,
        "success_rates": rates,
        "comparisons": comparisons,
        "tasks_CFG_positive_tie_negative_vs_Strong": signs,
        "catastrophic_task_harm_gt_10pp": any(x < -10 for x in deltas),
        "lambda": 0.5,
        "trust_region_kappa": 0.25,
        "posthoc_tuning": False,
    }
    (ARTIFACT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (ARTIFACT / "decision.json").write_text(json.dumps({"decision": decision, "analysis": analysis}, indent=2) + "\n")
    integrity = {
        "status": "PASS", "episodes": "2000/2000", "matched_units": "500/500",
        "arms": list(ARMS), "complete_and_finite": "2000/2000",
        "initial_sim_state_hash_match": "500/500",
        "initial_prepared_input_hash_match": "500/500", "first_noise_hash_match": "500/500",
    }
    (ARTIFACT / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    with (ARTIFACT / "task_success.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_rows[0].keys())
        writer.writeheader(); writer.writerows(task_rows)
    table = "\n".join(
        f"| {r['task_id']} | {r['Strong_15k']}/50 | {r['Weak_10k']}/50 | {r['Midpoint']}/50 | {r['CFG']}/50 | {r['CFG_minus_Strong_pp']:+.0f}pp |"
        for r in task_rows
    )
    report = f"""# LIBERO-10 Trained-Weak CFG 50-State Extension

## Decision

`{decision}`

| Arm | Success |
|---|---:|
| Strong 15k | {rates['Strong_15k']:.1%} |
| Weak 10k | {rates['Weak_10k']:.1%} |
| Midpoint | {rates['Midpoint']:.1%} |
| CFG (lambda=0.5) | {rates['CFG']:.1%} |

- CFG - Strong: {primary['effect_pp']:+.1f}pp
- paired bootstrap 95% CI: [{primary['paired_bootstrap_95_ci_pp'][0]:+.1f}, {primary['paired_bootstrap_95_ci_pp'][1]:+.1f}]pp
- task-stratified bootstrap 95% CI: [{primary['task_stratified_bootstrap_95_ci_pp'][0]:+.1f}, {primary['task_stratified_bootstrap_95_ci_pp'][1]:+.1f}]pp
- McNemar exact p: {primary['mcnemar_exact_p']:.6g}
- rescue / harm: {primary['rescued']} / {primary['harmed']}
- positive / tie / negative tasks: {signs[0]} / {signs[1]} / {signs[2]}
- CFG - Midpoint: {comparisons['CFG_minus_Midpoint']['effect_pp']:+.1f}pp

## Per-task successes

| Task | Strong | Weak | Midpoint | CFG | CFG-Strong |
|---:|---:|---:|---:|---:|---:|
{table}

## Integrity and scope

2000/2000 episodes complete and finite. All 500 four-arm units match exactly on initial simulator state, prepared input, and first flow noise. Strong=15k, Weak=10k, lambda=0.5, trust region=0.25; no retraining, tuning, task selection, or outcome-driven rerun. Init states 0-9 reuse the earlier locked run; init states 10-49 are new, so this is a larger exploratory extension rather than an independent confirmation.
"""
    (ARTIFACT / "report.md").write_text(report)
    (ARTIFACT / "status.json").write_text(json.dumps({"status": "complete", "episodes_complete": 2000, "episodes_planned": 2000, "decision": decision, "integrity": "PASS"}, indent=2) + "\n")
    files = sorted(
        path for path in ARTIFACT.rglob("*")
        if path.is_file() and path.name != "final_artifacts.sha256" and "logs" not in path.relative_to(ARTIFACT).parts
    )
    (ARTIFACT / "final_artifacts.sha256").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ARTIFACT)}" for p in files) + "\n")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
