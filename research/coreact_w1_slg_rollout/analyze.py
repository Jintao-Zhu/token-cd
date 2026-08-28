from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from research.coreact_w1_slg_rollout.sampler import ARMS


def paired_stats(rows, left, right, rng):
    values = np.asarray([int(row[left]) - int(row[right]) for row in rows], dtype=np.int8)
    bootstrap = np.empty(10_000)
    for start in range(0, 10_000, 1000):
        indices = rng.integers(0, len(values), (1000, len(values)))
        bootstrap[start:start + 1000] = values[indices].mean(axis=1) * 100
    rescue, harm = int((values == 1).sum()), int((values == -1).sum())
    discordant = rescue + harm
    return {
        "difference_pp": float(values.mean() * 100),
        "paired_bootstrap_95_ci_pp": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
        "rescue": rescue, "harm": harm,
        "mcnemar_exact_p": float(binomtest(min(rescue, harm), discordant).pvalue) if discordant else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    paths = sorted((artifact / "episodes").glob("*.json"))
    invalid = sorted((artifact / "invalid_pairs").glob("*.json"))
    if len(paths) != 2000 or invalid:
        raise RuntimeError(f"requires 2000 episodes and zero invalid pairs; got {len(paths)}, {len(invalid)}")
    groups = {}
    for path in paths:
        episode = json.loads(path.read_text())
        if episode["status"] != "complete" or not episode["all_actions_finite"] or episode["paired_initial_integrity"] != "PASS":
            raise RuntimeError(f"invalid episode: {path.name}")
        groups.setdefault(episode["pair_id"], {})[episode["arm"]] = episode
    if len(groups) != 500 or any(set(group) != set(ARMS) for group in groups.values()):
        raise RuntimeError("paired coverage failed")
    rows = []
    for pair_id in sorted(groups):
        group = groups[pair_id]
        initial_fields = ("instruction", "noise_sha256", "strong_velocity_sha256")
        for field in initial_fields:
            if len({group[arm]["initial_integrity"][field] for arm in ARMS}) != 1:
                raise RuntimeError(f"{pair_id}: {field}")
        rows.append({"pair_id": pair_id, "task_id": group["strong"]["task_id"], **{arm: bool(group[arm]["success"]) for arm in ARMS}})
    rates = {arm: sum(row[arm] for row in rows) / 500 for arm in ARMS}
    rng = np.random.default_rng(20260817)
    comparisons = {
        "low_minus_strong": paired_stats(rows, "low_w1", "strong", rng),
        "low_minus_full": paired_stats(rows, "low_w1", "full_w1", rng),
        "low_minus_high": paired_stats(rows, "low_w1", "high_w1", rng),
        "full_minus_strong": paired_stats(rows, "full_w1", "strong", rng),
        "high_minus_strong": paired_stats(rows, "high_w1", "strong", rng),
    }
    task_rows = []
    for task in range(10):
        subset = [row for row in rows if row["task_id"] == task]
        task_rows.append({"task": task, **{arm: sum(row[arm] for row in subset) for arm in ARMS}})
    primary = comparisons["low_minus_strong"]
    tasks_nonworse = sum(row["low_w1"] >= row["strong"] for row in task_rows)
    go = primary["difference_pp"] >= 5 and primary["paired_bootstrap_95_ci_pp"][0] > 0 and primary["rescue"] > primary["harm"] and tasks_nonworse >= 7
    if go:
        decision = "W1_LOW_NOISE_SLG_CLOSED_LOOP_CONFIRMED"
    elif primary["difference_pp"] > 0 and primary["paired_bootstrap_95_ci_pp"][0] <= 0 <= primary["paired_bootstrap_95_ci_pp"][1]:
        decision = "W1_LOW_NOISE_ROLLOUT_INCONCLUSIVE"
    else:
        decision = "W1_LOW_NOISE_OFFLINE_SIGNAL_NO_CLOSED_LOOP_GAIN"
    analysis = {"decision": decision, "rates": rates, "comparisons": comparisons, "tasks_nonworse_low_vs_strong": tasks_nonworse, "task_rows": task_rows, "episodes": 2000, "pairs": 500}
    (artifact / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"decision": decision}, indent=2) + "\n")
    with (artifact / "task_success.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=task_rows[0].keys()); writer.writeheader(); writer.writerows(task_rows)
    arm_rows = []
    for arm in ARMS:
        comparison = comparisons.get(f"{arm.split('_')[0]}_minus_strong") if arm != "strong" else None
        arm_rows.append(f"| {arm} | {sum(row[arm] for row in rows)}/500 ({rates[arm]:.1%}) | {comparison['difference_pp']:+.1f}pp | {comparison['rescue']} | {comparison['harm']} |" if comparison else f"| {arm} | {sum(row[arm] for row in rows)}/500 ({rates[arm]:.1%}) | - | - | - |")
    task_table = "\n".join(f"| {r['task']} | {r['strong']}/50 | {r['low_w1']}/50 | {r['full_w1']}/50 | {r['high_w1']}/50 |" for r in task_rows)
    report = f"""# W1 Low-Noise SLG Closed-Loop Success Validation

## Decision

`{decision}`

| Arm | Success | vs Strong | Rescue | Harm |
|---|---:|---:|---:|---:|
{chr(10).join(arm_rows)}

Primary Low-W1 minus Strong: {primary['difference_pp']:+.1f}pp, paired bootstrap 95% CI [{primary['paired_bootstrap_95_ci_pp'][0]:+.1f}, {primary['paired_bootstrap_95_ci_pp'][1]:+.1f}]pp, rescue/harm {primary['rescue']}/{primary['harm']}, exact McNemar p={primary['mcnemar_exact_p']:.6g}. Tasks non-worse: {tasks_nonworse}/10.

Low-W1 minus Full-W1: {comparisons['low_minus_full']['difference_pp']:+.1f}pp. Low-W1 minus High-W1: {comparisons['low_minus_high']['difference_pp']:+.1f}pp.

| Task | Strong | Low-W1 | Full-W1 | High-W1 |
|---:|---:|---:|---:|---:|
{task_table}

Integrity: 2000/2000 complete finite episodes, 500/500 four-arm paired units, no invalid or outcome-deleted episodes. W1, lambda=0.5, trust region=0.25, low steps 8-9, high steps 0-1, H50/execute10 and 520-step horizon remained frozen.
"""
    (artifact / "REPORT.md").write_text(report)
    (artifact / "status.json").write_text(json.dumps({"status": "complete", "episodes_complete": 2000, "episodes_planned": 2000, "decision": decision}, indent=2) + "\n")
    files = sorted(path for path in artifact.rglob("*") if path.is_file() and "logs" not in path.parts and path.name != "final_artifacts.sha256")
    (artifact / "final_artifacts.sha256").write_text("\n".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(artifact)}" for path in files) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
