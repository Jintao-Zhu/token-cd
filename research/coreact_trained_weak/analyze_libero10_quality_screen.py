from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ARMS = ("Strong_15k", "Weak_10k")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    paths = sorted((artifact / "episodes").glob("*.json"))
    if len(paths) != 200:
        raise RuntimeError(f"expected 200 episodes, got {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    if any(row["status"] != "complete" or not row["all_actions_finite"] for row in rows):
        raise RuntimeError("episode integrity failure")
    for task in range(10):
        for init in range(10):
            pair = [row for row in rows if row["task_id"] == task and row["init_state_id"] == init]
            if {row["arm"] for row in pair} != set(ARMS):
                raise RuntimeError(f"missing pair task={task} init={init}")
            for key in ("initial_sim_state_sha256", "initial_prepared_input_sha256"):
                if len({row[key] for row in pair}) != 1:
                    raise RuntimeError(f"paired {key} mismatch task={task} init={init}")
            if len({row["noise_sha256_by_replan"][0] for row in pair}) != 1:
                raise RuntimeError(f"paired noise mismatch task={task} init={init}")

    frame = pd.DataFrame(rows)
    wide = frame.pivot(index=["unit_id", "task_id", "init_state_id"], columns="arm", values="success").reset_index()
    strong_rate = float(wide.Strong_15k.mean())
    weak_rate = float(wide.Weak_10k.mean())
    gap = (strong_rate - weak_rate) * 100
    rescued = int(((wide.Strong_15k == True) & (wide.Weak_10k == False)).sum())
    reversed_units = int(((wide.Strong_15k == False) & (wide.Weak_10k == True)).sum())
    task_rows = []
    for task in range(10):
        subset = wide[wide.task_id == task]
        strong = int(subset.Strong_15k.sum())
        weak = int(subset.Weak_10k.sum())
        task_rows.append({"task_id": task, "Strong_15k": strong, "Weak_10k": weak, "gap_pp": strong * 10 - weak * 10})
    pd.DataFrame(task_rows).to_csv(artifact / "libero10_task_success.csv", index=False)
    nonnegative = sum(row["Strong_15k"] >= row["Weak_10k"] for row in task_rows)
    strong_range_pass = 0.40 <= strong_rate <= 0.75
    weak_floor_pass = weak_rate >= 0.20
    gap_pass = 8 <= gap <= 20
    cross_task_pass = nonnegative >= 7
    all_pass = strong_range_pass and weak_floor_pass and gap_pass and cross_task_pass
    decision = "LIBERO10_QUALITY_SCREEN_PASS_READY_FOR_EXPLORATORY_FOUR_ARM" if all_pass else "LIBERO10_QUALITY_SCREEN_NO_SUITABLE_PAIR_NO_FOUR_ARM"
    analysis = {
        "decision": decision,
        "episodes": 200,
        "matched_units": 100,
        "strong_success_rate": strong_rate,
        "weak_success_rate": weak_rate,
        "quality_gap_pp": gap,
        "strong_success_weak_failure_units": rescued,
        "strong_failure_weak_success_units": reversed_units,
        "task_nonnegative_count": nonnegative,
        "gates": {
            "strong_40_to_75_percent": strong_range_pass,
            "weak_at_least_20_percent": weak_floor_pass,
            "gap_8_to_20pp": gap_pass,
            "at_least_7_of_10_tasks_nonnegative": cross_task_pass,
            "all_pass": all_pass,
        },
        "paired_hash_audit": "PASS",
        "all_actions_finite": True,
    }
    (artifact / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"decision": decision}, indent=2) + "\n")
    table = "\n".join(
        f"| {row['task_id']} | {row['Strong_15k']} | {row['Weak_10k']} | {row['gap_pp']:+.0f}pp |"
        for row in task_rows
    )
    report = f"""# LIBERO-10 Trained-Weak Quality Screen

## Decision

`{decision}`

| Model | Success |
|---|---:|
| Strong 15k | {strong_rate:.0%} |
| Weak 10k | {weak_rate:.0%} |

- Quality gap: {gap:+.1f}pp
- Strong success / Weak failure: {rescued}
- Strong failure / Weak success: {reversed_units}
- Tasks Strong >= Weak: {nonnegative}/10

## Per-task success (/10)

| Task | Strong | Weak | Gap |
|---:|---:|---:|---:|
{table}

## Gates

- Strong in 40--75%: {'PASS' if strong_range_pass else 'FAIL'}
- Weak >=20%: {'PASS' if weak_floor_pass else 'FAIL'}
- Gap 8--20pp: {'PASS' if gap_pass else 'FAIL'}
- Strong >= Weak on >=7/10 tasks: {'PASS' if cross_task_pass else 'FAIL'}

This is an exploratory difficult-suite screen after the preregistered Spatial no-go. It does not alter the previous conclusion and does not automatically authorize a four-arm rollout.
"""
    (artifact / "report.md").write_text(report)
    status = {"status": decision, "episodes_complete": 200, "episodes_planned": 200, "analysis": analysis}
    (artifact / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(analysis))


if __name__ == "__main__":
    main()
