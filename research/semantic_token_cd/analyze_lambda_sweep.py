"""Aggregate the Attention-CD lambda (intervention-strength) sweep results.

Per task: per-arm success rate + vanilla vs each lambda arm Rescue/Harm cross-tab
and exact McNemar p (scipy.stats.binomtest on discordant pairs), sorted by lambda.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from research.semantic_token_cd.attn_semantic_lambda_sweep import ARMS, ARM_LAMBDAS


def arm_sort_key(arm: str):
    if arm == "vanilla":
        return (-1.0,)
    return (ARM_LAMBDAS[arm],)


def load_arm(task_root: Path, arm: str) -> dict[int, bool]:
    out: dict[int, bool] = {}
    arm_dir = task_root / arm
    for f in sorted(arm_dir.glob("episode_*_summary.json")):
        seed = int(f.name.split("_")[1])
        out[seed] = bool(json.loads(f.read_text())["success"])
    return out


def mcnemar(a: dict[int, bool], b: dict[int, bool], seeds) -> dict:
    rescue = harm = both_ok = both_fail = 0
    for seed in seeds:
        a_ok, b_ok = a[seed], b[seed]
        if not a_ok and b_ok:
            rescue += 1
        elif a_ok and not b_ok:
            harm += 1
        elif a_ok and b_ok:
            both_ok += 1
        else:
            both_fail += 1
    disc = rescue + harm
    p = binomtest(rescue, disc, 0.5).pvalue if disc > 0 else 1.0
    return {
        "rescue": rescue,
        "harm": harm,
        "both_ok": both_ok,
        "both_fail": both_fail,
        "discordant": disc,
        "mcnemar_p": float(p),
    }


def report_task(task_root: Path, task: str) -> dict:
    per_arm = {arm: load_arm(task_root, arm) for arm in ARMS}
    complete = sorted(set.intersection(*(set(per_arm[a]) for a in ARMS)))
    n = len(complete)
    if n == 0:
        print(f"=== {task}: no complete seeds yet ===")
        return {"task": task, "n_complete": 0}

    wins = {arm: sum(1 for s in complete if per_arm[arm][s]) for arm in ARMS}
    vanilla = per_arm["vanilla"]
    rows = []
    for arm in sorted(ARMS, key=arm_sort_key):
        row = {
            "arm": arm,
            "lambda": ARM_LAMBDAS[arm],
            "success": wins[arm],
            "rate": round(wins[arm] / n, 4),
        }
        if arm != "vanilla":
            row.update(mcnemar(vanilla, per_arm[arm], complete))
        rows.append(row)

    print(f"\n=== {task}  (n={n}/100 complete seeds) ===")
    print(f"{'arm':11s} {'lambda':>6s} {'succ':>7s} {'rescue':>7s} {'harm':>6s} {'mcnemar_p':>10s}")
    for r in rows:
        lam = "-" if r["lambda"] is None else f"{r['lambda']:g}"
        if r["arm"] == "vanilla":
            print(f"{r['arm']:11s} {lam:>6s} {r['success']:>3d}/{n:<3d} {'-':>7s} {'-':>6s} {'-':>10s}")
        else:
            print(
                f"{r['arm']:11s} {lam:>6s} {r['success']:>3d}/{n:<3d} "
                f"{r['rescue']:>7d} {r['harm']:>6d} {r['mcnemar_p']:>10.4g}"
            )
    return {"task": task, "n_complete": n, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    eps = root / "episodes"
    if eps.exists():
        for task_dir in sorted(eps.iterdir()):
            if task_dir.is_dir():
                report_task(task_dir, task_dir.name)


if __name__ == "__main__":
    main()
