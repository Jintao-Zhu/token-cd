#!/usr/bin/env python3
"""Fast status / result report for the OpenVLA budget-rule study (reads INDEX.tsv, no JSON parsing).

  python3 scripts/report_openvla_budget_rule.py [--study ...] [--ref ...] [--min-pairs 20]

Columns read from the index: task, seed, arm, success, steps, runtime_seconds, mean_m_t,
budget_param, mtime.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics
import time
from pathlib import Path

TASKS = [
    "google_robot_place_apple_in_closed_top_drawer", "google_robot_open_drawer",
    "google_robot_close_drawer", "google_robot_pick_coke_can", "google_robot_move_near",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
]
ARM_COST = {"top_p075_l100": 1.00, "top_p080_l100": 1.05, "top_p085_l100": 1.10, "top_p090_l100": 1.20,
            "fixed016_l100": 0.80, "fixed024_l100": 0.85, "fixed032_l100": 0.95, "fixed040_l100": 1.00,
            "fixed048_l100": 1.05, "fixed056_l100": 1.15, "fixed064_l100": 1.30}
TASK_SEC = {"google_robot_place_apple_in_closed_top_drawer": 214.4, "google_robot_open_drawer": 175.6,
            "google_robot_close_drawer": 194.6, "widowx_put_eggplant_in_basket": 101.8,
            "google_robot_move_near": 69.9, "google_robot_pick_coke_can": 68.5,
            "widowx_carrot_on_plate": 49.1, "widowx_stack_cube": 47.4, "widowx_spoon_on_towel": 45.1}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, run, m = [0.0] * len(pvals), 0.0, len(pvals)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, pvals[i] * (m - rank)))
        out[i] = run
    return out


def load(path: Path) -> dict[tuple, dict]:
    rows = {}
    with path.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[(r["task"], int(r["seed"]), r["arm"])] = {
                "success": r["success"] == "1", "runtime": float(r["runtime_seconds"]),
                "m_t": float(r["mean_m_t"]), "mtime": float(r["mtime"])}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="artifacts/openvla_budget_rule_9task_200_299_v1/INDEX.tsv")
    ap.add_argument("--ref", default="artifacts/openvla_lambda1_extension_200_299_v1/INDEX.tsv")
    ap.add_argument("--min-pairs", type=int, default=20)
    args = ap.parse_args()

    study = load(Path(args.study))
    ref = load(Path(args.ref))
    ref_s = {(t, s): d for (t, s, a), d in ref.items()}
    by_arm = collections.defaultdict(dict)
    for (t, s, a), d in study.items():
        by_arm[a][(t, s)] = d

    now = time.time()
    r10 = sum(1 for d in study.values() if now - d["mtime"] <= 600)
    print(f"完成 {len(study)}/9900 ({100*len(study)/9900:.1f}%) | 最近10min {r10*6} eps/h")

    pvals, order = [], []
    for a in sorted(by_arm):
        p = [(k, d) for k, d in by_arm[a].items() if k in ref_s]
        b = sum(1 for k, d in p if d["success"] and not ref_s[k]["success"])
        c = sum(1 for k, d in p if ref_s[k]["success"] and not d["success"])
        pvals.append(mcnemar_exact(b, c)); order.append((a, len(by_arm[a]), p, b, c))
    adj = {a: h for (a, _, _, _, _), h in zip(order, holm(pvals))}

    print("\n| arm | 集数 | 成功率 | 平均m_t | 配对 | 净 | 救/坏 | p | Holm p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a, n, p, b, c in order:
        v = by_arm[a]
        s = sum(1 for d in v.values() if d["success"])
        mt = statistics.mean(d["m_t"] for d in v.values())
        print(f"| {a} | {n} | {100*s/n:.1f}% | {mt:.1f} | {len(p)} | {b-c:+d} | {b}/{c} | "
              f"{mcnemar_exact(b,c):.3f} | {adj[a]:.3f} |")

    print(f"\n| arm | 任务 | 对数 | 该臂 | matched | 净 | (≥{args.min_pairs} 对)")
    print("|---|---|---|---|---|---|")
    for a, _, _, _, _ in order:
        for t in TASKS:
            p = [(k, d) for k, d in by_arm[a].items() if k[0] == t and k in ref_s]
            if len(p) < args.min_pairs:
                continue
            b = sum(1 for k, d in p if d["success"] and not ref_s[k]["success"])
            c = sum(1 for k, d in p if ref_s[k]["success"] and not d["success"])
            sa = sum(1 for k, d in p if d["success"]); sr = sum(1 for k, d in p if ref_s[k]["success"])
            print(f"| {a} | {t.replace('google_robot_','').replace('widowx_','')} | {len(p)} | "
                  f"{sa}/{len(p)} | {sr}/{len(p)} | {b-c:+d} |")

    have = collections.Counter((t) for (t, s, a) in study)
    print("\n各任务已完成:", {t.replace('google_robot_', 'g_').replace('widowx_', 'w_'): have[t] for t in TASKS})
    rem = 0.0
    for a in ARM_COST:
        for t in TASK_SEC:
            n = sum(1 for (tt, ss, aa) in study if aa == a and tt == t)
            rem += max(0, 100 - n) * TASK_SEC[t] * ARM_COST[a]
    print(f"剩余 {9900-len(study)} 集 ≈ {rem/3600:.0f} worker-h → 12 worker ≈ {rem/3600/12 + 1.9:.1f} h 后收尾")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
