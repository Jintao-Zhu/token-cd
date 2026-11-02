#!/usr/bin/env python3
"""Status report using the STRICTLY COMMON batch (cells covered by every arm).

Prints: progress/rate, per-arm nets on (a) each arm's own coverage and (b) the common batch,
then a per-task net matrix restricted to the common batch, then the ETA.
"""
from __future__ import annotations

import collections
import csv
import math
import statistics
import time
from pathlib import Path

STUDY = Path("artifacts/openvla_budget_rule_9task_200_299_v1")
REF = Path("artifacts/openvla_lambda1_extension_200_299_v1")
TASK_SEC = {"google_robot_place_apple_in_closed_top_drawer": 214.4, "google_robot_open_drawer": 175.6,
            "google_robot_close_drawer": 194.6, "widowx_put_eggplant_in_basket": 101.8,
            "google_robot_move_near": 69.9, "google_robot_pick_coke_can": 68.5,
            "widowx_carrot_on_plate": 49.1, "widowx_stack_cube": 47.4, "widowx_spoon_on_towel": 45.1}
ARM_COST = {"top_p075_l100": 1.00, "top_p080_l100": 1.05, "top_p085_l100": 1.10, "top_p090_l100": 1.20,
            "fixed016_l100": 0.80, "fixed024_l100": 0.85, "fixed032_l100": 0.95, "fixed040_l100": 1.00,
            "fixed048_l100": 1.05, "fixed056_l100": 1.15, "fixed064_l100": 1.30}


def mcn(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run, m = [0.0] * len(ps), 0.0, len(ps)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (m - rank)))
        out[i] = run
    return out


def main() -> int:
    study = collections.defaultdict(dict)
    mts = collections.defaultdict(list)
    mt = {}
    for r in csv.DictReader(open(STUDY / "INDEX.tsv"), delimiter="\t"):
        k = (r["task"], int(r["seed"]))
        study[r["arm"]][k] = r["success"] == "1"
        mts[r["arm"]].append(float(r["mean_m_t"]))
        mt[(r["arm"],) + k] = float(r["mtime"])
    ref = {}
    for r in csv.DictReader(open(REF / "INDEX.tsv"), delimiter="\t"):
        ref[(r["task"], int(r["seed"]))] = r["success"] == "1"

    arms = sorted(study)
    total = sum(len(v) for v in study.values())
    now = time.time()
    r10 = sum(1 for v in mt.values() if now - v <= 600)
    print(f"完成 {total}/9900 ({100*total/9900:.1f}%) | 最近10min {r10*6} eps/h")

    common = None
    for a in arms:
        s = set(study[a])
        common = s if common is None else (common & s)
    tasks = sorted({k[0] for k in common})
    print(f"共同覆盖 cell: {len(common)} ({', '.join(t.replace('google_robot_','').replace('widowx_','') for t in tasks)})")

    ps = []
    for a in arms:
        cp = [(k, v) for k, v in study[a].items() if k in common and k in ref]
        b = sum(1 for k, v in cp if v and not ref[k]); c = sum(1 for k, v in cp if ref[k] and not v)
        ps.append(mcn(b, c))
    adj = holm(ps)
    print("\n| arm | 平均m_t | 共同批净 | 救/坏 | p | Holm | 各臂自身覆盖净 |")
    print("|---|---|---|---|---|---|---|")
    for a, p, h in zip(arms, ps, adj):
        cp = [(k, v) for k, v in study[a].items() if k in common and k in ref]
        b = sum(1 for k, v in cp if v and not ref[k]); c = sum(1 for k, v in cp if ref[k] and not v)
        allp = [(k, v) for k, v in study[a].items() if k in ref]
        ab = sum(1 for k, v in allp if v and not ref[k]); ac = sum(1 for k, v in allp if ref[k] and not v)
        print(f"| {a.replace('_l100','')} | {statistics.mean(mts[a]):.1f} | **{b-c:+d}** | {b}/{c} | "
              f"{p:.3f} | {h:.3f} | {ab-ac:+d} ({len(allp)}) |")

    print("\n| arm | " + " | ".join(t.replace('google_robot_','').replace('widowx_','')[:10] for t in tasks) + " |")
    print("|---" * (len(tasks) + 1) + "|")
    for a in arms:
        cells = []
        for t in tasks:
            cp = [(k, v) for k, v in study[a].items() if k[0] == t and k in common and k in ref]
            b = sum(1 for k, v in cp if v and not ref[k]); c = sum(1 for k, v in cp if ref[k] and not v)
            cells.append(f"{b-c:+d}" if cp else "—")
        print(f"| {a.replace('_l100','')} | " + " | ".join(cells) + " |")

    have = collections.Counter(t for (t, s, a) in ((k[0], k[1], a) for a in arms for k in study[a]))
    print("\n各任务已完成:", {t.replace('google_robot_', 'g_').replace('widowx_', 'w_'): have[t] for t in TASK_SEC})
    rem = sum(max(0, 100 - sum(1 for k in study[a] if k[0] == t)) * TASK_SEC[t] * ARM_COST[a]
              for a in ARM_COST for t in TASK_SEC)
    print(f"剩余 {9900-total} 集 ≈ {rem/3600:.0f} worker-h → 12 worker ≈ {rem/3600/12 + 1.9:.1f} h 后收尾")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
