#!/usr/bin/env python3
"""Full interim results for the OpenVLA budget-rule study (common-batch view).

Writes three tables to stdout:
  A. per-arm: mean m_t, success on the common batch, paired net vs Matched, McNemar p, Holm p
  B. per-arm x per-task paired net (common batch only)
  C. per-arm x per-task absolute success counts (common batch only)
"""
from __future__ import annotations

import collections
import csv
import math
import statistics
from pathlib import Path

STUDY = Path("artifacts/openvla_budget_rule_9task_200_299_v1/INDEX.tsv")
REF = Path("artifacts/openvla_lambda1_extension_200_299_v1/INDEX.tsv")
CONTROL = "lambda_100"
ORDER = ["google_robot_close_drawer", "google_robot_open_drawer",
         "google_robot_place_apple_in_closed_top_drawer", "widowx_put_eggplant_in_basket",
         "google_robot_move_near", "google_robot_pick_coke_can",
         "widowx_carrot_on_plate", "widowx_spoon_on_towel", "widowx_stack_cube"]


def mcn(b, c):
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
    for r in csv.DictReader(open(STUDY), delimiter="\t"):
        study[r["arm"]][(r["task"], int(r["seed"]))] = r["success"] == "1"
        mts[r["arm"]].append(float(r["mean_m_t"]))
    ref = {}
    for r in csv.DictReader(open(REF), delimiter="\t"):
        ref[(r["task"], int(r["seed"]))] = r["success"] == "1"

    arms = [a for a in sorted(study) if a != CONTROL]
    total = sum(len(v) for v in study.values())
    common = None
    for a in arms:
        s = set(study[a])
        common = s if common is None else (common & s)
    tasks = [t for t in ORDER if any(k[0] == t for k in common)]
    print(f"总集数 {total}/9900 | 共同批 {len(common)} cell | 任务: "
          + ", ".join(t.replace('google_robot_', '').replace('widowx_', '') for t in tasks))
    print(f"控制臂 {CONTROL}（matched 复现）: {len(study[CONTROL])} 集\n")

    ps = []
    for a in arms:
        p = [(k, v) for k, v in study[a].items() if k in common and k in ref]
        b = sum(1 for k, v in p if v and not ref[k])
        c = sum(1 for k, v in p if ref[k] and not v)
        ps.append(mcn(b, c))
    adj = holm(ps)

    print("## A. 每臂（共同批，同 seed 配对 vs Matched）")
    print("| arm | 平均 m_t | n | 成功 | 成功率 | 净 | 救/坏 | p | Holm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a, pv, h in zip(arms, ps, adj):
        p = [(k, v) for k, v in study[a].items() if k in common and k in ref]
        b = sum(1 for k, v in p if v and not ref[k]); c = sum(1 for k, v in p if ref[k] and not v)
        s = sum(1 for k, v in p if v)
        print(f"| {a.replace('_l100','')} | {statistics.mean(mts[a]):.1f} | {len(p)} | {s} | "
              f"{100*s/len(p):.1f}% | **{b-c:+d}** | {b}/{c} | {pv:.4f} | {h:.3f} |")
    rp = [(k, v) for k, v in ref.items() if k in common]
    print(f"| **Matched（参考）** | 36.7 | {len(rp)} | {sum(1 for k,v in rp if v)} | "
          f"{100*sum(1 for k,v in rp if v)/len(rp):.1f}% | — | — | — | — |")

    print("\n## B. 逐任务配对净收益（共同批）")
    print("| arm | " + " | ".join(t.replace('google_robot_','').replace('widowx_','')[:11] for t in tasks) + " |")
    print("|---" * (len(tasks) + 1) + "|")
    for a in arms:
        cells = []
        for t in tasks:
            p = [(k, v) for k, v in study[a].items() if k[0] == t and k in common and k in ref]
            b = sum(1 for k, v in p if v and not ref[k]); c = sum(1 for k, v in p if ref[k] and not v)
            cells.append(f"{b-c:+d}" if p else "—")
        print(f"| {a.replace('_l100','')} | " + " | ".join(cells) + " |")

    print("\n## C. 逐任务绝对成功率（共同批；该臂成功数/该任务共同批对数）")
    print("| arm | " + " | ".join(t.replace('google_robot_','').replace('widowx_','')[:11] for t in tasks) + " |")
    print("|---" * (len(tasks) + 1) + "|")
    ref_cells = []
    for t in tasks:
        p = [(k, v) for k, v in ref.items() if k[0] == t and k in common]
        ref_cells.append(f"{sum(1 for k,v in p if v)}/{len(p)}")
    print("| **Matched（参考）** | " + " | ".join(ref_cells) + " |")
    for a in arms:
        cells = []
        for t in tasks:
            p = [(k, v) for k, v in study[a].items() if k[0] == t and k in common and k in ref]
            cells.append(f"{sum(1 for k,v in p if v)}/{len(p)}" if p else "—")
        print(f"| {a.replace('_l100','')} | " + " | ".join(cells) + " |")

    have = collections.Counter(t for a in arms for (t, s) in study[a])
    print("\n各任务进度: " + ", ".join(f"{t.replace('google_robot_','').replace('widowx_','')}={have[t]}/1100"
                                   for t in ORDER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
