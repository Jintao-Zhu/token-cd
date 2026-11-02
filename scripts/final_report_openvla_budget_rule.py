#!/usr/bin/env python3
"""FINAL report for the OpenVLA budget-rule study (reads INDEX.tsv, seconds not minutes).

Sections
  1. coverage / integrity
  2. per-arm: mean m_t, success, paired net vs Matched, exact McNemar p, Holm over the 11 tests
  3. per-task paired net matrix + per-task absolute table
  4. success vs realized mean budget (Figure 6c data)
  5. family trends (Cochran-Armitage over the ordered TopP thresholds / fixed budgets)
  6. budget distribution per arm (min / median / max of the per-episode mean m_t)

Writes docs/OPENVLA_BUDGET_RULE_9TASK_FINAL.md and .json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
import time
from pathlib import Path

STUDY = Path("artifacts/openvla_budget_rule_9task_200_299_v1")
REF = Path("artifacts/openvla_lambda1_extension_200_299_v1")
CONTROL = "lambda_100"
TOPP = [("top_p075_l100", 0.75), ("top_p080_l100", 0.80), ("top_p085_l100", 0.85), ("top_p090_l100", 0.90)]
FIXED = [("fixed016_l100", 16), ("fixed024_l100", 24), ("fixed032_l100", 32), ("fixed040_l100", 40),
         ("fixed048_l100", 48), ("fixed056_l100", 56), ("fixed064_l100", 64)]
TASKS = ["google_robot_close_drawer", "google_robot_open_drawer",
         "google_robot_place_apple_in_closed_top_drawer", "widowx_put_eggplant_in_basket",
         "google_robot_move_near", "google_robot_pick_coke_can",
         "widowx_carrot_on_plate", "widowx_spoon_on_towel", "widowx_stack_cube"]
SHORT = lambda t: t.replace("google_robot_", "").replace("widowx_", "")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run, m = [0.0] * len(ps), 0.0, len(ps)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (m - rank)))
        out[i] = run
    return out


def cochran_armitage(groups):
    N = sum(n for _, n, _ in groups)
    X = sum(x for _, _, x in groups)
    if N == 0:
        return 0.0, 1.0
    pbar = X / N
    if pbar in (0.0, 1.0):
        return 0.0, 1.0
    s1 = sum(n * t for t, n, _ in groups)
    s2 = sum(n * t * t for t, n, _ in groups)
    T = sum(t * (x - n * pbar) for t, n, x in groups)
    var = pbar * (1 - pbar) * (s2 - s1 ** 2 / N)
    if var <= 0:
        return 0.0, 1.0
    z = T / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    study = collections.defaultdict(dict)
    mts = collections.defaultdict(list)
    rt = collections.defaultdict(list)
    for r in csv.DictReader(open(STUDY / "INDEX.tsv"), delimiter="\t"):
        study[r["arm"]][(r["task"], int(r["seed"]))] = r["success"] == "1"
        mts[r["arm"]].append(float(r["mean_m_t"]))
        rt[r["arm"]].append(float(r["runtime_seconds"]))
    ref = {}
    for r in csv.DictReader(open(REF / "INDEX.tsv"), delimiter="\t"):
        ref[(r["task"], int(r["seed"]))] = r["success"] == "1"

    arms = [a for a in sorted(study) if a != CONTROL]
    total = sum(len(v) for v in study.values())
    ref_s = sum(1 for v in ref.values() if v)
    out = {"study": str(STUDY), "reference": str(REF), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "episodes": total, "reference_episodes": len(ref), "arms": {}}
    L = []
    A = L.append
    A("# OpenVLA 预算策略对照实验（Matched vs Attention-TopP vs Fixed）· 9 任务最终结果")
    A("")
    A(f"*生成：{out['generated']}*  ·  *协议：* `OPENVLA_BUDGET_RULE_9TASK_200_299_V1`（λ=1.0，L11 prompt attention，"
      "harmonic 反事实，contrast 执行，projector 干预）")
    A("")
    A("## 1. 覆盖与完整性")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 新增 episode（11 臂） | **{total - len(study[CONTROL])} / 9900** |")
    A(f"| matched 参考集（复用） | 900 / 900（seeds 200–299） |")
    A(f"| 控制臂（matched 本机复现） | {len(study[CONTROL])} 集，与参考集逐位一致 |")
    A("| 渲染 | 逐任务锁定：close/open = rt-full depth 8；其余 7 任务 = RT depth 0（spp 32, denoiser on） |")
    A("| 切块 | 与参考集相同的 per-process 10-seed 块（复现性要求，见 FINDING 文档） |")
    A("")

    ps = []
    for a in arms:
        p = [(k, v) for k, v in study[a].items() if k in ref]
        b = sum(1 for k, v in p if v and not ref[k]); c = sum(1 for k, v in p if ref[k] and not v)
        ps.append(mcnemar_exact(b, c))
    adj = holm(ps)

    A("## 2. 每臂 vs Matched（九任务同批，n=900/臂）")
    A("")
    A("| arm | 平均 m_t | 成功 | 成功率 | 净 | 救/坏 | exact p | Holm p |")
    A("|---|---|---|---|---|---|---|---|")
    rows = []
    for a, pv, h in zip(arms, ps, adj):
        p = [(k, v) for k, v in study[a].items() if k in ref]
        b = sum(1 for k, v in p if v and not ref[k]); c = sum(1 for k, v in p if ref[k] and not v)
        s = sum(1 for k, v in p if v)
        row = {"n": len(p), "success": s, "rate": s / len(p), "net": b - c, "rescue": b, "harm": c,
               "p": pv, "holm": h, "mean_m_t": statistics.mean(mts[a]),
               "median_m_t": statistics.median(mts[a]), "min_m_t": min(mts[a]), "max_m_t": max(mts[a]),
               "mean_runtime_s": statistics.mean(rt[a])}
        out["arms"][a] = row
        rows.append((a, row))
        A(f"| {a.replace('_l100','')} | {row['mean_m_t']:.1f} | {s} | {100*s/len(p):.1f}% | "
          f"**{row['net']:+d}** | {b}/{c} | {pv:.4f} | {h:.3f} |")
    A(f"| **Matched（参考）** | 36.7 | {ref_s} | {100*ref_s/len(ref):.1f}% | — | — | — | — |")
    A("")

    A("## 3. 逐任务配对净收益")
    A("")
    A("| arm | " + " | ".join(SHORT(t)[:11] for t in TASKS) + " |")
    A("|---" * (len(TASKS) + 1) + "|")
    for a, _ in rows:
        cells = []
        for t in TASKS:
            p = [(k, v) for k, v in study[a].items() if k[0] == t and k in ref]
            b = sum(1 for k, v in p if v and not ref[k]); c = sum(1 for k, v in p if ref[k] and not v)
            cells.append(f"{b-c:+d}")
        A(f"| {a.replace('_l100','')} | " + " | ".join(cells) + " |")
    A("")
    A("## 4. 逐任务绝对成功率（成功/100）")
    A("")
    A("| arm | " + " | ".join(SHORT(t)[:11] for t in TASKS) + " |")
    A("|---" * (len(TASKS) + 1) + "|")
    A("| **Matched** | " + " | ".join(
        str(sum(1 for (tt, ss), v in ref.items() if tt == t and v)) for t in TASKS) + " |")
    for a, _ in rows:
        cells = []
        for t in TASKS:
            p = [(k, v) for k, v in study[a].items() if k[0] == t]
            cells.append(str(sum(1 for k, v in p if v)))
        A(f"| {a.replace('_l100','')} | " + " | ".join(cells) + " |")
    A("")

    A("## 5. Figure 6c 数据：成功率 vs 实测平均预算")
    A("")
    A("| 方法 | 平均 m_t | 成功率 |")
    A("|---|---|---|")
    A(f"| matched | 36.7 | {100*ref_s/len(ref):.1f}% |")
    for a, row in rows:
        A(f"| {a.replace('_l100','')} | {row['mean_m_t']:.1f} | {100*row['rate']:.1f}% |")
    A("")

    A("## 6. 族内趋势检验（Cochran-Armitage）")
    A("")
    A("| 家族 | 顺序 | z | p |")
    A("|---|---|---|---|")
    trends = {}
    for name, fam in (("Attention-TopP", TOPP), ("Fixed budget", FIXED)):
        groups = [(float(sc), out["arms"][a]["n"], out["arms"][a]["success"]) for a, sc in fam]
        z, p = cochran_armitage(groups)
        trends[name] = {"z": z, "p": p}
        A(f"| {name} | {fam[0][1]} → {fam[-1][1]} | {z:+.3f} | {p:.4f} |")
    out["family_trends"] = trends
    A("")
    A("## 7. 预算分布（每臂逐集平均 m_t 的 min / 中位 / max）")
    A("")
    A("| arm | min | median | max | 单集平均耗时 |")
    A("|---|---|---|---|---|")
    for a, row in rows:
        A(f"| {a.replace('_l100','')} | {row['min_m_t']:.0f} | {row['median_m_t']:.1f} | {row['max_m_t']:.0f} | "
          f"{row['mean_runtime_s']:.0f}s |")
    A("")

    text = "\n".join(L) + "\n"
    print(text)
    if args.write:
        Path("docs/OPENVLA_BUDGET_RULE_9TASK_FINAL.md").write_text(text)
        Path("docs/OPENVLA_BUDGET_RULE_9TASK_FINAL.json").write_text(json.dumps(out, indent=2, default=str))
        print("wrote docs/OPENVLA_BUDGET_RULE_9TASK_FINAL.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
