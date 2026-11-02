#!/usr/bin/env python3
"""FINAL report for the lambda x budget-rule interaction grid.

  lambda 0.5 / 1.0 x rules {matched, fixed016, fixed048, random_fixed48, top_p090}
  4 google tasks x 100 seeds (200-299), reference-aligned 10-seed process blocks

Reads /tmp/pfwatch/grid.tsv (task, seed, arm, success) built by a grep pass over the episodes.
Writes docs/OPENVLA_LAMBDA_BUDGET_GRID_FINAL.md and .json
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

GRID = Path("artifacts/openvla_lambda_budget_grid_200_299_v1")
TSV = Path("/tmp/pfwatch/grid.tsv")
RULES = ["matched", "fixed016", "fixed048", "random_fixed48", "top_p090"]
TASKS = ["google_robot_close_drawer", "google_robot_open_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]
SHORT = lambda t: t.replace("google_robot_", "")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    R = collections.defaultdict(dict)
    for r in csv.DictReader(open(TSV), delimiter="\t"):
        R[r["arm"]][(r["task"], int(r["seed"]))] = r["success"] == "true"

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "grid": str(GRID)}
    L = []
    A = L.append
    A("# λ × 预算规则 交互实验（Matched / Fixed / Random / Attention-TopP）· 最终结果")
    A("")
    A(f"*生成 {out['generated']}*  ·  4 个 google 任务 × 100 seeds（200–299）× 5 规则 × λ∈{{0.5, 1.0}}")
    A("")
    A("| λ | 规则 | 集数 | 成功率 | close | open | pick_coke | move_near |")
    A("|---|---|---|---|---|---|---|---|")
    rates = {}
    for lam, suf in (("0.5", "l050"), ("1.0", "l100")):
        for rule in RULES:
            arm = f"{rule}_{suf}"
            v = R.get(arm, {})
            if not v:
                continue
            n = len(v); s = sum(1 for x in v.values() if x)
            rates[arm] = (n, s)
            cells = []
            for t in TASKS:
                p = [(k, x) for k, x in v.items() if k[0] == t]
                cells.append(f"{sum(1 for k,x in p if x)}/{len(p)}" if p else "—")
            A(f"| {lam} | {rule} | {n} | **{100*s/n:.1f}%** | " + " | ".join(cells) + " |")
    A("")
    A("## 同一 λ 内：各规则 vs 该 λ 的 matched（配对 McNemar）")
    A("")
    A("| λ | 规则 | 对数 | 该臂 | matched | 净 | 救/坏 | p | Holm(8) |")
    A("|---|---|---|---|---|---|---|---|---|")
    ps, keys = [], []
    for lam, suf in (("0.5", "l050"), ("1.0", "l100")):
        ref = R[f"matched_{suf}"]
        for rule in RULES[1:]:
            arm = f"{rule}_{suf}"
            p = [(k, x) for k, x in R[arm].items() if k in ref]
            b = sum(1 for k, x in p if x and not ref[k]); c = sum(1 for k, x in p if ref[k] and not x)
            ps.append(mcnemar_exact(b, c)); keys.append((lam, rule, arm, len(p), b, c))
    adj = holm(ps)
    res = {}
    for (lam, rule, arm, n, b, c), pv, h in zip(keys, ps, adj):
        ref = R[f"matched_{'l050' if lam=='0.5' else 'l100'}"]
        p = [(k, x) for k, x in R[arm].items() if k in ref]
        sa = sum(1 for k, x in p if x); sr = sum(1 for k, x in p if ref[k])
        A(f"| {lam} | {rule} | {n} | {sa}/{n} | {sr}/{n} | **{b-c:+d}** | {b}/{c} | {pv:.4f} | {h:.3f} |")
        res[f"{rule}_{lam}"] = {"n": n, "success": sa, "rate": sa / n, "matched": sr,
                                "net": b - c, "rescue": b, "harm": c, "p": pv, "holm": h}
    out["within_lambda"] = res
    A("")
    A("## λ 效应：同一规则 λ=0.5 → λ=1.0（配对）")
    A("")
    A("| 规则 | 对数 | λ=0.5 | λ=1.0 | 净 | 救/坏 | p | Holm(5) |")
    A("|---|---|---|---|---|---|---|---|")
    ps2, keys2 = [], []
    for rule in RULES:
        a, b_ = R[f"{rule}_l050"], R[f"{rule}_l100"]
        p = [(k, x) for k, x in a.items() if k in b_]
        bb = sum(1 for k, x in p if b_[k] and not x); cc = sum(1 for k, x in p if x and not b_[k])
        ps2.append(mcnemar_exact(bb, cc)); keys2.append((rule, len(p), bb, cc,
                                                         sum(1 for k, x in p if x),
                                                         sum(1 for k, x in p if b_[k])))
    adj2 = holm(ps2)
    lam_eff = {}
    for (rule, n, bb, cc, sa, sb), pv, h in zip(keys2, ps2, adj2):
        A(f"| {rule} | {n} | {sa}/{n} = {100*sa/n:.1f}% | {sb}/{n} = {100*sb/n:.1f}% | "
          f"**{sb-sa:+d}** | {bb}/{cc} | {pv:.4f} | {h:.3f} |")
        lam_eff[rule] = {"n": n, "rate_l050": sa / n, "rate_l100": sb / n, "net": sb - sa,
                         "p": pv, "holm": h}
    out["lambda_effect"] = lam_eff

    A("")
    A("## 结论")
    A("")
    A("> 本节的旧版本曾用「打平 / 差异消失 / 对 λ 不敏感」等表述，那是把「一侧显著、另一侧不显著」")
    A("> 当成交互与等效证据，已作废。直接估计的 λ×规则交互量见")
    A("> `docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md`（同 task/seed 配对 + bootstrap 区间 + Holm）。")
    A("")
    A("* λ=0.5：Matched 明显优于**数量过小**（fixed016）；与 fixed048、top_p090 的观测差异在 Holm")
    A("  校正后不显著 —— **不显著不等于等效**，区间仍可能容纳几个百分点。")
    A("* λ=1.0：Matched 相对 fixed016/fixed048 的观测差异回到 0 附近；top_p090 的观测值更高，")
    A("  但校正后不显著。若要主张「数量维度随 λ 变得无关」，必须报告交互量及其区间，而不是 p 值对比。")
    A("* 内容与数量的分离只能由 fixed048 vs random_fixed48（同数量、不同内容）给出；")
    A("  Matched vs random_fixed48 同时改变数量规则与内容，不能单独归因。")
    text = "\n".join(L) + "\n"
    print(text)
    if args.write:
        Path("docs/OPENVLA_LAMBDA_BUDGET_GRID_FINAL.md").write_text(text)
        Path("docs/OPENVLA_LAMBDA_BUDGET_GRID_FINAL.json").write_text(json.dumps(out, indent=2, default=str))
        print("wrote docs/OPENVLA_LAMBDA_BUDGET_GRID_FINAL.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
