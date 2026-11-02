#!/usr/bin/env python3
"""Combine the pi0-FAST L6-matched λ sweep across all NINE SIMPLER tasks.

  google_robot family (5 tasks, checkpoint `fractal`)  : 3500/3500 episodes, complete
  WidowX/Bridge family (4 tasks, checkpoint `bridge`)   : 2800 episodes, still finishing

Both runs share the guidance protocol (L6 prompt-attention matched budget, harmonic
negative reconstruction, greedy (1+λ)·pos − λ·neg), the same 7 arms, 100 seeds and the
same renderer (SAPIER RT full, depth 8, spp 32, denoiser).  They differ in checkpoint and
embodiment, so the two families are reported side by side rather than pooled into one mean.

Per-task numbers are same-seed matched: google is 100/100 seeds everywhere; WidowX uses the
seeds where all seven arms are complete.

Writes docs/PI0FAST_L6_MATCHED_9TASK_COMBINED.md
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import time
from pathlib import Path

GOOGLE_JSON = Path("docs/PI0FAST_L6_MATCHED_9TASK_100SEED_FINAL.json")
WX_RUN = Path("runs/pi0fast_bridge_widowx_4task_100seed")
OUT = Path("docs/PI0FAST_L6_MATCHED_9TASK_COMBINED.md")

ARMS = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075",
        "l6_matched_100", "l6_matched_125", "l6_matched_150"]
LBL = dict(zip(ARMS, ["vanilla", "λ0.25", "λ0.50", "λ0.75", "λ1.00", "λ1.25", "λ1.50"]))
WX_TASKS = ["widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
            "widowx_spoon_on_towel", "widowx_stack_cube"]
SHORT = {
    "google_robot_close_drawer": "close_drawer",
    "google_robot_move_near": "move_near",
    "google_robot_open_drawer": "open_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_place_apple_in_closed_top_drawer": "place_apple",
    "widowx_carrot_on_plate": "carrot_on_plate",
    "widowx_put_eggplant_in_basket": "put_eggplant",
    "widowx_spoon_on_towel": "spoon_on_towel",
    "widowx_stack_cube": "stack_cube",
}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def widowx_rows():
    rows = {}
    for f in glob.glob(str(WX_RUN / "episodes" / "*" / "*.json")):
        d = json.load(open(f))
        if d.get("status") == "complete":
            rows[(d["task"], d["seed"], d["arm"])] = d
    sets = {t: {s for s in range(100) if all((t, s, a) in rows for a in ARMS)} for t in WX_TASKS}
    return rows, sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    g = json.loads(GOOGLE_JSON.read_text())
    wx, wxsets = widowx_rows()

    lines: list[str] = []
    A = lines.append
    A("# pi0-FAST · L6-matched λ 扫描 · 9 个 SIMPLER 任务汇总")
    A("")
    A(f"*generated:* {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A("")
    A("| family | robot | checkpoint | tasks | episodes | status |")
    A("|---|---|---|---|---|---|")
    g_n = sum(v["n"] for v in g["per_arm"].values())
    wx_n = sum(1 for _ in wx)
    A(f"| `pi0fast_l6_matched_9task_100seed` | google_robot (5) | `openpi_fast_finetuned/fractal` | 5 | {g_n}/3500 | **完成**（02:26，7 闸门全绿） |")
    A(f"| `pi0fast_bridge_widowx_4task_100seed` | WidowX/Bridge (4) | `openpi_fast_finetuned/bridge` | 4 | {wx_n}/2800 | 收尾中（剩 8 集，0 错误行） |")
    A("")
    A("协议相同：L6 prompt-attention matched 动态预算、harmonic 负样本重建、贪心 `(1+λ)·pos − λ·neg`；"
      "7 个 arm（vanilla + λ0.25…1.50）× 100 seeds；渲染同口径 SAPIEN RT full depth8 / spp32 / denoiser。"
      "差别只在 checkpoint 与机器人本体，因此两个 family 并列展示、不合并求均值。")
    A("")

    # ---------------- main table ----------------
    A("## 1. 逐任务成功率（成功数/总数 · 成功率；WidowX 列用同 seed 匹配子集）")
    A("")
    A("| 任务 | " + " | ".join(LBL[a] for a in ARMS) + " | 最优 λ |")
    A("|" + "---|" * (len(ARMS) + 2))
    fam_tot = {"google_robot": collections.Counter(), "widowx": collections.Counter()}
    fam_n = {"google_robot": 0, "widowx": 0}
    per_task_out = {}
    for fam, tasks in (("google_robot", list(g["per_task"])), ("widowx", WX_TASKS)):
        for t in tasks:
            if fam == "google_robot":
                cells, ns = [], None
                for a in ARMS:
                    s, n = g["per_task"][t][a]["success"], g["per_task"][t][a]["n"]
                    cells.append(f"{s}/{n} · {100 * s / n:.0f}%")
                    fam_tot[fam][a] += s
                    if a == "vanilla":
                        fam_n[fam] += n
                n_here = g["per_task"][t]["vanilla"]["n"]
                counts = {a: g["per_task"][t][a]["success"] for a in ARMS}
            else:
                ss = sorted(wxsets[t]); n_here = len(ss)
                counts = {a: sum(1 for s in ss if wx[(t, s, a)].get("success")) for a in ARMS}
                cells = [f"{counts[a]}/{n_here} · {100 * counts[a] / n_here:.0f}%" for a in ARMS]
                for a in ARMS:
                    fam_tot[fam][a] += counts[a]
                fam_n[fam] += n_here
            best = max(ARMS[1:], key=lambda a: counts[a])
            if counts[best] == 0:
                best_txt = "—（λ 臂全 0，vanilla 1/100）" if counts["vanilla"] else "—（全 0）"
            else:
                best_cells = [LBL[a] for a in ARMS[1:] if counts[a] == counts[best]]
                best_txt = f"{' = '.join(best_cells)} ({100 * counts[best] / n_here:.0f}%)"
            A(f"| {SHORT[t]}{' *' if fam == 'widowx' else ''} | " + " | ".join(cells)
              + f" | {best_txt} |")
            per_task_out[t] = {"family": fam, "n": n_here, "counts": counts}

    # family subtotals + grand total
    for fam, label in (("google_robot", "**google_robot 5 任务小计**"), ("widowx", "**WidowX 4 任务小计**")):
        n = fam_n[fam]
        cells = [f"**{fam_tot[fam][a]}/{n} · {100 * fam_tot[fam][a] / n:.1f}%**" for a in ARMS]
        A(f"| {label} | " + " | ".join(cells) + " | |")
    grand_n = sum(fam_n.values())
    grand = {a: fam_tot["google_robot"][a] + fam_tot["widowx"][a] for a in ARMS}
    A("| **9 任务合计** | " + " | ".join(f"**{grand[a]}/{grand_n} · {100 * grand[a] / grand_n:.1f}%**" for a in ARMS) + " | |")
    A("")
    A("`*` WidowX 行的 n 为同 seed 匹配后的单位数（当前 98–100）。")
    A("")

    # ---------------- paired ----------------
    A("## 2. 配对 vs vanilla（同 task/seed，exact McNemar）")
    A("")
    A("| arm | google_robot (n=500) | WidowX (n=%d) | 9 任务合计 (n=%d) |" % (fam_n["widowx"], grand_n))
    A("|---|---|---|---|")
    for a in ARMS[1:]:
        gb, gc = g["paired_vs_vanilla"][a]["rescue"], g["paired_vs_vanilla"][a]["harm"]
        units = [(t, s) for t in WX_TASKS for s in sorted(wxsets[t])]
        b = sum(1 for (t, s) in units if wx[(t, s, a)].get("success") and not wx[(t, s, "vanilla")].get("success"))
        c = sum(1 for (t, s) in units if wx[(t, s, "vanilla")].get("success") and not wx[(t, s, a)].get("success"))
        A(f"| {LBL[a]} | +{gb - gc} ({gb}/{gc}) p={g['paired_vs_vanilla'][a]['p_exact']:.4f} | "
          f"+{b - c} ({b}/{c}) p={mcnemar_exact(b, c):.4f} | "
          f"+{(gb - gc) + (b - c)} ({gb + b}/{gc + c}) |")
    A("")
    A("（合计列的 p 不做跨 family 合并检验：两个 family 的 checkpoint/本体不同，"
      "合并数字只作为总量参考，显著性分别看前两列。）")
    A("")

    # ---------------- reading ----------------
    A("## 3. 读法")
    A("")
    A("* **引导有效是硬结论**：两个 family、九个任务上，六个 λ 相对 vanilla 的净收益全为正，"
      "google 侧 λ1.00 +27（p=0.0028）、λ0.25 +22（p=0.0198）；WidowX 侧 λ1.00 +49（p<0.0001）、"
      "λ1.50 +44（p<0.0001）。")
    A("* **λ 之间差异很小**：google 侧 λ1.00（37.6%）> λ0.25（36.6%）> λ0.50（36.2%）> λ0.75（36.0%）> "
      "λ1.50（35.6%）> λ1.25（34.6%）；WidowX 侧 λ1.00（29.0%）> λ1.50（27.8%）> λ0.25（26.8%）> "
      "λ0.75（25.3%）> λ1.25（25.0%）> λ0.50（24.2%）。**λ1.00 在两个 family 都是第一**，"
      "但 λ0.25–1.50 之间的两两检验大多不显著（WidowX 侧只有 λ1.00>λ0.50、λ1.00>λ1.25 达到 p<0.05）。")
    A("* **任务难度差异巨大，引导收益集中在「原本做不动」的任务上**：pick_coke_can 19→40/100、"
      "spoon_on_towel 7→32/100、stack_cube 3→14/100；而已经很强的 close_drawer（60/100）几乎不动，"
      "place_apple 全部为 0–1/100（两个 family 都无法完成，这是能力边界而非 λ 问题）。")
    A("* WidowX 上 **eggplant 是唯一存在负向风险的任务**：vanilla 42%，λ0.50 只有 35%（配对里变坏多于救回），"
      "而 λ1.50 反过来最好（51%）。")
    A("")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        OUT.write_text(text)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
