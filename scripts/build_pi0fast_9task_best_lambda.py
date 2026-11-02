#!/usr/bin/env python3
"""Per-task BEST-λ summary across all nine pi0-FAST SIMPLER tasks.

For each task: vanilla rate, the λ arm with the most successes (ties listed), its rate and its
same-seed paired net (rescue/harm + exact McNemar p) against vanilla, plus the runner-up.

Families: google_robot (5 tasks, checkpoint fractal, from runs/pi0fast_l6_matched_9task_100seed)
and WidowX/Bridge (4 tasks, checkpoint bridge, from runs/pi0fast_bridge_widowx_4task_100seed).
Only (task, seed) cells where all seven arms are complete are used.

Usage: python3 scripts/build_pi0fast_9task_best_lambda.py [--markdown]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

GOOGLE_RUN = Path("runs/pi0fast_l6_matched_9task_100seed")
WX_RUN = Path("runs/pi0fast_bridge_widowx_4task_100seed")
ARMS = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075",
        "l6_matched_100", "l6_matched_125", "l6_matched_150"]
LBL = dict(zip(ARMS, ["vanilla", "λ0.25", "λ0.50", "λ0.75", "λ1.00", "λ1.25", "λ1.50"]))
GOOGLE_TASKS = ["google_robot_close_drawer", "google_robot_move_near", "google_robot_open_drawer",
                "google_robot_pick_coke_can", "google_robot_place_apple_in_closed_top_drawer"]
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


def load(run: Path, tasks: list[str]):
    rows = {}
    for t in tasks:
        for f in glob.glob(str(run / "episodes" / t / "*.json")):
            d = json.load(open(f))
            if d.get("status") == "complete":
                rows[(t, int(d["seed"]), d["arm"])] = d
    sets = {t: {s for s in range(100) if all((t, s, a) in rows for a in ARMS)} for t in tasks}
    return rows, sets


def task_best(rows, ss, t):
    van = sum(1 for s in ss if rows[(t, s, "vanilla")].get("success"))
    scored = []
    for a in ARMS[1:]:
        sc = sum(1 for s in ss if rows[(t, s, a)].get("success"))
        b = sum(1 for s in ss if rows[(t, s, a)].get("success") and not rows[(t, s, "vanilla")].get("success"))
        c = sum(1 for s in ss if rows[(t, s, "vanilla")].get("success") and not rows[(t, s, a)].get("success"))
        scored.append((sc, a, b, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return van, len(ss), scored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    grows, gsets = load(GOOGLE_RUN, GOOGLE_TASKS)
    wrows, wsets = load(WX_RUN, WX_TASKS)

    print("| family | 任务 (n) | vanilla | **最优 λ** | 最优 λ 成绩 | 提升 | 配对净 (救/坏) | p | 次优 λ |")
    print("|---|---|---|---|---|---|---|---|---|")
    fam_rows = []
    for fam, rows, sets, tasks in (("google_robot", grows, gsets, GOOGLE_TASKS),
                                   ("widowx", wrows, wsets, WX_TASKS)):
        for t in tasks:
            ss = sorted(sets[t])
            van, n, scored = task_best(rows, ss, t)
            top = scored[0]
            ties = [LBL[a] for sc, a, b, c in scored if sc == top[0]]
            second = scored[1]
            if top[0] == 0:
                best_txt, second_txt = "—（λ 全 0）", "—"
            else:
                best_txt = " = ".join(ties)
                second_txt = f"{LBL[second[1]]} ({second[0]}/{n} · {100 * second[0] / n:.0f}%)"
            p = mcnemar_exact(top[2], top[3]) if top[0] else float("nan")
            p_txt = f"{p:.4f}" if top[0] else "—"
            print(f"| {fam} | {SHORT[t]} ({n}) | {van}/{n} · {100 * van / n:.0f}% | **{best_txt}** | "
                  f"{top[0]}/{n} · {100 * top[0] / n:.0f}% | {100 * (top[0] - van) / n:+.0f} pt | "
                  f"{top[2] - top[3]:+d} ({top[2]}/{top[3]}) | {p_txt} | {second_txt} |")
            fam_rows.append((fam, t, van, n, top, ties, second))

    # family + overall pooled
    print()
    for fam, rows, sets, tasks, label in (
            ("google_robot", grows, gsets, GOOGLE_TASKS, "**google_robot 汇总（5 任务）**"),
            ("widowx", wrows, wsets, WX_TASKS, "**WidowX 汇总（4 任务）**")):
        units = [(t, s) for t in tasks for s in sorted(sets[t])]
        van = sum(1 for (t, s) in units if rows[(t, s, "vanilla")].get("success"))
        scored = []
        for a in ARMS[1:]:
            sc = sum(1 for (t, s) in units if rows[(t, s, a)].get("success"))
            b = sum(1 for (t, s) in units if rows[(t, s, a)].get("success") and not rows[(t, s, "vanilla")].get("success"))
            c = sum(1 for (t, s) in units if rows[(t, s, "vanilla")].get("success") and not rows[(t, s, a)].get("success"))
            scored.append((sc, a, b, c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        n = len(units)
        print(f"{label}: n={n}，vanilla {van}/{n} · {100 * van / n:.1f}%")
        for sc, a, b, c in scored:
            print(f"  {LBL[a]:7s} {sc:3d}/{n} · {100 * sc / n:5.1f}%  净 {b - c:+3d} ({b}/{c})  p={mcnemar_exact(b, c):.4f}")

    # overall (both families pooled, per λ)
    units = [(t, s, rows) for (fam, rows, sets, tasks) in
             (("g", grows, gsets, GOOGLE_TASKS), ("w", wrows, wsets, WX_TASKS))
             for t in tasks for s in sorted(sets[t])]
    print()
    print(f"**9 任务合并（n={len(units)}，仅作总量参考，跨 family 不合并检验）**")
    for a in ARMS:
        sc = sum(1 for (t, s, rows) in units if rows[(t, s, a)].get("success"))
        print(f"  {LBL[a]:7s} {sc:3d}/{len(units)} · {100 * sc / len(units):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
