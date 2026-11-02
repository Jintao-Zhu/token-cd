#!/usr/bin/env python3
"""Export the 9-task pi0-FAST L6-matched λ table in the exact layout requested:

任务 | vanilla | λ0.25 | λ0.50 | λ0.75 | λ1.00 | λ1.25 | λ1.50 | 最优 λ

Every row carries its own best λ — including the two family subtotals and the 9-task total,
which are picked by pooled success count (and ties are listed).

Outputs: docs/PI0FAST_L6_MATCHED_9TASK_TABLE.{md,tsv,csv}
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

GOOGLE_JSON = Path("docs/PI0FAST_L6_MATCHED_9TASK_100SEED_FINAL.json")
WX_RUN = Path("runs/pi0fast_bridge_widowx_4task_100seed")
OUT = Path("docs/PI0FAST_L6_MATCHED_9TASK_TABLE")

ARMS = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075",
        "l6_matched_100", "l6_matched_125", "l6_matched_150"]
HDR = ["任务", "vanilla", "λ0.25", "λ0.50", "λ0.75", "λ1.00", "λ1.25", "λ1.50", "最优 λ"]
GOOGLE_TASKS = {
    "google_robot_close_drawer": "close_drawer",
    "google_robot_move_near": "move_near",
    "google_robot_open_drawer": "open_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_place_apple_in_closed_top_drawer": "place_apple",
}
WX_TASKS = {
    "widowx_carrot_on_plate": "carrot_on_plate *",
    "widowx_put_eggplant_in_basket": "put_eggplant *",
    "widowx_spoon_on_towel": "spoon_on_towel *",
    "widowx_stack_cube": "stack_cube *",
}
LBL = {"vanilla": "vanilla", "l6_matched_025": "λ0.25", "l6_matched_050": "λ0.50",
       "l6_matched_075": "λ0.75", "l6_matched_100": "λ1.00", "l6_matched_125": "λ1.25",
       "l6_matched_150": "λ1.50"}


def best_label(counts: dict[str, int], n: int) -> str:
    guided = {a: c for a, c in counts.items() if a != "vanilla"}
    top = max(guided.values())
    if top == 0:
        return "—（λ 臂全 0）" if counts.get("vanilla") else "—"
    ties = [LBL[a] for a in ARMS[1:] if guided[a] == top]
    return f"{' = '.join(ties)}（{100 * top / n:.0f}%）"


def main() -> int:
    g = json.loads(GOOGLE_JSON.read_text())

    wx: dict[tuple, dict] = {}
    for f in glob.glob(str(WX_RUN / "episodes" / "*" / "*.json")):
        d = json.load(open(f))
        if d.get("status") == "complete":
            wx[(d["task"], d["seed"], d["arm"])] = d
    wxsets = {t: {s for s in range(100) if all((t, s, a) in wx for a in ARMS)} for t in WX_TASKS}

    rows: list[list[str]] = []
    g_tot = {a: 0 for a in ARMS}
    w_tot = {a: 0 for a in ARMS}
    g_n = 0
    w_n = 0
    for t, name in GOOGLE_TASKS.items():
        n = g["per_task"][t]["vanilla"]["n"]
        counts = {a: g["per_task"][t][a]["success"] for a in ARMS}
        for a in ARMS:
            g_tot[a] += counts[a]
        g_n += n
        rows.append([name] + [f"{counts[a]}/{n} · {100 * counts[a] / n:.0f}%" for a in ARMS]
                    + [best_label(counts, n)])
    for t, name in WX_TASKS.items():
        ss = sorted(wxsets[t])
        n = len(ss)
        counts = {a: sum(1 for s in ss if wx[(t, s, a)].get("success")) for a in ARMS}
        for a in ARMS:
            w_tot[a] += counts[a]
        w_n += n
        rows.append([name] + [f"{counts[a]}/{n} · {100 * counts[a] / n:.0f}%" for a in ARMS]
                    + [best_label(counts, n)])
    rows.append(["google_robot 小计（5 任务）"]
                + [f"{g_tot[a]}/{g_n} · {100 * g_tot[a] / g_n:.1f}%" for a in ARMS]
                + [best_label(g_tot, g_n)])
    rows.append(["WidowX 小计（4 任务）"]
                + [f"{w_tot[a]}/{w_n} · {100 * w_tot[a] / w_n:.1f}%" for a in ARMS]
                + [best_label(w_tot, w_n)])
    tot = {a: g_tot[a] + w_tot[a] for a in ARMS}
    n_all = g_n + w_n
    rows.append(["9 任务合计"] + [f"{tot[a]}/{n_all} · {100 * tot[a] / n_all:.1f}%" for a in ARMS]
                + [best_label(tot, n_all)])

    md = ["| " + " | ".join(HDR) + " |", "|" + "---|" * len(HDR)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md.append("")
    md.append(f"`*` WidowX 行用同 seed 匹配子集（当前 {len(wxsets[list(WX_TASKS)[0]])}/"
              f"{len(wxsets[list(WX_TASKS)[1]])}/{len(wxsets[list(WX_TASKS)[2]])}/"
              f"{len(wxsets[list(WX_TASKS)[3]])}，收尾后为 100）。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n")
    with OUT.with_suffix(".tsv").open("w") as fh:
        fh.write("\t".join(HDR) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with OUT.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(rows)

    print("\n".join(md))
    print(f"\nwrote {OUT}.md / .tsv / .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
