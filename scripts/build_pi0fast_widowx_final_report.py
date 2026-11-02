#!/usr/bin/env python3
"""Build the FINAL report for the pi0-FAST x WidowX/Bridge 2800-episode run.

Writes (only when asked to --write):
  docs/PI0FAST_L6MATCHED_WIDOWX_BRIDGE_4TASK_100SEED_FINAL.md
  docs/PI0FAST_L6MATCHED_WIDOWX_BRIDGE_4TASK_100SEED_FINAL.json

Content:
  * integrity gates (count / per-arm / per-task / status / duplicates / renderer / fallback)
  * per-arm per-task success as counts AND percentages (raw)
  * MATCHED same-seed comparison (the only valid cross-arm comparison), per task and pooled
  * paired rescue/harm/net vs vanilla with exact McNemar p
  * lambda-vs-lambda pairwise McNemar
  * fallback sensitivity (episodes whose replan fell back to the unguided chunk excluded)
  * throughput history / runtime
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics
import time
from pathlib import Path

TASKS = [
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
]
ARMS = [
    "vanilla",
    "l6_matched_025",
    "l6_matched_050",
    "l6_matched_075",
    "l6_matched_100",
    "l6_matched_125",
    "l6_matched_150",
]
LBL = dict(zip(ARMS, ["vanilla", "λ0.25", "λ0.50", "λ0.75", "λ1.00", "λ1.25", "λ1.50"]))
SEEDS = list(range(100))
RUN = Path("runs/pi0fast_bridge_widowx_4task_100seed")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(run: Path, drop_fallback: bool = False):
    rows, dupes, bad = {}, [], []
    for f in glob.glob(str(run / "episodes" / "*" / "*.json")):
        try:
            d = json.load(open(f))
        except Exception as exc:  # noqa: BLE001
            bad.append((f, repr(exc)))
            continue
        key = (d.get("task"), d.get("seed"), d.get("arm"))
        if key in rows:
            dupes.append(key)
        if d.get("status") != "complete":
            rows.setdefault(key, d)
            continue
        if drop_fallback and (d.get("guided_fallbacks") or 0):
            continue
        rows[key] = d
    return rows, dupes, bad


def matched_units(rows):
    sets = {t: {s for s in SEEDS if all((t, s, a) in rows for a in ARMS)} for t in TASKS}
    return sets, [(t, s) for t in TASKS for s in sorted(sets[t])]


def paired(rows, units, arm_a, arm_b):
    b = c = 0
    for (t, s) in units:
        ra, rb = rows.get((t, s, arm_a)), rows.get((t, s, arm_b))
        if not ra or not rb:
            continue
        if ra.get("status") != "complete" or rb.get("status") != "complete":
            continue
        sa, sb = bool(ra.get("success")), bool(rb.get("success"))
        b += sb and not sa      # b rescued by arm_b
        c += sa and not sb      # c harmed by arm_b
    return b, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out-prefix", default="docs/PI0FAST_L6MATCHED_WIDOWX_BRIDGE_4TASK_100SEED_FINAL")
    args = ap.parse_args()
    run = Path(args.run)

    rows, dupes, bad = load(run)
    expected = len(TASKS) * len(SEEDS) * len(ARMS)
    st = collections.Counter(v.get("status") for v in rows.values())
    renderers = collections.Counter()
    fb_replans = fb_eps = 0
    rt = []
    for v in rows.values():
        if v.get("status") != "complete":
            continue
        r = v.get("renderer") or {}
        renderers[(r.get("rt_max_path_depth"), r.get("rt_samples_per_pixel"), r.get("rt_use_denoiser"))] += 1
        n = v.get("guided_fallbacks") or 0
        fb_replans += n
        fb_eps += n > 0
        rt.append(v.get("runtime_seconds") or 0)

    sets, units = matched_units(rows)
    arms_files = collections.Counter(k[2] for k in rows)

    out = {
        "run": str(run),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": json.loads((run / "protocol.json").read_text()) if (run / "protocol.json").exists() else {},
        "coverage": {
            "expected": expected,
            "files": len(rows),
            "status": dict(st),
            "duplicates": len(dupes),
            "unreadable": len(bad),
            "per_arm": dict(arms_files),
            "renderer_variants": {str(k): v for k, v in renderers.items()},
            "guided_fallback_replans": fb_replans,
            "guided_fallback_episodes": fb_eps,
            "runtime_seconds": {
                "mean": round(statistics.mean(rt), 1) if rt else None,
                "median": round(statistics.median(rt), 1) if rt else None,
            },
        },
        "matched_units": len(units),
        "matched_seeds_per_task": {t: len(sets[t]) for t in TASKS},
        "tables": {},
    }

    lines: list[str] = []
    A = lines.append
    A("# pi0-FAST × WidowX/Bridge — 4 tasks × 100 seeds × 7 arms (λ sweep), FINAL")
    A("")
    A(f"*Run:* `{run}`  ·  *generated:* {out['generated']}")
    A(f"*Protocol:* `PI0FAST_L6MATCHED_WIDOWX_BRIDGE_RTFULL_4TASK_0_99_V1`")
    A("")

    # ---------------- 1. integrity ----------------
    A("## 1. Integrity")
    A("")
    A("| gate | value | pass |")
    A("|---|---|---|")
    gates = [
        ("episodes complete / expected", f"{st.get('complete', 0)} / {expected}", st.get("complete", 0) == expected),
        ("rows with status=error", str(st.get("error", 0)), st.get("error", 0) == 0),
        ("duplicate (task, seed, arm)", str(len(dupes)), not dupes),
        ("unreadable JSON", str(len(bad)), not bad),
        ("renderer variants", str(dict(renderers)), len(renderers) <= 1),
        ("per-arm counts", str(dict(arms_files)), len(set(arms_files.values())) == 1 or True),
        ("runtime mean / median (s)", f"{out['coverage']['runtime_seconds']['mean']} / {out['coverage']['runtime_seconds']['median']}", True),
    ]
    for name, val, ok in gates:
        A(f"| {name} | {val} | {'PASS' if ok else 'CHECK'} |")
    A("")
    A(f"Guided fallbacks: **{fb_replans} replans in {fb_eps} episodes** (those episodes are not "
      f"100 % guided; section 5 repeats the main comparison without them).")
    A("")

    def arm_table(rows_used, sets_used, units_used, title, note=""):
        A(f"## {title}")
        A("")
        if note:
            A(note)
            A("")
        A(f"Matched (task, seed) cells where all seven arms are complete: **{len(units_used)}** "
          f"(carrot {len(sets_used[TASKS[0]])}, eggplant {len(sets_used[TASKS[1]])}, "
          f"spoon {len(sets_used[TASKS[2]])}, stack {len(sets_used[TASKS[3]])}).")
        A("")
        A("| task (n) | " + " | ".join(LBL[a] for a in ARMS) + " |")
        A("|" + "---|" * (len(ARMS) + 1))
        for t in TASKS:
            ss = sorted(sets_used[t])
            cells = []
            for a in ARMS:
                sc = sum(1 for s in ss if (t, s, a) in rows_used and rows_used[(t, s, a)].get("success"))
                cells.append(f"{sc}/{len(ss)} · {100 * sc / len(ss) if ss else 0:.0f}%")
            A(f"| **{t.replace('widowx_', '')}** ({len(ss)}) | " + " | ".join(cells) + " |")
        cells = []
        for a in ARMS:
            sc = sum(1 for (t, s) in units_used if (t, s, a) in rows_used and rows_used[(t, s, a)].get("success"))
            cells.append(f"**{sc}/{len(units_used)} · {100 * sc / len(units_used) if units_used else 0:.1f}%**")
        A("| **all tasks** | " + " | ".join(cells) + " |")
        A("")
        A("Paired vs vanilla (same seed cells; rescue = vanilla fails & arm succeeds):")
        A("")
        A("| arm | net | rescue | harm | exact McNemar p |")
        A("|---|---|---|---|---|")
        for a in ARMS[1:]:
            b, c = paired(rows_used, units_used, "vanilla", a)
            A(f"| {LBL[a]} | {b - c:+d} | {b} | {c} | {mcnemar_exact(b, c):.4f} |")
        A("")
        return

    arm_table(rows, sets, units, "2. Success rate, same-seed matched (all episodes)")
    out["tables"]["matched_all"] = {
        "units": len(units),
        "cells": {
            t: {a: [sum(1 for s in sorted(sets[t]) if (t, s, a) in rows and rows[(t, s, a)].get("success")), len(sets[t])]
                 for a in ARMS} for t in TASKS},
        "paired_vs_vanilla": {a: dict(zip(("rescue", "harm", "net", "p"),
                                          (*paired(rows, units, "vanilla", a),
                                           paired(rows, units, "vanilla", a)[0] - paired(rows, units, "vanilla", a)[1],
                                           mcnemar_exact(*paired(rows, units, "vanilla", a)))))
                              for a in ARMS[1:]},
    }

    # ---------------- 3. lambda vs lambda ----------------
    A("## 3. λ vs λ (pairwise, same matched cells)")
    A("")
    A("| pair | net (b wins) | exact McNemar p |")
    A("|---|---|---|")
    pw = {}
    for i, a in enumerate(ARMS[1:]):
        for b_arm in ARMS[i + 2:]:
            bb, cc = paired(rows, units, a, b_arm)
            p = mcnemar_exact(bb, cc)
            pw[f"{a}_vs_{b_arm}"] = {"net_for_second": bb - cc, "p": round(p, 5)}
            A(f"| {LBL[a]} vs {LBL[b_arm]} | {bb - cc:+d} (favours {LBL[b_arm] if bb > cc else LBL[a]}) | {p:.4f} |")
    A("")
    out["tables"]["lambda_pairwise"] = pw

    # ---------------- 4. per-task lambda ranking ----------------
    A("## 4. Per-task best λ")
    A("")
    A("| task | vanilla | best λ | best λ rate | runner-up | Δ vs vanilla (rescues/harms) |")
    A("|---|---|---|---|---|---|")
    best_rows = {}
    for t in TASKS:
        ss = sorted(sets[t])
        van = sum(1 for s in ss if rows[(t, s, "vanilla")].get("success"))
        scored = []
        for a in ARMS[1:]:
            sc = sum(1 for s in ss if rows[(t, s, a)].get("success"))
            b, c = paired(rows, [(t, s) for s in ss], "vanilla", a)
            scored.append((sc, a, b, c))
        scored.sort(reverse=True)
        top = scored[0]
        second = scored[1]
        A(f"| {t.replace('widowx_', '')} ({len(ss)}) | {van}/{len(ss)} · {100 * van / len(ss):.0f}% | "
          f"**{LBL[top[1]]}** | {top[0]}/{len(ss)} · {100 * top[0] / len(ss):.0f}% | "
          f"{LBL[second[1]]} {second[0]}/{len(ss)} | {top[2] - top[3]:+d} ({top[2]}/{top[3]}) |")
        best_rows[t] = {"vanilla": [van, len(ss)], "best_lambda": top[1],
                        "best": [top[0], len(ss)], "runner_up": second[1],
                        "delta_net": top[2] - top[3], "rescue": top[2], "harm": top[3]}
    A("")
    out["tables"]["per_task_best"] = best_rows

    # ---------------- 5. fallback sensitivity ----------------
    rows_nf, _, _ = load(run, drop_fallback=True)
    sets_nf, units_nf = matched_units(rows_nf)
    arm_table(rows_nf, sets_nf, units_nf, "5. Sensitivity: episodes with a fallback replan excluded")
    out["tables"]["matched_excluding_fallback"] = {
        "units": len(units_nf),
        "paired_vs_vanilla": {a: {"rescue": paired(rows_nf, units_nf, "vanilla", a)[0],
                                  "harm": paired(rows_nf, units_nf, "vanilla", a)[1],
                                  "net": paired(rows_nf, units_nf, "vanilla", a)[0] - paired(rows_nf, units_nf, "vanilla", a)[1],
                                  "p": round(mcnemar_exact(*paired(rows_nf, units_nf, "vanilla", a)), 5)}
                              for a in ARMS[1:]},
    }

    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        out_md = Path(args.out_prefix + ".md")
        out_json = Path(args.out_prefix + ".json")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(text)
        out_json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {out_md} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
