#!/usr/bin/env python3
"""Summarise the pi0-FAST WidowX/Bridge 4-task x 100-seed x 7-arm run.

Reports, in this order:
  1. coverage / integrity (complete, error, missing) per arm
  2. per-arm success as counts AND percentages, overall and per task
  3. paired rescue / harm / net vs vanilla on the common (task, seed) subset,
     with exact McNemar two-sided p-values
  4. guided_fallbacks (replans that used the unguided fallback) per arm
  5. throughput + ETA from episode file mtimes

Usage:
  python3 scripts/report_pi0fast_widowx_2800.py [--run DIR] [--rate-window-min 20]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import time

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
SEEDS = list(range(100))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(run: str):
    rows = {}
    bad = []
    for f in glob.glob(os.path.join(run, "episodes", "*", "*.json")):
        try:
            d = json.load(open(f))
        except Exception as exc:  # noqa: BLE001
            bad.append((f, repr(exc)))
            continue
        key = (d.get("task"), int(d.get("seed")), d.get("arm"))
        rows[key] = d
    return rows, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/pi0fast_bridge_widowx_4task_100seed")
    ap.add_argument("--rate-window-min", type=float, default=20.0)
    ap.add_argument(
        "--exclude-fallback",
        action="store_true",
        help="drop episodes that used the unguided fallback at any replan "
             "(sensitivity pass: those episodes are not 100%% guided)")
    args = ap.parse_args()
    run = args.run

    rows, bad = load(run)
    total_expected = len(TASKS) * len(SEEDS) * len(ARMS)

    # ---------- 1. coverage ----------
    st = collections.Counter(v.get("status") for v in rows.values())
    arm_n = collections.Counter(k[2] for k in rows)
    print("=" * 78)
    print(f"RUN  {run}")
    print(f"files={len(rows)}  expected={total_expected}  "
          f"missing={total_expected - len(rows)}  status={dict(st)}")
    if bad:
        print(f"unreadable files: {len(bad)} (first: {bad[0]})")

    def cell(task, arm):
        got = [rows.get((task, s, arm)) for s in SEEDS]
        ok = [g for g in got if g and g.get("status") == "complete"]
        if args.exclude_fallback:
            ok = [g for g in ok if not (g.get("guided_fallbacks") or 0)]
        succ = sum(1 for g in ok if g.get("success"))
        return succ, len(ok)

    # ---------- 2. per-arm success ----------
    print("-" * 78)
    print("PER-ARM SUCCESS (complete episodes only)")
    header = f"{'arm':16s} {'overall':>13s}"
    for t in TASKS:
        header += f" {t.replace('widowx_', '')[:11]:>12s}"
    print(header)
    overall = {}
    for arm in ARMS:
        s_all = n_all = 0
        line = f"{arm:16s}"
        for t in TASKS:
            s, n = cell(t, arm)
            s_all += s
            n_all += n
        overall[arm] = (s_all, n_all)
        line = f"{arm:16s} {s_all:5d}/{n_all:<5d}{100.0 * s_all / n_all if n_all else 0:5.1f}%"
        for t in TASKS:
            s, n = cell(t, arm)
            pct = 100.0 * s / n if n else 0.0
            line += f" {s:4d}/{n:<4d}{pct:5.1f}%"
        print(line)

    # ---------- 2b. MATCHED (same-seed) comparison ----------
    # Only (task, seed) cells where EVERY arm is complete are comparable: the raw
    # per-arm table above mixes different seed subsets (vanilla finishes seeds
    # earlier than the guided arms), so its percentages must not be compared
    # across arms.  This section fixes the seed set first, then compares.
    def matched_sets():
        return {t: {s for s in SEEDS if all((t, s, a) in rows for a in ARMS)} for t in TASKS}

    mset = matched_sets()
    units = [(t, s) for t in TASKS for s in sorted(mset[t])]
    print("-" * 78)
    print(f"MATCHED same-seed comparison -- {len(units)} (task, seed) cells where all "
          f"{len(ARMS)} arms are complete")
    print("  matched seeds per task: "
          + ", ".join(f"{t.replace('widowx_', '')}={len(mset[t])}" for t in TASKS))
    if not units:
        print("  (no fully matched cell yet -- wait for coverage)")
    else:
        print(f"{'arm':16s} {'succ':>9s} {'pct':>7s} {'net':>5s} {'resc/harm':>10s} "
              f"{'p(McNemar)':>11s}")
        for arm in ARMS:
            sc = sum(1 for (t, s) in units if rows[(t, s, arm)].get("success"))
            if arm == "vanilla":
                print(f"{arm:16s} {sc:4d}/{len(units):<4d} {100.0 * sc / len(units):6.1f}%"
                      f"     -          -           -")
                continue
            resc = sum(1 for (t, s) in units
                       if rows[(t, s, arm)].get("success") and not rows[(t, s, "vanilla")].get("success"))
            harm = sum(1 for (t, s) in units
                       if rows[(t, s, "vanilla")].get("success") and not rows[(t, s, arm)].get("success"))
            print(f"{arm:16s} {sc:4d}/{len(units):<4d} {100.0 * sc / len(units):6.1f}% "
                  f"{resc - harm:+4d} {resc:6d}/{harm:<3d} {mcnemar_exact(resc, harm):11.4f}")

    # ---------- 3. paired vs vanilla ----------
    print("-" * 78)
    print("PAIRED vs vanilla on common (task, seed) complete pairs")
    print(f"{'arm':16s} {'pairs':>6s} {'both':>5s} {'resc':>5s} {'harm':>5s} "
          f"{'net':>5s} {'p(McNemar)':>11s}   per-task net (resc/harm)")
    for arm in ARMS:
        if arm == "vanilla":
            continue
        both = resc = harm = neither = 0
        per_task = {}
        for t in TASKS:
            tb = tr = th = 0
            for s in SEEDS:
                v = rows.get((t, s, "vanilla"))
                g = rows.get((t, s, arm))
                if not v or not g:
                    continue
                if v.get("status") != "complete" or g.get("status") != "complete":
                    continue
                if args.exclude_fallback and ((v.get("guided_fallbacks") or 0)
                                              or (g.get("guided_fallbacks") or 0)):
                    continue
                vs, gs = bool(v.get("success")), bool(g.get("success"))
                if vs and gs:
                    both += 1
                    tb += 1
                elif gs and not vs:
                    resc += 1
                    tr += 1
                elif vs and not gs:
                    harm += 1
                    th += 1
                else:
                    neither += 1
            per_task[t] = (tr, th)
        pairs = both + resc + harm + neither
        p = mcnemar_exact(resc, harm)
        pt = "  ".join(f"{t.replace('widowx_', '')[:11]}:{r}/{h}" for t, (r, h) in per_task.items())
        print(f"{arm:16s} {pairs:6d} {both:5d} {resc:5d} {harm:5d} "
              f"{resc - harm:+5d} {p:11.4f}   {pt}")

    # ---------- 4. guided fallbacks ----------
    fb = collections.Counter()
    fb_eps = collections.Counter()
    for (t, s, arm), d in rows.items():
        n = d.get("guided_fallbacks") or 0
        if n:
            fb[arm] += n
            fb_eps[arm] += 1
    tot_fb = sum(fb.values())
    print("-" * 78)
    print(f"guided_fallbacks (unguided replans accepted by the fallback): {tot_fb}")
    if tot_fb:
        for arm in ARMS:
            if fb[arm]:
                print(f"  {arm:16s} replans={fb[arm]:4d} episodes={fb_eps[arm]:3d}")

    # ---------- 5. throughput / ETA ----------
    mt = []
    for f in glob.glob(os.path.join(run, "episodes", "*", "*.json")):
        try:
            mt.append(os.path.getmtime(f))
        except OSError:
            pass
    now = time.time()
    win = args.rate_window_min * 60.0
    recent = [m for m in mt if now - m <= win]
    rate = len(recent) / (win / 3600.0) if recent else 0.0
    remain = total_expected - len(rows)
    print("-" * 78)
    print(f"throughput: {len(recent)} episodes in last {args.rate_window_min:.0f} min "
          f"= {rate:.0f} eps/h   remaining={remain}")
    if rate > 0:
        eta_s = remain / (rate / 3600.0)
        print(f"ETA at current rate: {eta_s / 3600.0:.1f} h  "
              f"(≈ {time.strftime('%m-%d %H:%M', time.localtime(now + eta_s))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
