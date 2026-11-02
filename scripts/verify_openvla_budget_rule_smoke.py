#!/usr/bin/env python3
"""Verify the budget-rule smoke test.

1. Reproduction check: the smoke's `lambda_100` (matched) episodes must match the existing
   lambda=1.0 Ours reference exactly (success, steps, mean_selected_token_count, and the
   canonical/state/RGB hashes).  Any mismatch means the port changed the mechanism.
2. New-rule checks: per-arm episode count, success rate, mean/min/max m_t, the recorded
   TopP bounds (must be 1/256 i.e. unclipped, with no bound triggered) and, for fixed arms,
   an exact m_t == param.

Usage: python3 scripts/verify_openvla_budget_rule_smoke.py [--artifact ...] [--reference ...]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

DEFAULT_ARTIFACT = "artifacts/openvla_budget_rule_smoke_v1"
DEFAULT_REFERENCE = "artifacts/openvla_lambda1_extension_200_299_v1"


def load_arm(root: Path, task: str, arm: str) -> dict[int, dict]:
    out = {}
    for f in glob.glob(str(root / "episodes" / task / arm / "episode_*_summary.json")):
        d = json.load(open(f))
        out[int(d["seed"])] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--task", default="widowx_carrot_on_plate")
    ap.add_argument("--seeds", default="200-219")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    art, ref = Path(args.artifact), Path(args.reference)
    ok = True

    # ---- 1. reproduction ----------------------------------------------------------------
    sm = load_arm(art, args.task, "lambda_100")
    rf = load_arm(ref, args.task, "lambda_100")
    print("=" * 78)
    print(f"1. REPRODUCTION  {args.task} seeds {args.seeds}: smoke lambda_100 vs Ours reference")
    print(f"   smoke episodes: {len(sm)}   reference episodes: {len(rf)}")
    fields = ("success", "steps", "mean_selected_token_count", "canonical_snapshot_sha256",
              "initial_state_sha256", "initial_rgb_sha256", "lambda", "protocol_id")
    diffs = []
    for s in seeds:
        a, b = sm.get(s), rf.get(s)
        if not a or not b:
            diffs.append((s, "missing", a is not None, b is not None))
            continue
        for f in fields:
            if a.get(f) != b.get(f):
                diffs.append((s, f, a.get(f), b.get(f)))
    if not diffs:
        same_succ = sum(1 for s in seeds if sm.get(s, {}).get("success") == rf.get(s, {}).get("success"))
        print(f"   IDENTICAL on all {len(fields)} fields for every seed "
              f"(success agreement {same_succ}/{len(seeds)})  ->  PORT IS FAITHFUL")
    else:
        ok = False
        print(f"   MISMATCHES: {len(diffs)} (first 8)")
        for d in diffs[:8]:
            print("     seed", d)

    # ---- 2. new rules -------------------------------------------------------------------
    print("-" * 78)
    print("2. NEW BUDGET RULES")
    print(f"   {'arm':18s} {'n':>4s} {'succ':>7s} {'mean_m_t':>9s} {'min':>5s} {'max':>5s} "
          f"{'bounds':>10s} {'clip_hit':>9s}")
    for arm_dir in sorted((art / "episodes" / args.task).glob("*")):
        arm = arm_dir.name
        rows = load_arm(art, args.task, arm)
        rows = {s: d for s, d in rows.items() if s in seeds}
        if not rows: continue
        mts = [d.get("mean_selected_token_count") or 0 for d in rows.values()]
        mins = [d.get("min_selected_token_count") or 0 for d in rows.values()]
        maxs = [d.get("max_selected_token_count") or 0 for d in rows.values()]
        succ = sum(1 for d in rows.values() if d.get("success"))
        bounds, clip_hit = "-", "-"
        if arm.startswith("top_p"):
            b = {(d["selector_trace"][0].get("selection_min_count"),
                  d["selector_trace"][0].get("selection_max_count")) for d in rows.values()}
            bounds = "/".join(f"{x[0]}-{x[1]}" for x in b)
            hits = sum(1 for d in rows.values() for x in d["selector_trace"]
                       if x.get("lower_bound_triggered") or x.get("upper_bound_triggered"))
            clip_hit = str(hits)
            if hits:
                ok = False
            if b != {(1, 256)}:
                ok = False
                print("   !! TopP bounds are not 1/256 -- the run is NOT unclipped")
        if arm.startswith("fixed"):
            params = {d.get("budget_param") for d in rows.values()}
            exact = all(all(int(x.get("m_t", -1)) == int(next(iter(params))) for x in d["selector_trace"])
                        for d in rows.values())
            if len(params) != 1 or not exact:
                ok = False
                print(f"   !! fixed arm {arm} does not hold m_t constant: params={params} exact={exact}")
            clip_hit = "m_t==param"
        print(f"   {arm:18s} {len(rows):4d} {succ:3d}/{len(rows):<3d} {sum(mts)/len(mts):9.1f} "
              f"{min(mins):5.0f} {max(maxs):5.0f} {bounds:>10s} {clip_hit:>9s}")
    print("-" * 78)
    print("VERDICT:", "PASS" if ok else "FAIL -- do not start the full run")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
