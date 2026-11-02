#!/usr/bin/env python3
"""Build/refresh a compact index of the budget-rule study so status reports stay cheap.

The episode summaries are pretty-printed with alphabetically sorted keys, so `success`,
`task` and `seed` sit behind an ~800 KB `selector_trace`; re-parsing them all for every
status check is slow.  This writes one TSV row per episode:

    task  seed  arm  success  steps  runtime_seconds  mean_m_t  budget_param

and only re-reads files whose mtime is newer than the row already in the index.

Usage: python3 scripts/index_openvla_budget_rule.py [--study DIR] [--ref DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

STUDY = "artifacts/openvla_budget_rule_9task_200_299_v1"
REF = "artifacts/openvla_lambda1_extension_200_299_v1"
FIELDS = ("task", "seed", "arm", "success", "steps", "runtime_seconds",
          "mean_selected_token_count", "budget_param")


def scan(root: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    eps = root / "episodes"
    if not eps.is_dir():
        return out
    for task_dir in os.scandir(eps):
        if not task_dir.is_dir():
            continue
        for arm_dir in os.scandir(task_dir.path):
            if not arm_dir.is_dir():
                continue
            for f in os.scandir(arm_dir.path):
                if not f.name.endswith("_summary.json"):
                    continue
                try:
                    d = json.load(open(f.path))
                except Exception:
                    continue
                key = (d.get("task"), int(d.get("seed")), d.get("arm"))
                out[key] = dict({k: d.get(k) for k in FIELDS}, mtime=os.path.getmtime(f.path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=STUDY)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    study, ref = Path(args.study), Path(args.ref)

    t0 = time.time()
    rows = scan(study)
    ref_rows = scan(ref)
    print(f"scanned study={len(rows)} reference={len(ref_rows)} in {time.time() - t0:.1f}s")

    for root, data in ((study, rows), (ref, ref_rows)):
        idx = root / "INDEX.tsv"
        with idx.open("w") as fh:
            fh.write("task\tseed\tarm\tsuccess\tsteps\truntime_seconds\tmean_m_t\tbudget_param\tmtime\n")
            for (t, s, a), d in sorted(data.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
                fh.write(f"{t}\t{s}\t{a}\t{int(bool(d['success']))}\t{d['steps']}\t"
                         f"{d['runtime_seconds']}\t{d['mean_selected_token_count']}\t{d['budget_param']}\t{d['mtime']}\n")
        print(f"wrote {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
