#!/usr/bin/env python3
"""Throughput per GPU for the pi0-FAST WidowX run, attributed via manifests18/.

Each (task, seed, arm) triple belongs to exactly one shard, and shard i lives on GPU
i // (NSHARD/3) + 1 (w00-w05 -> GPU1, w06-w11 -> GPU2, w12-w17 -> GPU3).  Episode JSONs
do not record the worker, so shard membership is the way to attribute a completed
episode to a GPU.  Only episodes produced *after* the 18-way resplit belong to those
shards, which is exactly the window we care about for the extra-server experiment.

Usage:
  python3 scripts/pi0fast_widowx_throughput_by_gpu.py --minutes 30
  python3 scripts/pi0fast_widowx_throughput_by_gpu.py --since 22:05 --until 22:35
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path

RUN = Path("runs/pi0fast_bridge_widowx_4task_100seed")
NSHARD = 18


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--nshard", type=int, default=NSHARD)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--since", default=None, help="HH:MM (today) start of the window")
    ap.add_argument("--until", default=None, help="HH:MM (today) end of the window")
    args = ap.parse_args()
    run = Path(args.run)

    owner: dict[tuple, int] = {}
    for i in range(args.nshard):
        p = run / f"manifests{args.nshard}" / f"worker_{i}.json"
        if not p.exists():
            continue
        for job in json.loads(p.read_text())["jobs"]:
            for arm in job["arms"]:
                owner[(job["task"], int(job["seed"]), arm)] = i

    now = time.time()
    if args.since and args.until:
        today = datetime.now().strftime("%Y-%m-%d")
        t0 = datetime.strptime(f"{today} {args.since}", "%Y-%m-%d %H:%M").timestamp()
        t1 = datetime.strptime(f"{today} {args.until}", "%Y-%m-%d %H:%M").timestamp()
    else:
        t1, t0 = now, now - args.minutes * 60

    per_gpu = collections.Counter()
    per_worker = collections.Counter()
    attrib = unknown = 0
    for f in glob.glob(str(run / "episodes" / "*" / "*.json")):
        m = os.path.getmtime(f)
        if not (t0 <= m <= t1):
            continue
        d = json.load(open(f))
        key = (d.get("task"), d.get("seed"), d.get("arm"))
        wid = owner.get(key)
        if wid is None:
            unknown += 1
            continue
        gpu = wid // (args.nshard // 3) + 1
        per_gpu[gpu] += 1
        per_worker[wid] += 1
        attrib += 1

    hours = (t1 - t0) / 3600.0
    print(f"window {time.strftime('%H:%M:%S', time.localtime(t0))} - "
          f"{time.strftime('%H:%M:%S', time.localtime(t1))}  ({hours:.2f} h)")
    total = sum(per_gpu.values())
    for gpu in sorted(per_gpu):
        n = per_gpu[gpu]
        print(f"  GPU{gpu}: {n:4d} eps = {n / hours:6.1f} eps/h   "
              f"(workers {[w for w in sorted(per_worker) if w // (args.nshard // 3) + 1 == gpu]})")
    print(f"  TOTAL: {total:4d} eps = {total / hours:6.1f} eps/h"
          + (f"   [unattributed {unknown}]" if unknown else ""))
    if per_worker:
        print("  per worker: " + " ".join(f"w{w}={per_worker[w]}" for w in sorted(per_worker)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
