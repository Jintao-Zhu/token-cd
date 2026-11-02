#!/usr/bin/env python3
"""Resplit the REMAINING pi0-FAST WidowX episodes into N balanced worker shards.

Why: the 12-worker layout (4/GPU) left GPU 1/2/3 at single-digit SM%, so the
run was latency-bound rather than compute-bound.  The user approved raising it
to 6 workers per GPU (18 total) on GPU 1/2/3 only.

What counts as "remaining": every (task, seed, arm) triple whose episode JSON is
either absent or not {"status": "complete"} -- the same rule the rollout workers
use to resume, so nothing is lost and nothing is duplicated.

Balancing: units are (task, seed) groups carrying their still-missing arms; the
groups are shuffled with a fixed seed and each group is assigned to the shard
with the fewest episodes so far (LPT-style greedy).  That mixes tasks/arms
across shards and keeps per-shard episode counts within one group of each other.

Writes manifests18/worker_<i>.json and records the change in protocol.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
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
SEEDS = list(range(100))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/pi0fast_bridge_widowx_4task_100seed")
    ap.add_argument("--nshard", type=int, default=18)
    ap.add_argument("--shuffle-seed", type=int, default=20261013)
    ap.add_argument("--apply", action="store_true", help="write manifests18/ and patch protocol.json")
    args = ap.parse_args()

    run = Path(args.run)
    done = set()
    for f in glob.glob(str(run / "episodes" / "*" / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if d.get("status") == "complete":
            done.add((d.get("task"), int(d.get("seed")), d.get("arm")))

    units = []
    for t in TASKS:
        for s in SEEDS:
            missing = [a for a in ARMS if (t, s, a) not in done]
            if missing:
                units.append((t, s, missing))

    total_remaining = sum(len(u[2]) for u in units)
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(units)
    units.sort(key=lambda u: -len(u[2]))          # largest first, stable within size
    shards: list[list[dict]] = [[] for _ in range(args.nshard)]
    load = [0] * args.nshard
    for t, s, missing in units:
        k = min(range(args.nshard), key=lambda i: load[i])
        shards[k].append({"task": t, "seed": s, "arms": missing})
        load[k] += len(missing)

    print(f"complete episodes on disk : {len(done)}")
    print(f"remaining episodes        : {total_remaining}")
    print(f"shards ({args.nshard})            : {load}")
    print(f"max-min shard size        : {max(load) - min(load)}")
    for i, jobs in enumerate(shards):
        arms = {a for j in jobs for a in j["arms"]}
        tasks = {j["task"].replace("widowx_", "")[:9] for j in jobs}
        print(f"  w{i:02d}: {load[i]:4d} eps {len(jobs):3d} units  tasks={len(tasks)} arms={len(arms)}")

    if not args.apply:
        print("\n(dry run -- pass --apply to write manifests18/ and protocol.json)")
        return 0

    mdir = run / f"manifests{args.nshard}"
    mdir.mkdir(parents=True, exist_ok=True)
    for i, jobs in enumerate(shards):
        (mdir / f"worker_{i}.json").write_text(json.dumps({"worker_id": i, "jobs": jobs}, indent=2))

    proto_path = run / "protocol.json"
    proto = json.loads(proto_path.read_text())
    proto["workers"] = f"{args.nshard} ({args.nshard // 3} per GPU, 3 per server)"
    proto["shards"] = load
    proto["resplit"] = {
        "when": "2026-10-13 ~16:40 local",
        "why": "GPU 1/2/3 were latency-bound (SM% single-digit) with 12 workers; user approved 6/GPU",
        "from": "12 shards / seed-block split (manifests/)",
        "to": f"{args.nshard} shards over the REMAINING episodes only (manifests{args.nshard}/)",
        "remaining_episodes_at_resplit": total_remaining,
        "rule": "shard a (task, seed) group to the least-loaded shard; workers still skip status==complete files",
        "dir": f"manifests{args.nshard}/",
    }
    proto_path.write_text(json.dumps(proto, indent=2))
    print(f"\nwrote {args.nshard} manifests to {mdir} and patched {proto_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
