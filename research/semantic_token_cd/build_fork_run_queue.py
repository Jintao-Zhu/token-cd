#!/usr/bin/env python3
"""Expand the fork state manifest into a run-level queue.

Run-level (not fork-level) claiming means partial progress stays analyzable.
Ordering puts the highest-information, cheapest conditions first:
  wave 1: d in {1, 15, rem}   (dose-response endpoints + full episode)
  wave 2: d in {5, 30, 50}
Branches: reconstruction and guided.  Clean is reused from the vanilla episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WAVES = {
    1: (1, 5, 15, 30, 50, -1),
    2: (1, 5, 15, 30, 50, -1),
}
BRANCHES = ("reconstruction", "guided")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wave", type=int, default=1)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    durations = WAVES[args.wave]
    pending = args.out_dir / "cases" / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    count = 0
    for task_key, entry in manifest.items():
        for fork in entry["forks"]:
            # One job per (state, duration): the worker replays once, computes the
            # branch actions and diagnostics once, then runs BOTH branches for that
            # duration.  This removes the repeated replay that dominated runtime.
            case_id = f"t{fork['task_id']:02d}_i{fork['init_state_id']:03d}_{fork['stage']}"
            payload = {
                "case_id": case_id,
                "task_id": fork["task_id"],
                "task_name": fork["task_name"],
                "init_state_id": fork["init_state_id"],
                "stage": fork["stage"],
                "fork_step": fork["fork_step"],
                "episode_len": fork["episode_len"],
                "episode_file": fork["episode_file"],
                "branches": list(BRANCHES),
                "durations": list(durations),
            }
            (pending / f"{case_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
            count += 1
    print(f"wave {args.wave}: {count} runs queued into {pending}")
    print(f"durations={durations} branches={BRANCHES}")


if __name__ == "__main__":
    main()
