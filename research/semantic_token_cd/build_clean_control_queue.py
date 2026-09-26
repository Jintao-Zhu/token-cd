#!/usr/bin/env python3
"""Queue the CLEAN continuation control.

For every fork state, run the vanilla policy from that state to the episode
horizon with no intervention.  This is the correct same-batch null: comparing a
branch against the *stored* vanilla outcome is invalid because the clean
continuation itself only reproduces the stored outcome about 60% of the time
(physics chaos amplification flips contact-critical events).

Cheap: one continuation per fork, no branch decoding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    pending = args.out_dir / "cases" / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    count = 0
    for _task_key, entry in manifest.items():
        for fork in entry["forks"]:
            case_id = f"clean_t{fork['task_id']:02d}_i{fork['init_state_id']:03d}_{fork['stage']}"
            payload = {
                "case_id": case_id,
                "task_id": fork["task_id"],
                "task_name": fork["task_name"],
                "init_state_id": fork["init_state_id"],
                "stage": fork["stage"],
                "fork_step": fork["fork_step"],
                "episode_len": fork["episode_len"],
                "episode_file": fork["episode_file"],
                "branches": ["clean"],
                "durations": [1],
                "control": "clean_continuation",
            }
            (pending / f"{case_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
            count += 1
    print(f"queued {count} clean-continuation controls into {pending}")


if __name__ == "__main__":
    main()
