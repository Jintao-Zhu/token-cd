#!/usr/bin/env python3
"""Create the locked 50-pair task-4 manifest without reading outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LANGUAGE = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--artifact", type=Path, required=True); args = p.parse_args(); output = args.artifact / "episode_manifest.jsonl"
    if output.exists(): raise FileExistsError(output)
    rows = []
    for init_state in range(50):
        for condition in ("vanilla", "consistency_guided"):
            rows.append({"action_noise_seed": 65100000 + init_state, "condition": condition, "episode_id": f"task04__init{init_state:02d}__{condition}", "init_state_id": init_state, "language": LANGUAGE, "pair_id": f"task04__init{init_state:02d}", "reset_seed": 65000000 + init_state, "split": "development", "suite": "libero_spatial", "task_id": 4})
    with output.open("x") as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
    assert len(rows) == 100 and len({row["episode_id"] for row in rows}) == 100
    for init_state in range(50):
        pair = [row for row in rows if row["init_state_id"] == init_state]
        assert len(pair) == 2 and pair[0]["reset_seed"] == pair[1]["reset_seed"] and pair[0]["action_noise_seed"] == pair[1]["action_noise_seed"]
    print(json.dumps({"episodes": 100, "pairs": 50, "init_states": [0, 49], "complete": True}))


if __name__ == "__main__": main()
