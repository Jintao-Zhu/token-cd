#!/usr/bin/env python3
"""Build the SAME_STATE_FORK_V1 state manifest from vanilla trajectories.

Main state library comes from VANILLA trajectories only, so the analysis is not
restricted to states that Matched itself induced.  Sampling is deterministic:
lowest init_state_id first, fixed stage fractions, no cherry-picking.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from research.semantic_token_cd.same_state_fork_common import STAGE_FRACTIONS, atomic_json, select_stage_steps

EPISODES_PER_TASK = 40
MIN_EPISODE_LEN = 8


def libero_episodes(root: Path, task_id: int) -> list[dict]:
    rows = []
    for path in glob.glob(str(root / "episodes" / "*" / "vanilla" / "*.json")):
        payload = json.loads(Path(path).read_text())
        if payload.get("task_id") != task_id:
            continue
        if payload.get("arm") != "vanilla":
            continue
        payload["_file"] = path
        rows.append(payload)
    rows.sort(key=lambda r: int(r["init_state_id"]))
    return rows


def build_libero(root: Path, tasks: list[int], out: Path) -> dict:
    manifest = {}
    for task_id in tasks:
        rows = libero_episodes(root, task_id)
        chosen, stages = [], []
        for payload in rows:
            if len(chosen) >= EPISODES_PER_TASK:
                break
            if int(payload.get("steps", 0)) < MIN_EPISODE_LEN:
                continue
            entry = {
                "task_id": int(task_id),
                "task_name": payload["task_name"],
                "instruction": payload["instruction"],
                "init_state_id": int(payload["init_state_id"]),
                "episode_len": int(payload["steps"]),
                "episode_success": bool(payload["success"]),
                "init_state_sha256": payload["init_state_sha256"],
                "episode_file": str(payload["_file"]),
                "stages": select_stage_steps(int(payload["steps"])),
            }
            chosen.append(entry)
            for stage in entry["stages"]:
                if stage["status"] == "ok":
                    stages.append(
                        {
                            "task_id": int(task_id),
                            "task_name": payload["task_name"],
                            "init_state_id": int(payload["init_state_id"]),
                            "episode_len": int(payload["steps"]),
                            "stage": stage["stage"],
                            "fraction": stage["fraction"],
                            "fork_step": stage["step"],
                            "episode_file": str(payload["_file"]),
                        }
                    )
        manifest[str(task_id)] = {
            "task_name": rows[0]["task_name"] if rows else None,
            "episodes_available": len(rows),
            "episodes_selected": len(chosen),
            "forks": stages,
            "episodes": chosen,
        }
    atomic_json(out, manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", default="3,10,49,72,73")
    args = ap.parse_args()

    tasks = [int(x) for x in args.tasks.split(",") if x]
    manifest = build_libero(args.formal_root.resolve(), tasks, args.out.resolve())

    total_forks = 0
    print(f"{'task':<6}{'avail':<7}{'sel':<6}{'forks':<7}{'stage mix'}")
    for key, value in manifest.items():
        forks = value["forks"]
        total_forks += len(forks)
        mix = {}
        for fork in forks:
            mix[fork["stage"]] = mix.get(fork["stage"], 0) + 1
        print(f"{key:<6}{value['episodes_available']:<7}{value['episodes_selected']:<6}{len(forks):<7}{mix}")
    print(f"\nTOTAL FORKS: {total_forks}")
    print(f"expected runs (2 branches x 6 durations): {total_forks * 12}")


if __name__ == "__main__":
    main()
