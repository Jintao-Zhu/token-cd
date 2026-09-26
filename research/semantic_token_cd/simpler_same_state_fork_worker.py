#!/usr/bin/env python3
"""SAME_STATE_FORK_V1 worker for SIMPLER (Google Robot).

Mirrors the LIBERO worker: restore a vanilla state, compare branches at that
exact state, run ``d`` intervention steps, then hand back to vanilla.
States come from the canonical vanilla snapshots.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from research.semantic_token_cd.same_state_fork_common import (
    action_deltas, atomic_json, resolve_duration, select_stage_steps,
)

TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
EPISODES_PER_TASK = 40


def load_episode(root: Path, task: str, seed: int) -> dict:
    summary = json.loads((root / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json").read_text())
    arrays = np.load(root / "episodes" / task / "vanilla" / f"episode_{seed:03d}_arrays.npz")
    summary["executed_actions"] = arrays["executed_actions"]
    return summary


def build_manifest(root: Path) -> dict:
    manifest = {}
    for task in TASKS:
        summaries = []
        for path in sorted((root / "episodes" / task / "vanilla").glob("episode_*_summary.json")):
            payload = json.loads(path.read_text())
            summaries.append(payload)
        summaries.sort(key=lambda r: int(r["seed"]))
        chosen = summaries[:EPISODES_PER_TASK]
        forks = []
        for payload in chosen:
            length = int(payload["control_steps"])
            for stage in select_stage_steps(length):
                if stage["status"] != "ok":
                    continue
                forks.append({
                    "task": task, "seed": int(payload["seed"]), "stage": stage["stage"],
                    "fraction": stage["fraction"], "fork_step": stage["step"], "episode_len": length,
                })
        manifest[task] = {
            "episodes_available": len(summaries), "episodes_selected": len(chosen), "forks": forks,
            "vanilla_success": {str(p["seed"]): bool(p["success"]) for p in chosen},
        }
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--durations", default="1,5,15,30,50,-1")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--build-manifest", action="store_true")
    args = ap.parse_args()
    args.durations = tuple(int(x) for x in args.durations.split(",") if x)

    if args.build_manifest:
        manifest = build_manifest(args.canonical.resolve())
        atomic_json(args.artifact / "SIMPLER_STATE_MANIFEST.json", manifest)
        total = sum(len(v["forks"]) for v in manifest.values())
        print(f"tasks={len(manifest)} forks={total} runs={total * 2 * len(args.durations)}")
        for task, value in manifest.items():
            print(f"  {task:<30} episodes={value['episodes_selected']:<4} forks={len(value['forks'])}")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print("simpler fork worker ready (state restore + branch execution)", flush=True)


if __name__ == "__main__":
    main()
