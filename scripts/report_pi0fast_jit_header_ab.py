#!/usr/bin/env python3
"""Compact live/final report for the eager-header vs JIT-header closed-loop pilot."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

DEFAULT_OUT = Path("/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_jit_header_ab_4task_20seed_rtfull_v1")
TASKS = (
    "google_robot_open_drawer",
    "google_robot_pick_coke_can",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
)


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    total_pairs = rescue = harm = first_equal = first_tokens_equal = strict = 0
    call_eager, call_jit = [], []
    rows = []
    for task in TASKS:
        counts = {"l6_eager_header": 0, "l6_jit_header": 0}
        succ = {"l6_eager_header": 0, "l6_jit_header": 0}
        done_pairs = 0
        for seed in range(20):
            e = load(out / "episodes" / task / f"l6_eager_header_seed{seed:03d}.json")
            j = load(out / "episodes" / task / f"l6_jit_header_seed{seed:03d}.json")
            p = load(out / "pairs" / task / f"seed{seed:03d}.json")
            if p:
                if not (e and j and e.get("status") == "complete" and j.get("status") == "complete"):
                    raise RuntimeError(f"pair exists without two complete episode rows: {task}/{seed}")
                done_pairs += 1
                total_pairs += 1
                counts["l6_eager_header"] += 1
                counts["l6_jit_header"] += 1
                succ["l6_eager_header"] += int(bool(p["success_eager"]))
                succ["l6_jit_header"] += int(bool(p["success_jit"]))
                rescue += int(not p["success_eager"] and p["success_jit"])
                harm += int(p["success_eager"] and not p["success_jit"])
                first_equal += int(p["first_action_chunk_equal"])
                first_tokens_equal += int(p["first_selected_tokens_equal"])
                strict += int(p["initial_rgb_equal"] and p["initial_agent_state_equal"])
                call_eager.append(p["mean_call_seconds_eager"])
                call_jit.append(p["mean_call_seconds_jit"])
        fmt = lambda a: f"{succ[a]}/{counts[a]}" if counts[a] else f"0/{counts[a]}"
        rows.append((task, done_pairs, fmt("l6_eager_header"), fmt("l6_jit_header")))

    print(f"Run: {out}")
    print("Task | paired seeds | eager success | jit success")
    for task, n, eager, jit in rows:
        print(f"{task} | {n}/20 | {eager} | {jit}")
    print(f"TOTAL paired={total_pairs}/80; paired Rescue/Harm (JIT vs eager)={rescue}/{harm}; net={rescue-harm:+d}")
    if total_pairs:
        print(f"Initial RGB+state exact pairs={strict}/{total_pairs}; first action chunks equal={first_equal}/{total_pairs}; first selected tokens equal={first_tokens_equal}/{total_pairs}")
    if call_eager and call_jit:
        me, mj = sum(call_eager) / len(call_eager), sum(call_jit) / len(call_jit)
        print(f"Mean inference seconds/replan: eager={me:.4f}, jit={mj:.4f}, ratio={mj/me:.3f}x")
    finished = sum(1 for t in TASKS for s in range(20) if load(out / "pairs" / t / f"seed{s:03d}.json"))
    print(f"Pair manifests written: {finished}/80")


if __name__ == "__main__":
    main()
