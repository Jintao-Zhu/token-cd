#!/usr/bin/env python3
"""Build the WORK PLAN for the OpenVLA budget-rule study (Matched vs TopP vs Fixed, lambda=1.0).

Workload: 11 arms x 9 SIMPLER tasks x 100 seeds (200-299) = 9,900 episodes, run on GPU 1/2/3
with 4 workers per GPU (12 workers).  The matched reference (900 episodes) already exists as
`artifacts/openvla_lambda1_extension_200_299_v1` and is NOT re-run.

Chunks are (arm, task, seed-block) with 25 seeds per block, so a worker never waits hours for a
single arm/task; blocks are then packed onto workers with a longest-processing-time-first greedy
rule using the per-task episode cost measured on the reference run and a per-arm cost multiplier
for the budget size (large budgets guide more tokens per step, so they cost more).

The cost model is for LOAD BALANCING only -- never for reporting.

Usage: python3 scripts/plan_openvla_budget_rule_run.py [--write]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ARTIFACT = Path("artifacts/openvla_budget_rule_9task_200_299_v1")
PROTOCOL = "OPENVLA_BUDGET_RULE_9TASK_200_299_V1"
TASKS = [
    "google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can",
    "google_robot_move_near", "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
]
# mean episode seconds measured on the lambda=1.0 matched reference (891 episodes)
TASK_SECONDS = {
    "google_robot_place_apple_in_closed_top_drawer": 177.1,
    "google_robot_open_drawer": 168.7,
    "google_robot_close_drawer": 167.1,
    "widowx_put_eggplant_in_basket": 101.8,
    "google_robot_move_near": 69.9,
    "google_robot_pick_coke_can": 68.5,
    "widowx_carrot_on_plate": 49.1,
    "widowx_stack_cube": 47.4,
    "widowx_spoon_on_towel": 45.1,
}
ARMS = ["top_p075_l100", "top_p080_l100", "top_p085_l100", "top_p090_l100",
        "fixed016_l100", "fixed024_l100", "fixed032_l100", "fixed040_l100",
        "fixed048_l100", "fixed056_l100", "fixed064_l100"]
# per-arm multiplier on the matched cost (matched mean budget is 36.7 tokens per step)
ARM_COST = {
    "top_p075_l100": 1.00, "top_p080_l100": 1.05, "top_p085_l100": 1.10, "top_p090_l100": 1.20,
    "fixed016_l100": 0.80, "fixed024_l100": 0.85, "fixed032_l100": 0.95, "fixed040_l100": 1.00,
    "fixed048_l100": 1.05, "fixed056_l100": 1.15, "fixed064_l100": 1.30,
}
SEEDS = list(range(200, 300))
# The lambda=1.0 reference (matched) was produced with these exact per-process blocks:
# seed 200 alone (its preflight) and then 10-seed blocks 201-210 ... 291-299.  The smoke test
# showed that results depend on the process history, so the new arms MUST use the same blocks.
REFERENCE_BLOCKS = [[200]] + [[lo + i for i in range(10)] for lo in range(201, 291, 10)] + [[291, 292, 293, 294, 295, 296, 297, 298, 299]]
BLOCK = 10
NGPU = 3
WORKERS_PER_GPU = 4


def blocks_for(mode: str) -> list[list[int]]:
    if mode == "reference":
        return REFERENCE_BLOCKS
    return [SEEDS[i:i + BLOCK] for i in range(0, len(SEEDS), BLOCK)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--chunk-mode", choices=("reference", "uniform"), default="reference",
                    help="reference = the exact per-process seed blocks of the lambda=1.0 Ours run")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    blocks = blocks_for(args.chunk_mode)
    chunks = []
    for arm in arms:
        for task in TASKS:
            for block in blocks:
                estimate = len(block) * TASK_SECONDS[task] * ARM_COST[arm]
                chunks.append({"arm": arm, "task": task,
                               "seeds": f"{block[0]}-{block[-1]}" if len(block) > 1 else str(block[0]),
                               "n_episodes": len(block),
                               "estimated_seconds": round(estimate, 1)})
    chunks.sort(key=lambda c: -c["estimated_seconds"])

    n_workers = NGPU * WORKERS_PER_GPU
    workers = [{"worker": f"openvla_br_g{g}_w{w}", "gpu": g, "chunks": [], "estimated_seconds": 0.0}
               for g in range(1, NGPU + 1) for w in range(1, WORKERS_PER_GPU + 1)]
    for c in chunks:                                   # LPT packing
        w = min(workers, key=lambda x: x["estimated_seconds"])
        w["chunks"].append(c)
        w["estimated_seconds"] += c["estimated_seconds"]

    total_s = sum(w["estimated_seconds"] for w in workers)
    plan = {
        "protocol_id": f"{PROTOCOL}_SMOKE" if len(arms) < len(ARMS) else PROTOCOL,
        "artifact": str(ARTIFACT.resolve()),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpus": list(range(1, NGPU + 1)),
        "workers_per_gpu": WORKERS_PER_GPU,
        "workers": n_workers,
        "tasks": TASKS,
        "seeds": SEEDS,
        "seed_blocks_of": BLOCK,
        "chunk_mode": args.chunk_mode,
        "chunk_source": "the exact per-process seed blocks of artifacts/openvla_lambda1_extension_200_299_v1"
                        if args.chunk_mode == "reference" else "uniform blocks",
        "invocations": len(chunks),
        "arms": arms,
        "reused_reference": "artifacts/openvla_lambda1_extension_200_299_v1 (matched, lambda=1.0, 900 episodes)",
        "total_episode_evaluations": sum(c["n_episodes"] for c in chunks),
        "estimated_total_worker_hours": round(total_s / 3600.0, 2),
        "estimated_wall_hours": round(max(w["estimated_seconds"] for w in workers) / 3600.0, 2),
        "cost_model_note": "per-task seconds measured on the lambda=1.0 matched reference; "
                           "per-arm multipliers are balancing guesses only",
        "assignment": [{**w, "estimated_hours": round(w["estimated_seconds"] / 3600.0, 2)}
                       for w in workers],
    }
    print(f"arms={len(arms)}  chunks={len(chunks)}  episodes={plan['total_episode_evaluations']}  "
          f"workers={n_workers}")
    print(f"estimated worker-hours total = {plan['estimated_total_worker_hours']}  "
          f"-> wall clock ~{plan['estimated_wall_hours']} h")
    for w in plan["assignment"]:
        print(f"  {w['worker']:18s} gpu{w['gpu']}  chunks={len(w['chunks']):3d}  "
              f"eps={sum(c['n_episodes'] for c in w['chunks']):4d}  est={w['estimated_hours']:.2f} h")
    if args.write:
        ARTIFACT.mkdir(parents=True, exist_ok=True)
        out = ARTIFACT / "WORK_PLAN.json"
        out.write_text(json.dumps(plan, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
