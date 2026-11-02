#!/usr/bin/env python3
"""WORK PLAN for the lambda x budget-rule interaction grid.

Question: does Matched's advantage over fixed / random / attention-mass budgets depend on the
guidance strength lambda?

  lambda  0.5, 1.0
  arms    matched, fixed016, fixed048, random_fixed48, top_p090   (x2 lambdas = 10 arms)
  tasks   google_robot_close_drawer, open_drawer (rt-full depth 8)
          google_robot_pick_coke_can, move_near (RT depth 0)
  seeds   200-299, in the SAME per-process 10-seed blocks as the canonical reference
  size    10 arms x 4 tasks x 100 seeds = 4,000 episodes

Chunking mirrors the reference blocks ([200], [201-210], ..., [291-299]) because the smoke test
showed that results depend on process history (see
docs/OPENVLA_BUDGET_RULE_REPRODUCIBILITY_FINDING.md).

Usage: python3 scripts/plan_openvla_lambda_budget_grid.py [--write]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os
ARTIFACT = Path(os.environ.get("ARTIFACT", "artifacts/openvla_lambda_budget_grid_200_299_v1"))
PROTOCOL = os.environ.get("PROTOCOL", "OPENVLA_LAMBDA_BUDGET_GRID_200_299_V1")
TASKS = ["google_robot_close_drawer", "google_robot_open_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]
RULES = ["matched", "fixed016", "fixed048", "random_fixed48", "top_p090"]
LAMBDAS = [50, 100]
ARMS = [a for a in os.environ.get("ARMS", "").split(",") if a] or \
       [f"{r}_l{lam:03d}" for lam in LAMBDAS for r in RULES]
TASK_SECONDS = {"google_robot_close_drawer": 194.6, "google_robot_open_drawer": 175.6,
                "google_robot_pick_coke_can": 68.5, "google_robot_move_near": 69.9}
ARM_COST = {"matched": 1.00, "fixed016": 0.80, "fixed048": 1.05, "random_fixed48": 1.05,
            "top_p090": 1.20, "taskmean": 0.95}
SEEDS = list(range(200, 300))
REFERENCE_BLOCKS = [[200]] + [[lo + i for i in range(10)] for lo in range(201, 291, 10)] + [[291 + i for i in range(9)]]
NGPU = 3
WORKERS_PER_GPU = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    chunks = []
    for arm in ARMS:
        rule = arm.rsplit("_l", 1)[0]
        cost = ARM_COST.get(rule, 1.0)
        for task in TASKS:
            for block in REFERENCE_BLOCKS:
                est = len(block) * TASK_SECONDS[task] * cost
                chunks.append({"arm": arm, "task": task,
                               "seeds": f"{block[0]}-{block[-1]}" if len(block) > 1 else str(block[0]),
                               "n_episodes": len(block), "estimated_seconds": round(est, 1)})
    chunks.sort(key=lambda c: -c["estimated_seconds"])

    n_workers = NGPU * WORKERS_PER_GPU
    workers = [{"worker": f"g{g}_w{w}", "gpu": g, "chunks": [], "estimated_seconds": 0.0}
               for g in range(1, NGPU + 1) for w in range(1, WORKERS_PER_GPU + 1)]
    for c in chunks:
        w = min(workers, key=lambda x: x["estimated_seconds"])
        w["chunks"].append(c)
        w["estimated_seconds"] += c["estimated_seconds"]

    total_s = sum(w["estimated_seconds"] for w in workers)
    plan = {
        "protocol_id": PROTOCOL,
        "artifact": str(ARTIFACT.resolve()),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "does Matched's advantage over fixed/random/attention-mass budgets depend on lambda?",
        "gpus": list(range(1, NGPU + 1)),
        "workers_per_gpu": WORKERS_PER_GPU,
        "workers": n_workers,
        "tasks": TASKS,
        "seeds": SEEDS,
        "arms": ARMS,
        "chunk_mode": "reference",
        "chunk_source": "same per-process 10-seed blocks as artifacts/openvla_lambda1_extension_200_299_v1",
        "invocations": len(chunks),
        "total_episode_evaluations": sum(c["n_episodes"] for c in chunks),
        "estimated_total_worker_hours": round(total_s / 3600.0, 2),
        "estimated_wall_hours_without_load": round(max(w["estimated_seconds"] for w in workers) / 3600.0, 2),
        "rendering": {"google_robot_close_drawer": "rt-full depth 8", "google_robot_open_drawer": "rt-full depth 8",
                      "google_robot_pick_coke_can": "rt depth 0", "google_robot_move_near": "rt depth 0"},
        "assignment": [{**w, "estimated_hours": round(w["estimated_seconds"] / 3600.0, 2)} for w in workers],
    }
    print(f"arms={len(ARMS)} lambdas={LAMBDAS} tasks={len(TASKS)} episodes={plan['total_episode_evaluations']} "
          f"chunks={len(chunks)} workers={n_workers}")
    print(f"estimated worker-hours {plan['estimated_total_worker_hours']} -> "
          f"wall ~{plan['estimated_wall_hours_without_load']} h (+ model loads)")
    for w in plan["assignment"]:
        print(f"  {w['worker']:14s} gpu{w['gpu']} chunks={len(w['chunks']):3d} "
              f"eps={sum(c['n_episodes'] for c in w['chunks']):4d} est={w['estimated_hours']:.2f} h")
    if args.write:
        ARTIFACT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT / "WORK_PLAN.json").write_text(json.dumps(plan, indent=2))
        print(f"\nwrote {ARTIFACT/'WORK_PLAN.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
