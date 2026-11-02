#!/usr/bin/env python3
"""Fixed K16/K32 lambda=1 plan; --write creates WORK_PLAN.json, never a lock.

No rollouts are launched. Existing K8 and budget arms are reused only after the
separate control audit approves them. Cost weights are LPT balancing heuristics,
NOT runtime predictions. Each chunk is one arm/task/reference seed block/process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts/openvla_kmeans_budget_9task_200_299_v1"
SMOKE_ARTIFACT = REPO / "artifacts/openvla_kmeans_budget_smoke_v1"
PROTOCOL = "OPENVLA_KMEANS_BUDGET_9TASK_200_299_V1"
SMOKE_PROTOCOL = "OPENVLA_KMEANS_BUDGET_SMOKE_V1"
ARMS = ["matched_k16_l100", "matched_k32_l100"]
SMOKE_ARMS = ["matched_k08_l100", *ARMS]
TASKS = [
    "google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can",
    "google_robot_move_near", "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
]
SMOKE_TASKS = ["google_robot_close_drawer", "google_robot_pick_coke_can", "widowx_carrot_on_plate"]
TASK_SECONDS = {
    "google_robot_place_apple_in_closed_top_drawer": 177.1,
    "google_robot_open_drawer": 168.7, "google_robot_close_drawer": 167.1,
    "widowx_put_eggplant_in_basket": 101.8, "google_robot_move_near": 69.9,
    "google_robot_pick_coke_can": 68.5, "widowx_carrot_on_plate": 49.1,
    "widowx_stack_cube": 47.4, "widowx_spoon_on_towel": 45.1,
}
SEEDS = list(range(200, 300))
REFERENCE_BLOCKS = [[200]] + [list(range(lo, lo + 10)) for lo in range(201, 291, 10)] + [list(range(291, 300))]
GPUS = [1, 2, 3]
WORKERS_PER_GPU = 4


def build_plan() -> dict:
    chunks = []
    for arm in ARMS:
        for task in TASKS:
            for block in REFERENCE_BLOCKS:
                seeds = str(block[0]) if len(block) == 1 else f"{block[0]}-{block[-1]}"
                chunks.append({
                    "chunk_id": f"{arm}__{task}__{seeds}", "arm": arm, "task": task,
                    "seeds": seeds, "n_episodes": len(block),
                    "balance_weight": round(len(block) * TASK_SECONDS[task], 1),
                })
    chunks.sort(key=lambda chunk: -chunk["balance_weight"])
    workers = [{"worker": f"ovkm_g{gpu}_w{slot}", "gpu": gpu, "chunks": [], "balance_weight": 0.0}
               for gpu in GPUS for slot in range(1, WORKERS_PER_GPU + 1)]
    for chunk in chunks:
        worker = min(workers, key=lambda item: item["balance_weight"])
        worker["chunks"].append(chunk)
        worker["balance_weight"] = round(worker["balance_weight"] + chunk["balance_weight"], 1)
    return {
        "schema_version": 1, "protocol_id": PROTOCOL, "artifact": str(ARTIFACT),
        "gpus": GPUS, "workers_per_gpu": WORKERS_PER_GPU, "workers": len(workers),
        "tasks": TASKS, "seeds": SEEDS, "arms": ARMS, "lambda": 1.0,
        "reference_blocks": REFERENCE_BLOCKS, "chunk_mode": "reference",
        "chunk_source": "artifacts/openvla_lambda1_extension_200_299_v1",
        "invocations": len(chunks), "total_episode_evaluations": sum(c["n_episodes"] for c in chunks),
        "assignment_method": "stable longest-processing-time-first (LPT), least-loaded worker",
        "cost_model_note": "Reference task seconds, equal K16/K32 multipliers: balancing heuristics only, not ETA.",
        "comparison_policy": "Reuse ALL existing budget arms and matched K8 only as approved by CONTROL_REUSE_AUDIT.json; do not rerun controls in the full plan.",
        "resume_policy": "No retries or partial-block resume; supervisor refuses rerun after any launch.",
        "smoke": {"artifact": str(SMOKE_ARTIFACT), "protocol_id": SMOKE_PROTOCOL,
                  "tasks": SMOKE_TASKS, "arms": SMOKE_ARMS, "seeds": [200],
                  "gpu": 1, "episodes": 9, "sequential_separate_processes": True},
        "assignment": workers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create fixed WORK_PLAN.json exclusively; never overwrite")
    parser.add_argument("--json", action="store_true", help="Print complete plan instead of summary")
    args = parser.parse_args()
    plan = build_plan()
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"arms=2 tasks=9 episodes={plan['total_episode_evaluations']} chunks={plan['invocations']} workers=12 GPUs=1,2,3")
        for worker in plan["assignment"]:
            print(f"{worker['worker']} gpu={worker['gpu']} chunks={len(worker['chunks'])} episodes={sum(c['n_episodes'] for c in worker['chunks'])}")
        print("Balance weights are scheduling heuristics, not ETAs. No experiment launched.")
    if args.write:
        ARTIFACT.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT / "WORK_PLAN.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(plan, indent=2) + "\n")
        print(f"Wrote {path}; immutable locks and approval gates must be supplied separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
