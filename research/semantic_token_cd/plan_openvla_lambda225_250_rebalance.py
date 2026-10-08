from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("artifacts/openvla_global_lambda_grid_225_250_0_199_v1")
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer", "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube",
)
ARMS = {"lambda_225": 2.25, "lambda_250": 2.50}
FALLBACK_SECONDS = {
    "google_robot_open_drawer": 165.0, "google_robot_close_drawer": 160.0,
    "google_robot_pick_coke_can": 55.0, "google_robot_move_near": 55.0,
    "google_robot_place_apple_in_closed_top_drawer": 210.0,
    "widowx_carrot_on_plate": 50.0, "widowx_put_eggplant_in_basket": 100.0,
    "widowx_spoon_on_towel": 50.0, "widowx_stack_cube": 100.0,
}
complete: dict[tuple[str, int, str], dict] = {}
times: dict[str, list[float]] = defaultdict(list)
for task in TASKS:
    for arm, expected_lambda in ARMS.items():
        base = ROOT / "episodes" / task / arm
        for summary in base.glob("episode_*_summary.json"):
            seed = int(summary.stem.split("_")[1])
            arrays = summary.with_name(summary.name.replace("_summary.json", "_arrays.npz"))
            if not arrays.is_file():
                continue
            data = json.loads(summary.read_text())
            if (data.get("lambda") != expected_lambda or data.get("technical_pass") is not True
                    or data.get("task") != task or int(data.get("seed", -1)) != seed):
                continue
            complete[(task, seed, arm)] = data
            runtime = float(data.get("runtime_seconds", 0.0))
            if runtime > 0:
                times[task].append(runtime)

jobs = []
for task in TASKS:
    mean = sorted(times[task])[len(times[task]) // 2] if times[task] else FALLBACK_SECONDS[task]
    for seed in range(200):
        missing = [arm for arm in ARMS if (task, seed, arm) not in complete]
        if missing:
            jobs.append((mean * len(missing), task, seed, missing))
jobs.sort(reverse=True)
bins = [[] for _ in range(15)]
loads = [0.0] * 15
for weight, task, seed, missing in jobs:
    idx = min(range(15), key=lambda i: loads[i])
    bins[idx].append((task, seed, missing))
    loads[idx] += weight

tsv = ROOT / "WORK_PLAN_REBALANCED_G12345.tsv"
with tsv.open("w") as f:
    for idx, entries in enumerate(bins):
        gpu = 1 + idx // 3
        worker = f"lambda225_g{gpu}_w{idx % 3 + 1}_rebalance"
        grouped: dict[str, list[int]] = defaultdict(list)
        for task, seed, _missing in entries:
            grouped[task].append(seed)
        args = [f"{task}={','.join(map(str, sorted(seeds)))}" for task, seeds in grouped.items()]
        f.write("\t".join([worker, str(gpu), *args]) + "\n")

manifest = {
    "protocol_id": "OPENVLA_GLOBAL_LAMBDA_GRID_0_199_V1",
    "artifact": str(ROOT.resolve()), "gpus": [1, 2, 3, 4, 5],
    "workers_per_gpu": 3, "workers": 15, "tasks": list(TASKS),
    "seeds": list(range(200)), "lambdas": list(ARMS.values()),
    "target_episode_arm_evaluations": 3600,
    "completed_valid_arm_evaluations": len(complete),
    "remaining_arm_evaluations": sum(len(job[3]) for job in jobs),
    "remaining_task_seed_pairs": len(jobs),
    "worker_load_estimate_hours": [round(x / 3600, 2) for x in loads],
    "assignments": [
        {"worker": f"lambda225_g{1 + idx // 3}_w{idx % 3 + 1}_rebalance",
         "gpu": 1 + idx // 3, "estimated_hours": round(loads[idx] / 3600, 2),
         "task_seed_pairs": len(entries)} for idx, entries in enumerate(bins)
    ],
}
(ROOT / "WORK_PLAN_REBALANCED_G12345.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
