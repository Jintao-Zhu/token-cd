"""Freeze donor-budget schedules for the L11 matched-budget causal test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/prompt_attn_l11_token_count_v1/episodes"
OUTPUT = ROOT / "artifacts/l11_matched_budget_causal_pilot_v1/BUDGET_SCHEDULES.json"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SEEDS = tuple(range(100, 200))
RANDOM_SEED = 20260914


def digest(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int16).tobytes()).hexdigest()


def main() -> None:
    matched: dict[str, dict[str, list[int]]] = {}
    slots: list[tuple[str, int, int]] = []
    values: list[int] = []
    for task in TASKS:
        matched[task] = {}
        for seed in SEEDS:
            path = SOURCE / task / "l11_matched" / f"episode_{seed:03d}_summary.json"
            summary = json.loads(path.read_text())
            counts = [int(record["m_t"]) for record in summary["selector_trace"]]
            matched[task][str(seed)] = counts
            for step, count in enumerate(counts):
                slots.append((task, seed, step))
                values.append(count)

    rng = np.random.default_rng(RANDOM_SEED)
    global_values = np.asarray(values, dtype=int)[rng.permutation(len(values))].tolist()
    global_schedule = {task: {str(seed): [0] * len(matched[task][str(seed)]) for seed in SEEDS} for task in TASKS}
    for (task, seed, step), count in zip(slots, global_values):
        global_schedule[task][str(seed)][step] = int(count)

    within_schedule = {task: {str(seed): [0] * len(matched[task][str(seed)]) for seed in SEEDS} for task in TASKS}
    for task_index, task in enumerate(TASKS):
        task_slots = [(seed, step) for seed in SEEDS for step in range(len(matched[task][str(seed)]))]
        task_values = [matched[task][str(seed)][step] for seed, step in task_slots]
        task_rng = np.random.default_rng(RANDOM_SEED + 1000 + task_index)
        shuffled = np.asarray(task_values, dtype=int)[task_rng.permutation(len(task_values))].tolist()
        for (seed, step), count in zip(task_slots, shuffled):
            within_schedule[task][str(seed)][step] = int(count)

    episode_fixed = {
        task: {
            str(seed): [matched[task][str(seed)][0]] * len(matched[task][str(seed)])
            for seed in SEEDS
        }
        for task in TASKS
    }
    checks = {
        "global_distribution_exact": sorted(values) == sorted(
            count for task in TASKS for seed in SEEDS for count in global_schedule[task][str(seed)]
        ),
        "within_task_distribution_exact": {
            task: sorted(count for seed in SEEDS for count in matched[task][str(seed)])
            == sorted(count for seed in SEEDS for count in within_schedule[task][str(seed)])
            for task in TASKS
        },
        "episode_fixed_exact": all(
            len(set(episode_fixed[task][str(seed)])) == 1
            and episode_fixed[task][str(seed)][0] == matched[task][str(seed)][0]
            for task in TASKS for seed in SEEDS
        ),
    }
    if not checks["global_distribution_exact"] or not all(checks["within_task_distribution_exact"].values()) or not checks["episode_fixed_exact"]:
        raise RuntimeError(checks)
    payload = {
        "protocol_id": "PROMPT_ATTN_L11_MATCHED_BUDGET_CAUSAL_PILOT_V1",
        "created_date": "2026-09-14",
        "source": str(SOURCE),
        "random_seed": RANDOM_SEED,
        "tasks": list(TASKS),
        "seeds": [100, 199],
        "definitions": {
            "global_shuffle": "Exact permutation of all 38,600 matched step counts across the four-task 100-seed slot grid.",
            "within_task_shuffle": "Exact permutation of matched step counts within each task's 100-seed slot grid.",
            "episode_fixed": "Repeat the same task/seed matched first-step count for the full canonical horizon.",
        },
        "source_count_sha256": digest(values),
        "global_count_sha256": digest(global_values),
        "checks": checks,
        "schedules": {
            "global_shuffle": global_schedule,
            "within_task_shuffle": within_schedule,
            "episode_fixed": episode_fixed,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "slots": len(values), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
