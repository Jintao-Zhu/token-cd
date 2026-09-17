"""Resumable 4-worker coordinator for the six-task Prompt-Attn layer extension."""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


TASKS = (
    "google_robot_close_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
ARMS = ("prompt_v1", "prompt_single", "prompt_sparse")
CHUNKS = ((0, 19), (20, 39), (40, 59), (60, 79), (80, 99))
GPU_SLOTS = {1: 1, 2: 4, 3: 4, 4: 1, 5: 1}
GPUS = tuple(GPU_SLOTS)
LAUNCH_STAGGER_SECONDS = 12


@dataclass
class Job:
    task: str
    lo: int
    hi: int
    attempts: int = 0

    @property
    def name(self) -> str:
        return f"{self.task}_{self.lo:03d}_{self.hi:03d}"


def complete(root: Path, job: Job) -> bool:
    return all(
        (root / "episodes" / job.task / arm / f"episode_{seed:03d}_summary.json").exists()
        and (root / "episodes" / job.task / arm / f"episode_{seed:03d}_arrays.npz").exists()
        for seed in range(job.lo, job.hi + 1)
        for arm in ARMS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--layer-artifact", type=Path, required=True)
    parser.add_argument("--closed-loop-v1", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--first3-artifact", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact.resolve()
    repo = Path(__file__).resolve().parents[2]
    logs = root / "rollout_logs"
    logs.mkdir(parents=True, exist_ok=True)
    rollout = repo / "research/semantic_token_cd/prompt_attn_layer_rollout.py"
    analyzer = repo / "research/semantic_token_cd/analyze_prompt_attn_layer_all9.py"
    jobs = [Job(task, *chunk) for task in TASKS for chunk in CHUNKS]
    queue = collections.deque(job for job in jobs if not complete(root, job))
    slots = [(gpu, slot) for gpu, count in GPU_SLOTS.items() for slot in range(count)]
    active: dict[tuple[int, int], tuple] = {}
    available_after = {slot: 0.0 for slot in slots}
    capacity = dict(GPU_SLOTS)
    device_lost_counts = {gpu: 0 for gpu in GPUS}
    failures = []
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": f"{repo}:{pcd}:{env.get('PYTHONPATH', '')}",
        }
    )
    task_csv = ",".join(TASKS)
    print(
        json.dumps(
            {
                "protocol": "PROMPT_ATTN_LAYER_SELECTION_REMAINING6_V1",
                "tasks": list(TASKS),
                "new_arm_episodes": 1800,
                "gpus": list(GPUS),
                "workers": len(slots),
                "queued_jobs": len(queue),
            }
        ),
        flush=True,
    )

    while queue or active:
        for gpu, slot_index in slots:
            slot = (gpu, slot_index)
            if (
                slot_index >= capacity[gpu]
                or slot in active
                or not queue
                or time.monotonic() < available_after[slot]
            ):
                continue
            job = queue.popleft()
            if complete(root, job):
                continue
            job.attempts += 1
            log = logs / f"{job.name}_gpu{gpu}_slot{slot_index}_attempt{job.attempts}.log"
            handle = log.open("a")
            command = [
                sys.executable,
                str(rollout),
                "--artifact", str(root),
                "--layer-artifact", str(args.layer_artifact.resolve()),
                "--closed-loop-v1", str(args.closed_loop_v1.resolve()),
                "--snapshot-artifact", str(args.snapshot_artifact.resolve()),
                "--task", job.task,
                "--seeds", f"{job.lo}-{job.hi}",
                "--gpu", str(gpu),
                "--worker-id", f"remaining6_gpu{gpu}_slot{slot_index}_{job.task}",
                "--config-tasks", task_csv,
                "--arms", ",".join(ARMS),
            ]
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[slot] = (process, job, handle, log)
            print(json.dumps({"started": job.name, "gpu": gpu, "slot": slot_index, "pid": process.pid}), flush=True)
            time.sleep(LAUNCH_STAGGER_SECONDS)

        time.sleep(5)
        for slot, (process, job, handle, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[slot]
            available_after[slot] = time.monotonic() + 90
            if status == 0 and complete(root, job):
                print(json.dumps({"completed": job.name, "gpu": slot[0], "slot": slot[1]}), flush=True)
            elif job.attempts < 12:
                log_tail = log.read_text(errors="replace")[-20000:].lower()
                oom = "out of memory" in log_tail or "cuda error: out of memory" in log_tail
                if status == -6 or oom:
                    device_lost_counts[slot[0]] += 1
                    if slot[0] in (1, 4, 5) and status == -6 and device_lost_counts[slot[0]] >= 2:
                        # Quarantine historically unstable Vulkan devices after
                        # two DeviceLost events.
                        capacity[slot[0]] = 0
                    else:
                        # High-density GPUs 2/3 degrade gracefully 4 -> 3 -> 2
                        # -> 1 on either OOM or DeviceLost.
                        capacity[slot[0]] = max(1, capacity[slot[0]] - 1)
                    for candidate in slots:
                        if candidate[0] == slot[0]:
                            available_after[candidate] = max(available_after[candidate], time.monotonic() + 120)
                queue.append(job)
                print(
                    json.dumps(
                        {
                            "retry": job.name,
                            "status": status,
                            "attempt": job.attempts,
                            "gpu_capacity": capacity[slot[0]],
                            "device_lost_count": device_lost_counts[slot[0]],
                            "oom": oom,
                            "log": str(log),
                        }
                    ),
                    flush=True,
                )
            else:
                failures.append({"job": job.name, "status": status, "log": str(log)})

    missing = [job.name for job in jobs if not complete(root, job)]
    if missing:
        payload = {"complete": False, "missing": missing, "failures": failures}
        (logs / "FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(payload)

    subprocess.run(
        [
            sys.executable,
            str(analyzer),
            "--remaining6-artifact", str(root),
            "--first3-artifact", str(args.first3_artifact.resolve()),
            "--canonical-artifact", str(args.snapshot_artifact.resolve()),
            "--standard-artifact", str(args.closed_loop_v1.resolve()),
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    failed = logs / "FAILED.json"
    if failed.exists():
        failed.unlink()
    (logs / "COMPLETE").write_text("1800/1800 remaining-six attention arm-episodes complete\n")
    print(json.dumps({"complete": True, "new_arm_episodes": 1800}), flush=True)


if __name__ == "__main__":
    main()
