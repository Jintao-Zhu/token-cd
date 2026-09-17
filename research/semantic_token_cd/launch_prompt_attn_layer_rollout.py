"""Resumable one-renderer-per-GPU coordinator for the 3x100 layer screen."""
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


TASKS = ("google_robot_open_drawer", "google_robot_pick_coke_can", "google_robot_move_near")
ARMS = ("prompt_single", "prompt_sparse")
CHUNKS = ((0, 19), (20, 39), (40, 59), (60, 79), (80, 99))
# GPUs 1/4/5 repeatedly enter a Vulkan DeviceLost state after renderer teardown
# on this host.  GPUs 2/3 have completed cross-task transitions reliably.
GPUS = (2, 3)
SLOTS_PER_GPU = 2


@dataclass
class Job:
    task: str
    lo: int
    hi: int
    attempts: int = 0

    @property
    def name(self):
        return f"{self.task}_{self.lo:03d}_{self.hi:03d}"


def complete(root: Path, job: Job) -> bool:
    return all(
        (root / "episodes" / job.task / arm / f"episode_{seed:03d}_summary.json").exists()
        and (root / "episodes" / job.task / arm / f"episode_{seed:03d}_arrays.npz").exists()
        for seed in range(job.lo, job.hi + 1) for arm in ARMS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--layer-artifact", type=Path, required=True)
    parser.add_argument("--closed-loop-v1", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve(); repo = Path(__file__).resolve().parents[2]
    logs = root / "rollout_logs"; logs.mkdir(parents=True, exist_ok=True)
    rollout = repo / "research/semantic_token_cd/prompt_attn_layer_rollout.py"
    analyzer = repo / "research/semantic_token_cd/analyze_prompt_attn_layer_closed_loop.py"
    jobs = [Job(task, *chunk) for task in TASKS for chunk in CHUNKS]
    queue = collections.deque(job for job in jobs if not complete(root, job))
    active = {}; failures = []
    slots = [(gpu, slot) for gpu in GPUS for slot in range(SLOTS_PER_GPU)]
    available_after = {key: 0.0 for key in slots}
    capacity = {gpu: SLOTS_PER_GPU for gpu in GPUS}
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": f"{repo}:{pcd}:{env.get('PYTHONPATH','')}"})
    print(json.dumps({"protocol": "PROMPT_ATTN_LAYER_SELECTION_V1", "new_episodes": 600,
                      "gpus": list(GPUS), "workers": len(slots), "queued_jobs": len(queue),
                      "fallback": "reduce to one slot/GPU after Vulkan DeviceLost"}), flush=True)
    while queue or active:
        for gpu, slot in slots:
            key = (gpu, slot)
            if slot >= capacity[gpu] or key in active or not queue or time.monotonic() < available_after[key]:
                continue
            job = queue.popleft()
            if complete(root, job):
                continue
            job.attempts += 1
            log = logs / f"{job.name}_gpu{gpu}_slot{slot}_attempt{job.attempts}.log"
            handle = log.open("a")
            command = [sys.executable, str(rollout), "--artifact", str(root),
                       "--layer-artifact", str(args.layer_artifact.resolve()),
                       "--closed-loop-v1", str(args.closed_loop_v1.resolve()),
                       "--snapshot-artifact", str(args.snapshot_artifact.resolve()),
                       "--task", job.task, "--seeds", f"{job.lo}-{job.hi}",
                       "--gpu", str(gpu), "--worker-id", f"gpu{gpu}_slot{slot}_{job.task}"]
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = (process, job, handle, log)
            print(json.dumps({"started": job.name, "gpu": gpu, "slot": slot, "pid": process.pid}), flush=True)
            time.sleep(20)
        time.sleep(5)
        for key, (process, job, handle, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close(); del active[key]
            # SAPIEN/Vulkan can retain device resources briefly after process
            # exit.  Reopening immediately caused vk::DeviceLost; enforce a
            # full cleanup interval before this physical GPU is reused.
            available_after[key] = time.monotonic() + 90
            if status == 0 and complete(root, job):
                print(json.dumps({"completed": job.name, "gpu": key[0], "slot": key[1]}), flush=True)
            elif job.attempts < 12:
                if status == -6:
                    capacity[key[0]] = 1
                    for candidate in slots:
                        if candidate[0] == key[0]:
                            available_after[candidate] = max(available_after[candidate], time.monotonic() + 120)
                queue.append(job)
                print(json.dumps({"retry": job.name, "status": status, "attempt": job.attempts,
                                  "log": str(log), "gpu_capacity": capacity[key[0]]}), flush=True)
            else:
                failures.append({"job": job.name, "status": status, "log": str(log)})
    missing = [job.name for job in jobs if not complete(root, job)]
    if missing:
        payload = {"complete": False, "missing": missing, "failures": failures}
        (logs / "FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(payload)
    subprocess.run([sys.executable, str(analyzer), "--artifact", str(root),
                    "--layer-artifact", str(args.layer_artifact.resolve()),
                    "--closed-loop-v1", str(args.closed_loop_v1.resolve()),
                    "--canonical-artifact", str(args.snapshot_artifact.resolve())],
                   cwd=repo, env=env, check=True)
    failed_marker = logs / "FAILED.json"
    if failed_marker.exists():
        failed_marker.unlink()
    (logs / "COMPLETE").write_text("600/600 new candidate arm-episodes complete; 1200/1200 four-arm results analyzed\n")
    print(json.dumps({"complete": True, "new_episodes": 600, "four_arm_total": 1200}), flush=True)


if __name__ == "__main__":
    main()
