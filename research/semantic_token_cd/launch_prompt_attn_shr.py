"""Resumable 18-worker coordinator for Prompt-Attn-SHR v1 on GPUs 0..5."""
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
    "google_robot_open_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ARMS = ("standard_shr", "prompt_attn_shr", "random_shr")
CHUNKS = ((0, 16), (17, 33), (34, 50), (51, 67), (68, 83), (84, 99))
GPUS = (0, 1, 2, 3, 4, 5)
WORKERS_PER_GPU = 3


@dataclass
class Job:
    task: str
    lo: int
    hi: int
    attempts: int = 0

    @property
    def name(self) -> str:
        return f"{self.task}_{self.lo:03d}_{self.hi:03d}"


def complete(artifact: Path, job: Job) -> bool:
    return all(
        (artifact / "episodes" / job.task / arm / f"episode_{seed:03d}_summary.json").exists()
        and (artifact / "episodes" / job.task / arm / f"episode_{seed:03d}_arrays.npz").exists()
        for seed in range(job.lo, job.hi + 1)
        for arm in ARMS
    )


def jobs() -> list[Job]:
    # Each GPU gets one shard from every task: exactly three resident models/GPU.
    return [Job(task, *CHUNKS[gpu]) for gpu in GPUS for task in TASKS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    repo = Path(__file__).resolve().parents[2]
    rollout = repo / "research/semantic_token_cd/prompt_attn_shr_rollout.py"
    analyzer = repo / "research/semantic_token_cd/analyze_prompt_attn_shr.py"
    logs = artifact / "rollout_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{repo}:{pcd}:{env.get('PYTHONPATH', '')}",
    })
    queue = collections.deque(job for job in jobs() if not complete(artifact, job))
    slots = [(gpu, slot) for gpu in GPUS for slot in range(WORKERS_PER_GPU)]
    active = {}
    failures = []
    print(json.dumps({
        "protocol": "PROMPT_ATTN_SHR_V1", "gpus": list(GPUS),
        "workers_per_gpu": WORKERS_PER_GPU, "total_workers": len(slots),
        "queued_jobs": len(queue),
    }), flush=True)
    while queue or active:
        for gpu, slot in slots:
            key = (gpu, slot)
            if key in active or not queue:
                continue
            job = queue.popleft()
            if complete(artifact, job):
                continue
            job.attempts += 1
            log = logs / f"{job.name}_gpu{gpu}_slot{slot}_attempt{job.attempts}.log"
            handle = log.open("a")
            command = [
                sys.executable, str(rollout),
                "--artifact", str(artifact),
                "--snapshot-artifact", str(source),
                "--task", job.task,
                "--seeds", f"{job.lo}-{job.hi}",
                "--gpu", str(gpu),
                "--worker-id", f"gpu{gpu}_slot{slot}_{job.lo:03d}_{job.hi:03d}",
            ]
            process = subprocess.Popen(
                command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[key] = (process, job, handle, log)
            print(json.dumps({"started": job.name, "gpu": gpu, "slot": slot, "pid": process.pid}), flush=True)
            time.sleep(2)  # Stagger model loads and first attention allocations.
        time.sleep(5)
        for key, (process, job, handle, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[key]
            if status == 0 and complete(artifact, job):
                print(json.dumps({"completed": job.name, "gpu": key[0]}), flush=True)
            elif job.attempts < 3:
                queue.append(job)
                print(json.dumps({
                    "retry": job.name, "status": status, "attempt": job.attempts,
                    "log": str(log),
                }), flush=True)
            else:
                failures.append({"job": job.name, "status": status, "log": str(log)})
                print(json.dumps({"quarantined": failures[-1]}), flush=True)

    missing = [job.name for job in jobs() if not complete(artifact, job)]
    if missing:
        payload = {"complete": False, "missing": missing, "failures": failures}
        (logs / "FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(payload)
    subprocess.run(
        [sys.executable, str(analyzer), "--artifact", str(artifact)],
        cwd=repo, env=env, check=True,
    )
    (logs / "COMPLETE").write_text("900/900 Prompt-Attn-SHR v1 arm-episodes complete\n")
    print(json.dumps({"complete": True, "episodes": 900}), flush=True)


if __name__ == "__main__":
    main()
