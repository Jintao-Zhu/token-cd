"""Persistent coordinator for XSWAP-V1 closed-loop rollout on GPUs 2,3.

Queue = (task, [manifest seeds]) chunks sized for fine-grained resume.  A job is
complete only when all five arms have episode summaries+arrays for every seed.
Retries with a per-job attempt cap.  Writes a JSON state file after each event.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from research.semantic_token_cd.xswap_protocol import ARMS, ARTIFACT

GPUS = (2, 3)
MAX_WORKERS_PER_GPU = 3
SLOTS = [(gpu, slot) for gpu in GPUS for slot in range(MAX_WORKERS_PER_GPU)]
MAX_ATTEMPTS = 4
CHUNK_TARGET = 11          # manifest scenes per job


@dataclass
class Job:
    task: str
    seeds: list[int]
    attempts: int = 0
    status: str = "queued"

    @property
    def name(self) -> str:
        return f"{self.task.removeprefix('google_robot_')}_{self.seeds[0]:03d}_{self.seeds[-1]:03d}"

    @property
    def seed_spec(self) -> str:
        return ",".join(str(s) for s in self.seeds)


def load_manifest() -> dict[str, list[int]]:
    path = ARTIFACT / "scene_manifest.json"
    return json.loads(path.read_text())["scenes"]


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for task, seeds in load_manifest().items():
        for i in range(0, len(seeds), CHUNK_TARGET):
            jobs.append(Job(task, seeds[i:i + CHUNK_TARGET]))
    return jobs


def complete(root: Path, job: Job) -> bool:
    for seed in job.seeds:
        for arm in ARMS:
            if not (root / "runs/episodes" / job.task / arm / f"episode_{seed:03d}_summary.json").exists():
                return False
            if not (root / "runs/episodes" / job.task / arm / f"episode_{seed:03d}_arrays.npz").exists():
                return False
    return True


def rollout_command(repo: Path, task: str, seed_spec: str, gpu: int, worker: str) -> list[str]:
    return [
        sys.executable, str(repo / "research/semantic_token_cd/xswap_rollout.py"),
        "--task", task, "--seeds", seed_spec, "--gpu", str(gpu),
        "--arms", ",".join(ARMS), "--emit", "--worker-id", worker,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-per-gpu", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    args = parser.parse_args()
    repo = Path("/home/leju-suzhou/zjt_ws/token-cd")
    root = ARTIFACT.resolve()
    logs = Path("/home/leju-suzhou/zjt_ws/token-cd/logs/xswap/full")
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{repo}:{pcd}:{env.get('PYTHONPATH', '')}",
    })

    slots = [(gpu, slot) for gpu in GPUS for slot in range(args.workers_per_gpu)]
    queue = collections.deque(job for job in build_jobs() if not complete(root, job))
    active: dict[tuple, tuple] = {}
    state_path = logs / "coordinator_state.json"
    started_at = time.monotonic()
    state = {"jobs_total": len(build_jobs()), "remaining": len(queue),
             "workers_per_gpu": args.workers_per_gpu, "events": []}
    print(json.dumps({"protocol": "PROMPT_ATTN_INSTR_SWAP_V1_FULL",
                      "jobs_total": state["jobs_total"], "jobs_remaining": state["remaining"],
                      "slots": len(slots)}), flush=True)

    def persist() -> None:
        state["remaining"] = len(queue) + len(active)
        state["updated_epoch"] = time.time()
        state_path.write_text(json.dumps(state, indent=1))

    while queue or active:
        for key in slots:
            if key in active or not queue:
                continue
            job = queue.popleft()
            job.attempts += 1
            job.status = "running"
            log = logs / f"{job.name}_gpu{key[0]}_slot{key[1]}_attempt{job.attempts}.log"
            handle = log.open("a")
            command = rollout_command(repo, job.task, job.seed_spec, key[0],
                                      f"gpu{key[0]}_slot{key[1]}_{job.name}")
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = (process, job, handle, log, time.monotonic())
            state["events"].append({"event": "start", "job": job.name, "task": job.task,
                                    "seeds": job.seeds, "gpu": key[0], "slot": key[1],
                                    "attempt": job.attempts, "epoch": time.time()})
            print(json.dumps({"started": job.name, "task": job.task, "n_seeds": len(job.seeds),
                              "gpu": key[0], "slot": key[1], "attempt": job.attempts}), flush=True)
            persist()
            time.sleep(2)
        time.sleep(10)
        for key, (process, job, handle, log, t0) in list(active.items()):
            if process.poll() is None:
                continue
            handle.close()
            del active[key]
            elapsed = time.monotonic() - t0
            if process.returncode == 0 and complete(root, job):
                job.status = "complete"
                print(json.dumps({"completed": job.name, "gpu": key[0], "elapsed_s": round(elapsed)}), flush=True)
                state["events"].append({"event": "complete", "job": job.name, "gpu": key[0],
                                        "elapsed_s": round(elapsed), "epoch": time.time()})
            elif job.attempts < args.max_attempts:
                queue.append(job)
                job.status = "queued"
                print(json.dumps({"retry": job.name, "attempt": job.attempts,
                                  "status": process.returncode, "log": str(log)}), flush=True)
                state["events"].append({"event": "retry", "job": job.name,
                                        "attempt": job.attempts, "log": str(log), "epoch": time.time()})
            else:
                job.status = "failed"
                print(json.dumps({"failed_permanent": job.name, "log": str(log)}), flush=True)
                state["events"].append({"event": "failed_permanent", "job": job.name,
                                        "log": str(log), "epoch": time.time()})
            persist()
    elapsed_total = time.monotonic() - started_at
    print(json.dumps({"ALL_JOBS_FINISHED": True, "elapsed_s": round(elapsed_total)}), flush=True)
    state["done"] = True
    persist()


if __name__ == "__main__":
    main()
