"""Persistent two-H100, four-worker coordinator for the 2400-episode sweep."""
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

from research.semantic_token_cd.prompt_attn_l11_count_rollout import ARM_COUNTS, TASKS


GPUS = (2, 3)
SLOTS = ((2, 0), (2, 1), (3, 0), (3, 1))
CHUNKS = ((101, 125), (126, 150), (151, 175), (176, 199))
MAX_ATTEMPTS = 8


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
        for seed in range(job.lo, job.hi + 1) for arm in ARM_COUNTS
    )


def launch(command: list[str], log: Path, cwd: Path, env: dict):
    handle = log.open("a")
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle


def rollout_command(repo: Path, root: Path, source: Path, task: str, seeds: str,
                    gpu: int, worker: str) -> list[str]:
    return [
        sys.executable,
        str(repo / "research/semantic_token_cd/prompt_attn_l11_count_rollout.py"),
        "--artifact", str(root), "--snapshot-artifact", str(source),
        "--task", task, "--seeds", seeds, "--gpu", str(gpu),
        "--worker-id", worker,
    ]


def run_preflight(repo: Path, root: Path, source: Path, logs: Path, env: dict) -> None:
    active = []
    for index, task in enumerate(TASKS):
        gpu, slot = SLOTS[index]
        log = logs / f"preflight_{task}_gpu{gpu}_slot{slot}.log"
        command = rollout_command(repo, root, source, task, "100", gpu, f"preflight_gpu{gpu}_slot{slot}")
        process, handle = launch(command, log, repo, env)
        active.append((process, handle, task, log))
        print(json.dumps({"preflight_started": task, "gpu": gpu, "pid": process.pid}), flush=True)
        time.sleep(12)
    failed = []
    for process, handle, task, log in active:
        status = process.wait()
        handle.close()
        if status != 0:
            failed.append({"task": task, "status": status, "log": str(log)})
    if failed:
        raise RuntimeError({"preflight_rollout_failed": failed})
    validator_log = logs / "preflight_validation.log"
    with validator_log.open("a") as handle:
        subprocess.run(
            [sys.executable, str(repo / "research/semantic_token_cd/validate_prompt_attn_l11_count_preflight.py"),
             "--artifact", str(root), "--seed", "100"],
            cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True,
        )
    print(json.dumps({"preflight_complete": True, "episode_count": 24}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    repo = Path(__file__).resolve().parents[2]
    logs = root / "rollout_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{repo}:{pcd}:{env.get('PYTHONPATH', '')}",
    })

    preflight_report = root / "preflight/PREFLIGHT_REPORT.json"
    if not preflight_report.exists() or not json.loads(preflight_report.read_text()).get("passed"):
        run_preflight(repo, root, source, logs, env)

    jobs = [Job(task, lo, hi) for lo, hi in CHUNKS for task in TASKS]
    queue = collections.deque(job for job in jobs if not complete(root, job))
    active = {}
    capacity = {2: 2, 3: 2}
    available_after = {slot: 0.0 for slot in SLOTS}
    failures = []
    started_at = time.monotonic()
    print(json.dumps({
        "protocol": "PROMPT_ATTN_L11_TOKEN_COUNT_V1", "total_episodes": 2400,
        "remaining_jobs": len(queue), "gpus": list(GPUS), "workers": 4,
    }), flush=True)

    while queue or active:
        for gpu, slot_index in SLOTS:
            slot = (gpu, slot_index)
            if slot_index >= capacity[gpu] or slot in active or not queue:
                continue
            if time.monotonic() < available_after[slot]:
                continue
            job = queue.popleft()
            if complete(root, job):
                continue
            job.attempts += 1
            log = logs / f"{job.name}_gpu{gpu}_slot{slot_index}_attempt{job.attempts}.log"
            command = rollout_command(
                repo, root, source, job.task, f"{job.lo}-{job.hi}", gpu,
                f"gpu{gpu}_slot{slot_index}_{job.name}",
            )
            process, handle = launch(command, log, repo, env)
            active[slot] = (process, handle, job, log)
            print(json.dumps({"started": job.name, "gpu": gpu, "slot": slot_index,
                              "pid": process.pid, "attempt": job.attempts}), flush=True)
            time.sleep(12)

        time.sleep(5)
        for slot, (process, handle, job, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[slot]
            available_after[slot] = time.monotonic() + 30
            if status == 0 and complete(root, job):
                print(json.dumps({"completed": job.name, "gpu": slot[0], "slot": slot[1]}), flush=True)
                continue
            tail = log.read_text(errors="replace")[-20000:].lower()
            oom = "out of memory" in tail
            if oom:
                capacity[slot[0]] = 1
                available_after[(slot[0], 0)] = time.monotonic() + 120
                available_after[(slot[0], 1)] = time.monotonic() + 120
            if job.attempts < MAX_ATTEMPTS:
                queue.append(job)
                print(json.dumps({"retry": job.name, "status": status, "oom": oom,
                                  "attempt": job.attempts, "gpu_capacity": capacity[slot[0]],
                                  "log": str(log)}), flush=True)
            else:
                failures.append({"job": job.name, "status": status, "log": str(log)})

    missing = [job.name for job in jobs if not complete(root, job)]
    if missing:
        payload = {"complete": False, "missing": missing, "failures": failures}
        (logs / "FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(payload)
    elapsed = time.monotonic() - started_at
    payload = {"complete": True, "episodes": 2400, "elapsed_seconds_after_preflight": elapsed}
    (logs / "COMPLETE.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
