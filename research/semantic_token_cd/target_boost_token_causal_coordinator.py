"""Six-slot resumable coordinator for the 1200-state token causal audit."""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from research.semantic_token_cd.target_boost_token_causal_worker import ARTIFACT
from research.semantic_token_cd.target_specific_protocol import TASKS


@dataclass
class Job:
    task: str
    seeds: str
    attempts: int = 0

    @property
    def name(self) -> str:
        return f"{self.task.removeprefix('google_robot_')}_{self.seeds.replace('-', '_')}"


def complete(job: Job) -> bool:
    lo, hi = map(int, job.seeds.split("-"))
    return all((ARTIFACT / "results" / job.task / f"seed_{seed:03d}.json").exists() and
               (ARTIFACT / "results" / job.task / f"seed_{seed:03d}.npz").exists()
               for seed in range(lo, hi + 1))


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_id": "TARGET_BOOST_TOKEN_CAUSAL_AUDIT_V1",
        "purpose": "causally isolate matched-budget token swaps introduced by Positive-Boost-0.5",
        "tasks": list(TASKS), "seeds": "0-99", "episodes": 400,
        "state_fractions": [0.12, 0.50, 0.88], "same_states": 1200,
        "primary_comparison": "recomputed Correct versus Positive-Boost-0.5 on the same observation",
        "atomic_interventions": ["Correct plus exactly one budget-preserving swap",
                                 "full Boost with exactly one swap undone"],
        "gpus": [2, 3], "workers_per_gpu": 3,
        "no_new_closed_loop_rollouts": True,
    }
    config_path = ARTIFACT / "CONFIG_LOCK.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("causal audit config lock mismatch")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    jobs = collections.deque(Job(task, f"{lo}-{lo + 9}") for task in TASKS for lo in range(0, 100, 10))
    jobs = collections.deque(job for job in jobs if not complete(job))
    slots = [(gpu, slot) for gpu in (2, 3) for slot in range(3)]
    active = {}; events = []
    logs = ARTIFACT / "logs"; logs.mkdir(parents=True, exist_ok=True)
    state_path = ARTIFACT / "coordinator_state.json"
    env = os.environ.copy(); env.update({
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": "/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:" + env.get("PYTHONPATH", ""),
    })

    def persist() -> None:
        state_path.write_text(json.dumps({"remaining_jobs": len(jobs) + len(active),
            "completed_seed_files": len(list((ARTIFACT / "results").rglob("seed_*.json"))) if (ARTIFACT / "results").exists() else 0,
            "active": [entry[1].name for entry in active.values()], "events": events[-500:],
            "updated": time.time()}, indent=2) + "\n")

    while jobs or active:
        for slot in slots:
            if slot in active or not jobs:
                continue
            job = jobs.popleft(); job.attempts += 1
            log = logs / f"{job.name}_gpu{slot[0]}_slot{slot[1]}_a{job.attempts}.log"
            handle = log.open("a")
            command = [sys.executable,
                "/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/target_boost_token_causal_worker.py",
                "--task", job.task, "--seeds", job.seeds, "--gpu", str(slot[0]),
                "--worker-id", f"gpu{slot[0]}_slot{slot[1]}"]
            process = subprocess.Popen(command, cwd="/home/leju-suzhou/zjt_ws/token-cd", env=env,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[slot] = (process, job, handle, log, time.monotonic())
            events.append({"event": "start", "job": job.name, "slot": slot}); persist(); time.sleep(2)
        time.sleep(10)
        for slot, (process, job, handle, log, started) in list(active.items()):
            if process.poll() is None:
                continue
            handle.close(); del active[slot]
            if process.returncode == 0 and complete(job):
                events.append({"event": "complete", "job": job.name,
                               "seconds": round(time.monotonic() - started)})
            elif job.attempts < 3:
                jobs.append(job); events.append({"event": "retry", "job": job.name,
                                                 "returncode": process.returncode, "log": str(log)})
            else:
                persist(); raise RuntimeError(f"permanent failure: {job.name}: {log}")
            persist()
    (ARTIFACT / "COMPLETE").write_text("400 episodes / 1200 same-state audits complete\n")
    persist()


if __name__ == "__main__":
    main()
