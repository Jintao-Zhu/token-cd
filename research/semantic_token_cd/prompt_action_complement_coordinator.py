"""Persistent six-slot GPU2/3 coordinator for the three new complement arms."""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from research.semantic_token_cd.prompt_action_complement_protocol import (
    ARTIFACT, NEW_ARMS, REPO, SEEDS, TASKS,
)

GPUS = (2, 3)
CHUNK = 5


@dataclass
class Job:
    task: str
    arm: str
    seeds: list[int]
    attempts: int = 0

    @property
    def name(self):
        return f"{self.task.removeprefix('google_robot_')}_{self.arm}_{self.seeds[0]:03d}_{self.seeds[-1]:03d}"


def jobs():
    return [Job(task, arm, list(SEEDS[start:start + CHUNK]))
            for task in TASKS for arm in NEW_ARMS for start in range(0, len(SEEDS), CHUNK)]


def complete(job: Job) -> bool:
    for seed in job.seeds:
        root = ARTIFACT / "closed_loop/episodes" / job.task / job.arm
        if not (root / f"episode_{seed:03d}_summary.json").exists(): return False
        if not (root / f"episode_{seed:03d}_arrays.npz").exists(): return False
        if not (ARTIFACT / "closed_loop/videos" / job.task / job.arm / f"episode_{seed:03d}.mp4").exists(): return False
    return True


def main():
    logs = ARTIFACT / "closed_loop/logs"; logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{REPO}:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:{env.get('PYTHONPATH', '')}",
    })
    queue = collections.deque(job for job in jobs() if not complete(job))
    slots = [(gpu, slot) for gpu in GPUS for slot in range(3)]
    active = {}
    state = {"jobs_total": len(jobs()), "remaining": len(queue), "events": [],
             "gpus": list(GPUS), "workers_per_gpu": 3, "slot_count": len(slots)}
    state_path = ARTIFACT / "closed_loop/coordinator_state.json"

    def persist():
        state["remaining"] = len(queue) + len(active)
        state["updated_epoch"] = time.time()
        atomic = state_path.with_suffix(".json.tmp")
        atomic.write_text(json.dumps(state, indent=1)); atomic.replace(state_path)

    while queue or active:
        for slot in slots:
            if slot in active or not queue: continue
            job = queue.popleft(); job.attempts += 1
            log = logs / f"{job.name}_gpu{slot[0]}_slot{slot[1]}_attempt{job.attempts}.log"
            handle = log.open("a")
            command = [sys.executable, str(REPO / "research/semantic_token_cd/prompt_action_complement_rollout.py"),
                       "--task", job.task, "--seeds", ",".join(map(str, job.seeds)),
                       "--gpu", str(slot[0]), "--arm", job.arm,
                       "--worker-id", f"gpu{slot[0]}_slot{slot[1]}"]
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            active[slot] = (process, job, handle, log, time.monotonic())
            state["events"].append({"event": "start", "job": job.name, "gpu": slot[0],
                                    "slot": slot[1], "epoch": time.time()})
            persist(); print(json.dumps({"started": job.name, "gpu": slot[0], "slot": slot[1]}), flush=True)
            time.sleep(2)
        time.sleep(10)
        for slot, (process, job, handle, log, started) in list(active.items()):
            if process.poll() is None: continue
            handle.close(); del active[slot]
            if process.returncode == 0 and complete(job):
                state["events"].append({"event": "complete", "job": job.name,
                                        "elapsed": round(time.monotonic() - started), "epoch": time.time()})
            elif job.attempts < 4:
                queue.append(job)
                state["events"].append({"event": "retry", "job": job.name,
                                        "rc": process.returncode, "log": str(log), "epoch": time.time()})
            else:
                state["events"].append({"event": "failed", "job": job.name,
                                        "rc": process.returncode, "log": str(log), "epoch": time.time()})
                persist(); raise RuntimeError(f"permanent rollout failure: {job.name}; {log}")
            persist()
    state["done"] = True; persist()
    (ARTIFACT / "closed_loop/COMPLETE").write_text("1200/1200 new arm episodes complete\n")
    print("CLOSED_LOOP_COMPLETE", flush=True)


if __name__ == "__main__": main()
