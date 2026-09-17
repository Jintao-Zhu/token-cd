"""Six-slot resumable coordinator for 1200 target-specific rollouts."""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from research.semantic_token_cd.target_specific_protocol import ARTIFACT as DIFFERENCE_ARTIFACT, NEW_ARMS as DIFFERENCE_ARMS, TASKS
from research.semantic_token_cd.target_positive_boost_protocol import ARTIFACT as BOOST_ARTIFACT, NEW_ARMS as BOOST_ARMS

CHUNK = 5


@dataclass
class Job:
    task: str; arm: str; seeds: list[int]; attempts: int = 0
    @property
    def name(self): return f"{self.task.removeprefix('google_robot_')}_{self.arm}_{self.seeds[0]:03d}_{self.seeds[-1]:03d}"


def jobs(arms):
    return [Job(task, arm, list(range(start, min(100, start + CHUNK))))
            for task in TASKS for arm in arms for start in range(0, 100, CHUNK)]


def complete(job, artifact):
    for seed in job.seeds:
        root = artifact / "closed_loop/episodes" / job.task / job.arm
        if not (root / f"episode_{seed:03d}_summary.json").exists(): return False
        if not (root / f"episode_{seed:03d}_arrays.npz").exists(): return False
        if not (artifact / "closed_loop/videos" / job.task / job.arm / f"episode_{seed:03d}.mp4").exists(): return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--variant", choices=("difference", "positive_boost"), default="difference"); args = parser.parse_args()
    artifact = BOOST_ARTIFACT if args.variant == "positive_boost" else DIFFERENCE_ARTIFACT
    arms = BOOST_ARMS if args.variant == "positive_boost" else DIFFERENCE_ARMS
    gpus = tuple(int(x) for x in args.gpus.split(",")); slots = [(gpu, slot) for gpu in gpus for slot in range(args.workers_per_gpu)]
    queue = collections.deque(job for job in jobs(arms) if not complete(job, artifact)); active = {}
    logs = artifact / "closed_loop/logs"; logs.mkdir(parents=True, exist_ok=True)
    state_path = artifact / "closed_loop/coordinator_state.json"
    environment = os.environ.copy(); environment.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:{environment.get('PYTHONPATH','')}"})
    events = []
    def persist():
        state_path.write_text(json.dumps({"remaining_jobs": len(queue) + len(active), "active": [x[1].name for x in active.values()],
                                          "events": events[-1000:], "updated": time.time()}, indent=2))
    while queue or active:
        for slot in slots:
            if slot in active or not queue: continue
            job = queue.popleft(); job.attempts += 1; log = logs / f"{job.name}_gpu{slot[0]}_slot{slot[1]}_a{job.attempts}.log"
            handle = log.open("a"); command = [sys.executable,
                "/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/target_specific_rollout.py",
                "--task", job.task, "--arm", job.arm, "--seeds", ",".join(map(str, job.seeds)),
                "--gpu", str(slot[0]), "--worker-id", f"gpu{slot[0]}_slot{slot[1]}"]
            process = subprocess.Popen(command, cwd="/home/leju-suzhou/zjt_ws/token-cd", env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[slot] = (process, job, handle, log, time.monotonic()); events.append({"event": "start", "job": job.name, "slot": slot})
            persist(); time.sleep(2)
        time.sleep(10)
        for slot, (process, job, handle, log, started) in list(active.items()):
            if process.poll() is None: continue
            handle.close(); del active[slot]
            if process.returncode == 0 and complete(job, artifact):
                events.append({"event": "complete", "job": job.name, "seconds": round(time.monotonic() - started)})
            elif job.attempts < 4:
                queue.append(job); events.append({"event": "retry", "job": job.name, "returncode": process.returncode, "log": str(log)})
            else:
                persist(); raise RuntimeError(f"permanent failure: {job.name}; {log}")
            persist()
    (artifact / "closed_loop/COMPLETE").write_text("1200 new episodes complete\n"); persist()


if __name__ == "__main__": main()
