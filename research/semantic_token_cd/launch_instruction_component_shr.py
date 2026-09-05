"""Capacity-aware persistent launcher for the IC-SHR v1 protocol."""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Job:
    task: str
    lo: int
    hi: int
    attempts: int = 0

    @property
    def name(self):
        return f"{self.task}_{self.lo:03d}_{self.hi:03d}"


def initial_jobs() -> list[Job]:
    tasks = (
        "google_robot_open_drawer", "google_robot_close_drawer",
        "google_robot_pick_coke_can", "google_robot_move_near",
    )
    return [Job(task, lo, lo + 74) for task in tasks for lo in range(0, 300, 75)]


def complete(artifact: Path, job: Job) -> bool:
    summary = artifact / "episode_summary" / job.task / "ic_shr"
    arrays = artifact / "rollout" / job.task / "ic_shr"
    masks = artifact / "masks" / job.task
    return all(
        (summary / f"episode_{seed:03d}_summary.json").exists()
        and (arrays / f"episode_{seed:03d}_arrays.npz").exists()
        and (masks / f"episode_{seed:03d}_components.json").exists()
        for seed in range(job.lo, job.hi + 1)
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, required=True)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--gpu1-workers", type=int, default=4, choices=range(0, 7))
    ap.add_argument("--gpu4-workers", type=int, default=4, choices=range(0, 7))
    ap.add_argument("--gpu5-workers", type=int, default=1, choices=range(0, 7))
    args = ap.parse_args()
    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    log_dir = artifact / "rollout_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = collections.deque(initial_jobs())
    capacities = {1: args.gpu1_workers, 4: args.gpu4_workers, 5: args.gpu5_workers}
    slots = [(gpu, slot) for gpu, count in capacities.items() for slot in range(count)]
    active: dict[tuple[int, int], tuple[subprocess.Popen, Job, object]] = {}
    repo = Path(__file__).resolve().parents[2]
    rollout = repo / "research/semantic_token_cd/instruction_component_shr_rollout.py"
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    pcd = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
    env["PYTHONPATH"] = f"{repo}:{pcd}:{env.get('PYTHONPATH', '')}"
    failed = []
    print(json.dumps({"capacities": capacities, "jobs": len(jobs)}), flush=True)
    while jobs or active:
        for gpu, slot in slots:
            key = (gpu, slot)
            if key in active or not jobs:
                continue
            job = jobs.popleft()
            if complete(artifact, job):
                print(json.dumps({"skip_complete_job": job.name}), flush=True)
                continue
            job.attempts += 1
            path = log_dir / f"{job.name}_gpu{gpu}_slot{slot}_attempt{job.attempts}.log"
            handle = path.open("a")
            command = [
                sys.executable, str(rollout), "--artifact", str(artifact),
                "--snapshot-artifact", str(source), "--task", job.task,
                "--seeds", f"{job.lo}-{job.hi}", "--gpu", str(gpu),
                "--worker-id", f"gpu{gpu}_slot{slot}_{job.lo:03d}_{job.hi:03d}_attempt{job.attempts}",
            ]
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = (process, job, handle)
            print(json.dumps({"started": job.name, "gpu": gpu, "slot": slot,
                              "pid": process.pid}), flush=True)
        time.sleep(5)
        for key, (process, job, handle) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[key]
            if status == 0 and complete(artifact, job):
                print(json.dumps({"completed": job.name, "gpu": key[0], "slot": key[1]}), flush=True)
            elif job.attempts < args.max_attempts:
                jobs.append(job)
                print(json.dumps({"retry": job.name, "gpu": key[0], "status": status,
                                  "attempt": job.attempts}), flush=True)
            else:
                failed.append({"job": job.name, "gpu": key[0], "status": status,
                               "attempts": job.attempts})
                print(json.dumps({"quarantined": failed[-1]}), flush=True)
    if failed:
        (log_dir / "FAILED_JOBS.json").write_text(json.dumps(failed, indent=2) + "\n")
        print(json.dumps({"complete": False, "failed": len(failed)}), flush=True)
        return
    analyzer = repo / "research/semantic_token_cd/analyze_instruction_component_shr.py"
    subprocess.run([sys.executable, str(analyzer), "--artifact", str(artifact),
                    "--snapshot-artifact", str(source)], cwd=repo, env=env, check=True)
    (log_dir / "COMPLETE").write_text("1200/1200 IC-SHR episodes complete and analyzed\n")
    print(json.dumps({"complete": True, "episodes": 1200}), flush=True)


if __name__ == "__main__":
    main()
