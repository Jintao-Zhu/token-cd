"""Expand IC-SHR onto healthy idle GPUs without overlapping live drawer jobs."""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


TASKS = (
    "google_robot_close_drawer", "google_robot_open_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
ARM = "ic_shr"


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
        (root / "episode_summary" / job.task / ARM / f"episode_{s:03d}_summary.json").exists()
        and (root / "rollout" / job.task / ARM / f"episode_{s:03d}_arrays.npz").exists()
        and (root / "masks" / job.task / f"episode_{s:03d}_components.json").exists()
        for s in range(job.lo, job.hi + 1)
    )


def initial_jobs() -> list[Job]:
    # Queue order plus slot order yields four initial workers on each of 0/2/3.
    return [
        Job("google_robot_pick_coke_can", 0, 49),
        Job("google_robot_pick_coke_can", 50, 99),
        Job("google_robot_move_near", 0, 49),
        Job("google_robot_move_near", 50, 99),
        Job("google_robot_pick_coke_can", 100, 149),
        Job("google_robot_pick_coke_can", 150, 199),
        Job("google_robot_move_near", 100, 149),
        Job("google_robot_move_near", 150, 199),
        Job("google_robot_pick_coke_can", 200, 249),
        Job("google_robot_pick_coke_can", 250, 299),
        Job("google_robot_move_near", 200, 249),
        Job("google_robot_move_near", 250, 299),
    ]


def live_drawer_workers(artifact: Path) -> int:
    marker = str(artifact).encode()
    count = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (b"instruction_component_shr_rollout.py" in cmd and marker in cmd
                and (b"google_robot_open_drawer" in cmd or b"google_robot_close_drawer" in cmd)):
            count += 1
    return count


def missing_jobs(root: Path, chunk: int = 50) -> list[Job]:
    jobs = []
    for task in TASKS:
        missing = []
        for seed in range(300):
            probe = Job(task, seed, seed)
            if not complete(root, probe):
                missing.append(seed)
        start = 0
        while start < len(missing):
            end = start
            while (end + 1 < len(missing) and missing[end + 1] == missing[end] + 1
                   and end - start + 1 < chunk):
                end += 1
            jobs.append(Job(task, missing[start], missing[end]))
            start = end + 1
    return jobs


def run_queue(jobs: list[Job], slots, artifact: Path, source: Path,
              rollout: Path, repo: Path, env: dict, log_dir: Path) -> list[dict]:
    queue = collections.deque(jobs)
    active = {}
    failed = []
    while queue or active:
        for gpu, slot in slots:
            key = (gpu, slot)
            if key in active or not queue:
                continue
            job = queue.popleft()
            if complete(artifact, job):
                print(json.dumps({"skip_complete": job.name}), flush=True)
                continue
            job.attempts += 1
            path = log_dir / f"expanded_{job.name}_gpu{gpu}_slot{slot}_attempt{job.attempts}.log"
            handle = path.open("a")
            cmd = [sys.executable, str(rollout), "--artifact", str(artifact),
                   "--snapshot-artifact", str(source), "--task", job.task,
                   "--seeds", f"{job.lo}-{job.hi}", "--gpu", str(gpu),
                   "--worker-id", f"expanded_gpu{gpu}_slot{slot}_{job.lo:03d}_{job.hi:03d}"]
            process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=handle,
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
                print(json.dumps({"completed": job.name, "gpu": key[0]}), flush=True)
            elif job.attempts < 3:
                queue.append(job)
                print(json.dumps({"retry": job.name, "status": status}), flush=True)
            else:
                failed.append({"job": job.name, "status": status, "gpu": key[0]})
                print(json.dumps({"quarantined": failed[-1]}), flush=True)
    return failed


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, required=True)
    args = ap.parse_args()
    artifact, source = args.artifact.resolve(), args.snapshot_artifact.resolve()
    repo = Path(__file__).resolve().parents[2]
    rollout = repo / "research/semantic_token_cd/instruction_component_shr_rollout.py"
    analyzer = repo / "research/semantic_token_cd/analyze_instruction_component_shr.py"
    log_dir = artifact / "rollout_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    slots = [(gpu, slot) for gpu in (0, 2, 3) for slot in range(4)]
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env["PYTHONPATH"] = f"{repo}:{pcd}:{env.get('PYTHONPATH', '')}"
    print(json.dumps({"phase": "expanded", "gpus": [0, 2, 3],
                      "workers_per_gpu": 4, "jobs": 12}), flush=True)
    failures = run_queue(initial_jobs(), slots, artifact, source, rollout, repo, env, log_dir)
    while live_drawer_workers(artifact):
        print(json.dumps({"waiting_for_drawer_workers": live_drawer_workers(artifact)}), flush=True)
        time.sleep(30)
    backfill = missing_jobs(artifact)
    if backfill:
        print(json.dumps({"phase": "backfill", "jobs": len(backfill)}), flush=True)
        failures.extend(run_queue(backfill, slots, artifact, source, rollout, repo, env, log_dir))
    remaining = missing_jobs(artifact, chunk=1)
    if remaining:
        payload = {"complete": False, "missing": len(remaining), "failures": failures}
        (log_dir / "EXPANDED_FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(payload)
    subprocess.run([sys.executable, str(analyzer), "--artifact", str(artifact),
                    "--snapshot-artifact", str(source)], cwd=repo, env=env, check=True)
    (log_dir / "COMPLETE").write_text("1200/1200 IC-SHR episodes complete and analyzed\n")
    print(json.dumps({"complete": True, "episodes": 1200}), flush=True)


if __name__ == "__main__":
    main()
