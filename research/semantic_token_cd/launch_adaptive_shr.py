"""Bounded launcher over explicitly selected physical GPUs."""
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

from research.semantic_token_cd.semantic_recon_rollout import TASKS


ALLOWED_GPUS = (1, 2, 3, 4, 5)


def gpu_has_pi0_worker(gpu: int) -> bool:
    """Keep OpenVLA/SAPIEN off a GPU until its older Pi0 renderer exits."""
    result = subprocess.run(
        ["pgrep", "-af", "pi0_shr_rollout.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return any(
        "pi0_shr_rollout.py" in line and f"--gpu {gpu}" in line
        for line in result.stdout.splitlines()
    )


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
    root = artifact / "episode_json" / job.task / "adaptive_shr"
    return all(
        (root / f"episode_{seed:03d}_summary.json").exists()
        and (root / f"episode_{seed:03d}_arrays.npz").exists()
        for seed in range(job.lo, job.hi + 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--wait-for", type=Path, help="completion marker for the currently running experiment")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--job-size", type=int, default=75)
    parser.add_argument(
        "--quarantine-after",
        type=int,
        default=0,
        help="after this many failed attempts, record a job and continue; 0 disables quarantine",
    )
    parser.add_argument("--gpus", default="1,4", help="comma-separated physical GPU ids")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument(
        "--allow-pi0-colocation",
        action="store_true",
        help="allow Adaptive-SHR workers to share a selected GPU with Pi0 workers",
    )
    args = parser.parse_args()
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpus or any(gpu not in ALLOWED_GPUS for gpu in gpus):
        raise ValueError(f"--gpus must be a non-empty subset of {ALLOWED_GPUS}")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    if args.job_size < 1 or args.job_size > 300:
        raise ValueError("--job-size must be in [1, 300]")
    if args.quarantine_after < 0:
        raise ValueError("--quarantine-after must be non-negative")
    gpu_slots = tuple(gpu for gpu in gpus for _ in range(args.workers_per_gpu))
    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    log_dir = artifact / "rollout_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.wait_for:
        marker = args.wait_for.resolve()
        while not marker.exists():
            print(json.dumps({"waiting_for_current_experiment": str(marker)}), flush=True)
            time.sleep(args.poll_seconds)
        print(json.dumps({"current_experiment_complete": str(marker)}), flush=True)

    jobs = collections.deque(
        Job(task, lo, min(lo + args.job_size - 1, 299))
        for task in TASKS
        for lo in range(0, 300, args.job_size)
    )
    quarantined: list[dict] = []
    active: dict[int, tuple[subprocess.Popen, Job, object]] = {}
    repo = Path(__file__).resolve().parents[2]
    rollout = repo / "research/semantic_token_cd/adaptive_shr_rollout.py"
    python = Path(sys.executable)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    pcd_source = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
    env["PYTHONPATH"] = f"{repo}:{pcd_source}:{env.get('PYTHONPATH', '')}"

    while jobs or active:
        for slot, gpu in enumerate(gpu_slots):
            if slot in active:
                continue
            if not args.allow_pi0_colocation and gpu_has_pi0_worker(gpu):
                continue
            while jobs and complete(artifact, jobs[0]):
                done = jobs.popleft()
                print(json.dumps({"skip_complete_job": done.name}), flush=True)
            if not jobs:
                break
            job = jobs.popleft()
            job.attempts += 1
            log_path = log_dir / f"{job.name}_gpu{gpu}_attempt{job.attempts}.log"
            handle = log_path.open("a")
            command = [
                str(python), str(rollout),
                "--artifact", str(artifact),
                "--snapshot-artifact", str(source),
                "--task", job.task,
                "--seeds", f"{job.lo}-{job.hi}",
                "--gpu", str(gpu),
                "--worker-id", f"slot{slot}_attempt{job.attempts}",
            ]
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[slot] = (process, job, handle)
            print(json.dumps({"started": job.name, "gpu": gpu, "slot": slot, "pid": process.pid}), flush=True)

        time.sleep(2.0)
        for slot, (process, job, handle) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[slot]
            if status != 0 or not complete(artifact, job):
                if args.quarantine_after and job.attempts >= args.quarantine_after:
                    failure = {
                        "job": job.name,
                        "task": job.task,
                        "lo": job.lo,
                        "hi": job.hi,
                        "attempts": job.attempts,
                        "last_exit_code": status,
                    }
                    quarantined.append(failure)
                    print(json.dumps({"quarantined": failure}), flush=True)
                    continue
                if job.attempts >= args.max_attempts:
                    raise RuntimeError(f"job failed after {job.attempts} attempts: {job.name}")
                jobs.append(job)
                print(json.dumps({"retry": job.name, "exit_code": status, "attempt": job.attempts}), flush=True)
            else:
                print(json.dumps({"completed": job.name, "gpu": gpu_slots[slot]}), flush=True)

    if quarantined:
        failure_path = log_dir / "quarantined_jobs.json"
        failure_path.write_text(json.dumps(quarantined, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"complete": False, "quarantined_jobs": len(quarantined),
                          "failure_file": str(failure_path)}), flush=True)
        return

    analyze = repo / "research/semantic_token_cd/analyze_adaptive_shr.py"
    subprocess.run([
        str(python), str(analyze), "--artifact", str(artifact),
        "--snapshot-artifact", str(source),
    ], cwd=repo, env=env, check=True)
    (log_dir / "COMPLETE").write_text("2700/2700 Adaptive-SHR episodes complete and analyzed\n")
    print(json.dumps({"complete": True, "episodes": 2700}), flush=True)


if __name__ == "__main__":
    main()
