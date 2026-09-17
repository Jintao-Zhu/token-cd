"""Persistent four-worker coordinator for the 400-episode L11 TopP90 extension."""
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

from research.semantic_token_cd.prompt_attn_l11_count_rollout import TASKS
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


ARM = "l11_top_p90"
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
        (root / "episodes" / job.task / ARM / f"episode_{seed:03d}_summary.json").exists()
        and (root / "episodes" / job.task / ARM / f"episode_{seed:03d}_arrays.npz").exists()
        for seed in range(job.lo, job.hi + 1)
    )


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def adaptive_workers_alive() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "prompt_attn_l11_adaptive_lambda_rollout.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def wait_for_previous_run(pid_file: Path | None, root: Path) -> None:
    if pid_file is None:
        return
    last_notice = 0.0
    while True:
        pid = int(pid_file.read_text().strip()) if pid_file.exists() else -1
        waiting = (pid > 0 and process_alive(pid)) or adaptive_workers_alive()
        if not waiting:
            break
        now = time.monotonic()
        if now - last_notice >= 300 or last_notice == 0.0:
            payload = {
                "queued": True,
                "coordinator_pid": os.getpid(),
                "waiting_for_pid": pid,
                "reason": "GPU2/3 reserved by adaptive-lambda",
            }
            atomic_json(root / "QUEUED.json", payload)
            print(json.dumps(payload), flush=True)
            last_notice = now
        time.sleep(30)
    time.sleep(60)
    (root / "QUEUED.json").unlink(missing_ok=True)


def launch(command: list[str], log: Path, cwd: Path, env: dict):
    handle = log.open("a")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle


def rollout_command(repo: Path, root: Path, canonical: Path, matched: Path, task: str,
                    seeds: str, gpu: int, worker: str) -> list[str]:
    return [
        sys.executable,
        str(repo / "research/semantic_token_cd/prompt_attn_l11_top_p90_rollout.py"),
        "--artifact", str(root),
        "--canonical", str(canonical),
        "--matched-artifact", str(matched),
        "--task", task,
        "--seeds", seeds,
        "--gpu", str(gpu),
        "--worker-id", worker,
    ]


def run_preflight(repo: Path, root: Path, canonical: Path, matched: Path,
                  base_top_p: Path, logs: Path, env: dict) -> None:
    active = []
    for index, task in enumerate(TASKS):
        gpu, slot = SLOTS[index]
        log = logs / f"preflight_{task}_gpu{gpu}_slot{slot}.log"
        command = rollout_command(repo, root, canonical, matched, task, "100", gpu,
                                  f"preflight_gpu{gpu}_slot{slot}")
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
    with (logs / "preflight_validation.log").open("a") as handle:
        subprocess.run([
            sys.executable,
            str(repo / "research/semantic_token_cd/validate_prompt_attn_l11_top_p90_preflight.py"),
            "--artifact", str(root),
            "--matched-artifact", str(matched),
            "--base-top-p-artifact", str(base_top_p),
            "--seed", "100",
        ], cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"preflight_complete": True, "new_episode_count": 4}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--base-top-p-artifact", type=Path, required=True)
    parser.add_argument("--wait-for-pid-file", type=Path)
    args = parser.parse_args()
    root = args.artifact.resolve()
    canonical = args.canonical.resolve()
    matched = args.matched_artifact.resolve()
    base_top_p = args.base_top_p_artifact.resolve()
    repo = Path(__file__).resolve().parents[2]
    logs = root / "rollout_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{repo}:{pcd}:{env.get('PYTHONPATH', '')}",
    })

    atomic_json(root / "QUEUED.json", {
        "queued": True,
        "coordinator_pid": os.getpid(),
        "gpus": [2, 3],
        "workers_per_gpu": 2,
        "wait_for_pid_file": str(args.wait_for_pid_file.resolve()) if args.wait_for_pid_file else None,
    })
    wait_for_previous_run(args.wait_for_pid_file.resolve() if args.wait_for_pid_file else None, root)
    atomic_json(root / "RUNNING.json", {
        "running": True,
        "coordinator_pid": os.getpid(),
        "protocol": "PROMPT_ATTN_L11_VISUAL_TOP_P90_EXTENSION_V1",
        "new_episodes": 400,
        "gpus": [2, 3],
        "workers_per_gpu": 2,
    })

    preflight_report = root / "preflight/PREFLIGHT_REPORT.json"
    if not preflight_report.exists() or not json.loads(preflight_report.read_text()).get("passed"):
        run_preflight(repo, root, canonical, matched, base_top_p, logs, env)

    jobs = [Job(task, lo, hi) for lo, hi in CHUNKS for task in TASKS]
    queue = collections.deque(job for job in jobs if not complete(root, job))
    active = {}
    capacity = {2: 2, 3: 2}
    available_after = {slot: 0.0 for slot in SLOTS}
    failures = []
    started_at = time.monotonic()
    print(json.dumps({
        "protocol": "PROMPT_ATTN_L11_VISUAL_TOP_P90_EXTENSION_V1",
        "new_episodes": 400,
        "reused_matched_episodes": 400,
        "reused_top_p75_80_85_episodes": 1200,
        "remaining_jobs": len(queue),
        "gpus": [2, 3],
        "workers": 4,
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
            command = rollout_command(repo, root, canonical, matched, job.task,
                                      f"{job.lo}-{job.hi}", gpu,
                                      f"gpu{gpu}_slot{slot_index}_{job.name}")
            process, handle = launch(command, log, repo, env)
            active[slot] = (process, handle, job, log)
            print(json.dumps({
                "started": job.name,
                "gpu": gpu,
                "slot": slot_index,
                "pid": process.pid,
                "attempt": job.attempts,
            }), flush=True)
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
            device_lost = "devicelost" in tail or "device lost" in tail
            if oom:
                capacity[slot[0]] = 1
                for gpu_slot in SLOTS:
                    if gpu_slot[0] == slot[0]:
                        available_after[gpu_slot] = time.monotonic() + 120
            if job.attempts < MAX_ATTEMPTS:
                queue.append(job)
                print(json.dumps({
                    "retry": job.name,
                    "status": status,
                    "oom": oom,
                    "device_lost": device_lost,
                    "attempt": job.attempts,
                    "gpu_capacity": capacity[slot[0]],
                    "log": str(log),
                }), flush=True)
            else:
                failures.append({"job": job.name, "status": status, "log": str(log)})

    missing = [job.name for job in jobs if not complete(root, job)]
    if missing:
        payload = {"complete": False, "missing": missing, "failures": failures}
        atomic_json(root / "FAILED.json", payload)
        raise RuntimeError(payload)
    payload = {
        "complete": True,
        "new_episodes": 400,
        "reused_matched_episodes": 400,
        "reused_top_p75_80_85_episodes": 1200,
        "elapsed_seconds_after_preflight": time.monotonic() - started_at,
    }
    with (logs / "final_analysis.log").open("a") as handle:
        subprocess.run([
            sys.executable,
            str(repo / "research/semantic_token_cd/analyze_prompt_attn_l11_top_p90.py"),
            "--artifact", str(root),
            "--matched-artifact", str(matched),
            "--base-top-p-artifact", str(base_top_p),
        ], cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    atomic_json(root / "COMPLETE.json", payload)
    (root / "RUNNING.json").unlink(missing_ok=True)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
