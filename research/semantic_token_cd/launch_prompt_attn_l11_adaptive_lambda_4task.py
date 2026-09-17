"""Two-stage four-worker launcher for L11 Adaptive-lambda on GPUs 2 and 3."""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from research.semantic_token_cd.prompt_attn_l11_adaptive_lambda_rollout import TASKS


REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "artifacts/prompt_attn_l11_adaptive_lambda_4task_0_299_v1"
CANONICAL = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
ROLLOUT = REPO / "research/semantic_token_cd/prompt_attn_l11_adaptive_lambda_rollout.py"
ANALYZE = REPO / "research/semantic_token_cd/analyze_prompt_attn_l11_adaptive_lambda_4task.py"
SLOTS = ((2, 0), (2, 1), (3, 0), (3, 1))
DEV_ARMS = ("l11_fixed_025", "l11_adaptive_raw", "l11_adaptive_ema")
MEAN_RUNTIME_SECONDS = {
    "google_robot_close_drawer": 152.0,
    "google_robot_pick_coke_can": 57.0,
    "google_robot_move_near": 60.0,
    "widowx_carrot_on_plate": 54.0,
}
MAX_ATTEMPTS = 20
JOB_SIZE = 10


@dataclass(frozen=True)
class Job:
    stage: str
    task: str
    lo: int
    hi: int
    arms: tuple[str, ...]
    attempts: int = 0

    @property
    def name(self) -> str:
        return f"{self.stage}_{self.task}_{self.lo:03d}_{self.hi:03d}"

    @property
    def estimated_seconds(self) -> float:
        return MEAN_RUNTIME_SECONDS[self.task] * len(self.arms) * (self.hi - self.lo + 1)


def complete(job: Job) -> bool:
    return all(
        (ARTIFACT / "episodes" / job.task / arm / f"episode_{seed:03d}_summary.json").exists()
        and (ARTIFACT / "episodes" / job.task / arm / f"episode_{seed:03d}_arrays.npz").exists()
        for arm in job.arms
        for seed in range(job.lo, job.hi + 1)
    )


def jobs_for(stage: str, lo: int, hi: int, arms: tuple[str, ...]) -> list[Job]:
    jobs = [
        Job(stage, task, start, min(start + JOB_SIZE - 1, hi), arms)
        for task in TASKS
        for start in range(lo, hi + 1, JOB_SIZE)
    ]
    return sorted(jobs, key=lambda job: (-job.estimated_seconds, job.name))


def run_jobs(jobs: list[Job], log_dir: Path) -> None:
    queue = collections.deque(job for job in jobs if not complete(job))
    active: dict[int, tuple[subprocess.Popen, Job, object, tuple[int, int]]] = {}
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    pcd_source = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env["PYTHONPATH"] = f"{REPO}:{pcd_source}:{env.get('PYTHONPATH', '')}"

    while queue or active:
        for slot_index, slot in enumerate(SLOTS):
            if slot_index in active or not queue:
                continue
            job = queue.popleft()
            job = Job(job.stage, job.task, job.lo, job.hi, job.arms, job.attempts + 1)
            gpu, local_worker = slot
            log_path = log_dir / f"{job.name}_gpu{gpu}_worker{local_worker}_attempt{job.attempts}.log"
            handle = log_path.open("a")
            command = [
                sys.executable,
                str(ROLLOUT),
                "--artifact",
                str(ARTIFACT),
                "--snapshot-artifact",
                str(CANONICAL),
                "--task",
                job.task,
                "--seeds",
                f"{job.lo}-{job.hi}",
                "--gpu",
                str(gpu),
                "--worker-id",
                f"gpu{gpu}_worker{local_worker}_{job.stage}_attempt{job.attempts}",
                "--arms",
                ",".join(job.arms),
            ]
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
            active[slot_index] = (process, job, handle, slot)
            print(json.dumps({
                "event": "start",
                "job": job.name,
                "slot": slot,
                "pid": process.pid,
                "arms": job.arms,
                "attempt": job.attempts,
            }), flush=True)

        time.sleep(5.0)
        for slot_index, (process, job, handle, slot) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[slot_index]
            if status == 0 and complete(job):
                print(json.dumps({"event": "complete", "job": job.name, "slot": slot}), flush=True)
                continue
            if job.attempts >= MAX_ATTEMPTS:
                raise RuntimeError(f"job failed after {job.attempts} attempts: {job.name}")
            queue.append(job)
            print(json.dumps({
                "event": "retry",
                "job": job.name,
                "slot": slot,
                "exit_code": status,
                "attempt": job.attempts,
            }), flush=True)


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    log_dir = ARTIFACT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "protocol": "PROMPT_ATTN_L11_ADAPTIVE_LAMBDA_4TASK_V1",
        "tasks": TASKS,
        "slots": SLOTS,
        "development": {"seeds": [0, 99], "arms": DEV_ARMS, "episodes": 1200},
        "confirmation": {"seeds": [100, 299], "arms": 2, "episodes": 1600},
    }), flush=True)

    run_jobs(jobs_for("development", 0, 99, DEV_ARMS), log_dir)
    selection_process = subprocess.run(
        [sys.executable, str(ANALYZE), "--artifact", str(ARTIFACT), "--select-development"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print(selection_process.stdout, flush=True)
    selection = json.loads((ARTIFACT / "DEVELOPMENT_SELECTION.json").read_text())
    selected_arm = selection["selected_arm"]
    confirmation_arms = ("l11_fixed_025", selected_arm)
    run_jobs(jobs_for("confirmation", 100, 299, confirmation_arms), log_dir)
    subprocess.run(
        [sys.executable, str(ANALYZE), "--artifact", str(ARTIFACT), "--selected-arm", selected_arm],
        cwd=REPO,
        check=True,
    )
    (ARTIFACT / "COMPLETE").write_text(
        f"four-task Adaptive-lambda complete; selected={selected_arm}; 2800 new arm episodes\n"
    )
    print(json.dumps({"complete": True, "selected_arm": selected_arm, "new_arm_episodes": 2800}), flush=True)


if __name__ == "__main__":
    main()
