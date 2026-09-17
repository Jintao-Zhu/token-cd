"""Resumable 4-worker extension of L11-Matched to canonical seeds 100-299.

Runs only the frozen ``prompt_single`` arm (L11 Prompt Attention, Standard-SHR
KMeans coverage).  Seeds 0-99 already exist in the original layer-selection
artifacts and are combined later by the analyzer.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_attn_l11_matched_full_9task_0_299_v1"
RUN_ROOT = ROOT / "run"
CANONICAL = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
LAYER_REFERENCE = REPO / "artifacts/prompt_attn_layer_selection_v1"
ROLLOUT = REPO / "research/semantic_token_cd/run_prompt_attn_l11_full_0_299.py"
ARM = "prompt_single"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
MEAN_RUNTIME_SECONDS = {
    "google_robot_open_drawer": 94.0,
    "google_robot_close_drawer": 152.0,
    "google_robot_pick_coke_can": 57.0,
    "google_robot_move_near": 60.0,
    "google_robot_place_apple_in_closed_top_drawer": 332.0,
    "widowx_carrot_on_plate": 54.0,
    "widowx_put_eggplant_in_basket": 99.0,
    "widowx_spoon_on_towel": 52.0,
    "widowx_stack_cube": 46.0,
}
# Four workers on the currently reserved GPU2/GPU3 pair: two per GPU.
SLOTS = ((2, 0), (2, 1), (3, 0), (3, 1))
INITIAL_CAPACITY = {2: 2, 3: 2}
MAX_ATTEMPTS = 20
LAUNCH_STAGGER_SECONDS = 15
SEED_START = 100
SEED_END = 299


@dataclass(frozen=True)
class Job:
    task: str
    lo: int
    hi: int
    attempts: int = 0

    @property
    def name(self) -> str:
        return f"{self.task}_{self.lo:03d}_{self.hi:03d}"

    @property
    def estimated_seconds(self) -> float:
        return MEAN_RUNTIME_SECONDS[self.task] * (self.hi - self.lo + 1)


def complete(job: Job) -> bool:
    out = RUN_ROOT / "episodes" / job.task / ARM
    return all(
        (out / f"episode_{seed:03d}_summary.json").exists()
        and (out / f"episode_{seed:03d}_arrays.npz").exists()
        for seed in range(job.lo, job.hi + 1)
    )


def chunk_size(task: str) -> int:
    # Apple placement is much slower; shorter chunks reduce retry tail loss.
    return 5 if task == "google_robot_place_apple_in_closed_top_drawer" else 10


def ensure_locks() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    source_lock = LAYER_REFERENCE / "CANDIDATES_LOCK.json"
    target_lock = ROOT / "CANDIDATES_LOCK.json"
    if not target_lock.exists():
        target_lock.write_bytes(source_lock.read_bytes())
    seed_lock = {
        "protocol_id": "PROMPT_ATTN_L11_MATCHED_FULL_9TASK_0_299_V1",
        "policy": "L11 prompt_single, own-state Standard-SHR KMeans m_t; canonical snapshots reused",
        "seed_range": [0, 299],
        "existing_seed_range": [0, 99],
        "new_seed_range": [100, 299],
        "new_episode_count": 1800,
        "tasks": list(TASKS),
        "arm": ARM,
        "canonical_snapshot_artifact": str(CANONICAL),
    }
    (ROOT / "CLOSED_LOOP_SEEDS.json").write_text(json.dumps(seed_lock, indent=2) + "\n")


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for task in TASKS:
        step = chunk_size(task)
        for lo in range(SEED_START, SEED_END + 1, step):
            hi = min(lo + step - 1, SEED_END)
            jobs.append(Job(task=task, lo=lo, hi=hi))
    # Longest-processing-time-first reduces the final worker tail.
    return sorted(jobs, key=lambda job: (-job.estimated_seconds, job.task, job.lo))


def main() -> None:
    ensure_locks()
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / "coordinator_state.json"
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{REPO}:{pcd}:{env.get('PYTHONPATH', '')}",
    })

    all_jobs = build_jobs()
    queue = collections.deque(job for job in all_jobs if not complete(job))
    active: dict[tuple[int, int], tuple[subprocess.Popen, Job, object, Path, float]] = {}
    available_after = {slot: 0.0 for slot in SLOTS}
    capacity = dict(INITIAL_CAPACITY)
    device_lost = collections.Counter()
    events: list[dict] = []
    finished = len(all_jobs) - len(queue)

    def persist() -> None:
        payload = {
            "protocol_id": "PROMPT_ATTN_L11_MATCHED_FULL_9TASK_0_299_V1",
            "total_jobs": len(all_jobs),
            "new_episode_target": 1800,
            "completed_episodes": len(list((RUN_ROOT / "episodes").rglob("*_summary.json")))
            if (RUN_ROOT / "episodes").exists() else 0,
            "completed_jobs": sum(complete(job) for job in all_jobs),
            "queued_or_active_jobs": len(queue) + len(active),
            "active": [entry[1].name for entry in active.values()],
            "capacity": capacity,
            "updated": time.time(),
            "events": events[-500:],
        }
        state_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps({
        "protocol": "PROMPT_ATTN_L11_MATCHED_FULL_9TASK_0_299_V1",
        "tasks": list(TASKS),
        "new_seeds": [SEED_START, SEED_END],
        "new_episodes": 1800,
        "queued_jobs": len(queue),
        "slots": SLOTS,
    }), flush=True)
    persist()

    while queue or active:
        for gpu, slot_index in SLOTS:
            slot = (gpu, slot_index)
            if slot in active or slot_index >= capacity[gpu] or not queue:
                continue
            if time.monotonic() < available_after[slot]:
                continue
            job = queue.popleft()
            if complete(job):
                finished += 1
                persist()
                continue
            attempt = job.attempts + 1
            job = Job(job.task, job.lo, job.hi, attempt)
            log = logs / f"{job.name}_gpu{gpu}_slot{slot_index}_attempt{attempt}.log"
            handle = log.open("a")
            command = [
                sys.executable,
                str(ROLLOUT),
                "--artifact", str(RUN_ROOT),
                "--layer-artifact", str(ROOT),
                "--closed-loop-v1", str(LAYER_REFERENCE),
                "--snapshot-artifact", str(CANONICAL),
                "--task", job.task,
                "--seeds", f"{job.lo}-{job.hi}",
                "--gpu", str(gpu),
                "--worker-id", f"full_l11_gpu{gpu}_slot{slot_index}_{job.task}",
                "--config-tasks", ",".join(TASKS),
                "--arms", ARM,
            ]
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            active[slot] = (process, job, handle, log, time.monotonic())
            events.append({"event": "start", "job": job.name, "slot": slot, "pid": process.pid, "attempt": attempt})
            print(json.dumps(events[-1]), flush=True)
            persist()
            time.sleep(LAUNCH_STAGGER_SECONDS)

        time.sleep(5)
        for slot, (process, job, handle, log, started) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[slot]
            available_after[slot] = time.monotonic() + 45
            if status == 0 and complete(job):
                finished += 1
                events.append({"event": "complete", "job": job.name, "slot": slot, "seconds": round(time.monotonic() - started)})
                print(json.dumps(events[-1]), flush=True)
                persist()
                continue

            tail = log.read_text(errors="replace")[-30000:].lower()
            oom = "out of memory" in tail
            vk_lost = "devicelost" in tail or "device lost" in tail or status == -6
            if vk_lost or oom:
                device_lost[slot[0]] += 1
                capacity[slot[0]] = 1
                for candidate in SLOTS:
                    if candidate[0] == slot[0]:
                        available_after[candidate] = max(available_after[candidate], time.monotonic() + 120)
            if job.attempts < MAX_ATTEMPTS:
                queue.appendleft(job)
                events.append({
                    "event": "retry", "job": job.name, "slot": slot,
                    "attempt": job.attempts, "status": status, "oom": oom,
                    "device_lost": vk_lost, "log": str(log),
                })
                print(json.dumps(events[-1]), flush=True)
                persist()
            else:
                payload = {"complete": False, "failed_job": job.name, "status": status, "log": str(log)}
                (ROOT / "FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
                raise RuntimeError(payload)

    missing = [job.name for job in all_jobs if not complete(job)]
    if missing:
        payload = {"complete": False, "missing": missing}
        (ROOT / "FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(payload)
    failed_path = ROOT / "FAILED.json"
    if failed_path.exists():
        failed_path.unlink()
    (ROOT / "COMPLETE").write_text("1800/1800 L11-Matched canonical seeds 100-299 episodes complete\n")
    persist()
    print(json.dumps({"complete": True, "new_episodes": 1800, "completed_jobs": finished}), flush=True)


if __name__ == "__main__":
    main()
