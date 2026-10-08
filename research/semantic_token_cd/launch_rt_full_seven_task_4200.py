#!/usr/bin/env python3
"""Preflight then run paired seven-task SIMPLER RT-full study on GPUs 1-3."""
from __future__ import annotations

import json
import os
import collections
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PYTHON = Path("/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python")
PCD = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
OUT = ROOT / "artifacts/simpler_rt_full_7task_4200_v1"
CANON = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
DEPTH0 = ROOT / "artifacts/simpler_rt_depth0_7task_4200_v1"
RUN = ROOT / "research/semantic_token_cd/run_rt_full_seven_task_paired.py"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
GPUS = (1, 2, 3)
WORKERS_PER_GPU = 3
SEED_RANGES = ((0, 99), (100, 199), (200, 299))
PROTOCOL = "SIMPLER_RT_FULL_7TASK_L11_MATCHED_PAIRED_0_299_V1"
RENDERER = "SAPIEN RT full depth8, spp=32, denoiser=true"
MAX_ATTEMPTS = 6


@dataclass(frozen=True)
class Job:
    task: str
    lo: int
    hi: int
    attempts: int = 0

    @property
    def name(self) -> str:
        return f"{self.task}_{self.lo:03d}_{self.hi:03d}"


def summary_path(task: str, arm: str, seed: int) -> Path:
    return OUT / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"


def job_complete(job: Job) -> bool:
    for seed in range(job.lo, job.hi + 1):
        for arm in ("vanilla", "l11_matched"):
            p = summary_path(job.task, arm, seed)
            arrays = p.with_name(p.name.replace("_summary.json", "_arrays.npz"))
            if not p.is_file() or not arrays.is_file():
                return False
    return True


def setup_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    snapshot_lock = json.loads((DEPTH0 / "PROTOCOL_LOCK.json").read_text())
    lock = {
        "protocol_id": PROTOCOL,
        "benchmark": "SIMPLER",
        "purpose": "paired RT-full comparison against the existing RT-depth0 and IBL results",
        "tasks": list(TASKS),
        "arms": ["vanilla", "l11_matched"],
        "seeds": list(range(300)),
        "episodes_total": 4200,
        "canonical_root": str(CANON),
        "paired_seed_reference": str(DEPTH0),
        "same_seed_range_as_depth0": True,
        "renderer": {
            "shader_dir": "rt", "rt_max_path_depth": 8,
            "rt_samples_per_pixel": 32, "rt_use_denoiser": True,
            "offscreen_only": True,
        },
        "method": "Historical PromptAttentionSHRInference matched-budget path; lambda=0.5; no method changes",
        "preflight": "paired Vanilla/L11 seed 0 for every task, run before formal shards; included in 0-299 cohort",
        "gpu_indices": list(GPUS),
        "workers_per_gpu": WORKERS_PER_GPU,
        "snapshot_protocol": snapshot_lock["protocol_id"],
        "cross_renderer_identity_rule": "canonical snapshot and initial simulator state must match depth0; RGB expected to differ by renderer; paired RT8 arms must have identical RGB",
        "retry_policy": "retry failed shard up to six times; DeviceLost only triggers cooldown and lower same-GPU concurrency",
    }
    path = OUT / "PROTOCOL_LOCK.json"
    encoded = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError(f"existing protocol lock differs: {path}")
    path.write_text(encoded)


def validate_preflight() -> None:
    for task in TASKS:
        rows = {}
        for arm in ("vanilla", "l11_matched"):
            p = summary_path(task, arm, 0)
            row = json.loads(p.read_text())
            if row.get("protocol_id") != PROTOCOL or row.get("renderer") != RENDERER:
                raise RuntimeError(f"preflight protocol/renderer mismatch: {p}")
            if row.get("canonical_snapshot_sha256") != json.loads(
                (CANON / "episodes" / task / "vanilla" / "episode_000_summary.json").read_text()
            ).get("canonical_snapshot_sha256"):
                raise RuntimeError(f"preflight canonical snapshot mismatch: {task}")
            rows[arm] = row
        for field in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
            if rows["vanilla"].get(field) != rows["l11_matched"].get(field):
                raise RuntimeError(f"preflight paired identity mismatch {field}: {task}")
        trace = rows["l11_matched"].get("selector_trace", [])
        if not trace or any(x.get("coverage_exact") is not True for x in trace):
            raise RuntimeError(f"preflight matched coverage audit failed: {task}")
        if any(abs(float(x.get("lambda", -1)) - .5) > 1e-12 for x in trace):
            raise RuntimeError(f"preflight lambda audit failed: {task}")
    (OUT / "PREFLIGHT_PASS").write_text(
        "PASS: seven tasks, paired seed 0, RT path depth 8; snapshot/state/RGB and Matched audits validated.\n"
    )


def run_preflight(env: dict[str, str], log_path: Path) -> None:
    # At most one worker per GPU during RT8 smoke. Seven tasks run in batches,
    # keeping the preflight conservative before 3 workers/GPU are admitted.
    for start in range(0, len(TASKS), len(GPUS)):
        procs = []
        for idx, task in enumerate(TASKS[start:start + len(GPUS)]):
            gpu = GPUS[idx]
            name = f"preflight_gpu{gpu}_{task}"
            log = OUT / "logs" / f"{name}.log"
            command = [str(PYTHON), str(RUN), "--gpu", str(gpu), "--task", task,
                       "--seeds", "0", "--worker-id", name, "--output", str(OUT)]
            with log.open("a") as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            procs.append((task, proc, log))
        failures = []
        for task, proc, log in procs:
            status = proc.wait()
            if status != 0:
                failures.append({"task": task, "status": status, "log": str(log)})
        if failures:
            raise RuntimeError(f"RT8 preflight failed; no formal work launched: {failures}")
    validate_preflight()
    with log_path.open("a") as log:
        log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] PREFLIGHT PASS for all 7 tasks\n")


def build_jobs() -> list[Job]:
    jobs = [Job(task, lo, hi) for lo, hi in SEED_RANGES for task in TASKS]
    return [job for job in jobs if not job_complete(job)]


def main() -> None:
    setup_lock()
    root_log = OUT / "logs" / "launcher.log"
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONPATH": f"{ROOT}:{ROOT / 'task1/shim_site'}:{PCD}:{env.get('PYTHONPATH', '')}",
    })
    if not (OUT / "PREFLIGHT_PASS").exists():
        with root_log.open("a") as log:
            log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] PRE-FLIGHT RT8: 7 paired seed-0 cases\n")
        run_preflight(env, root_log)

    jobs = [Job(task, lo, hi) for lo, hi in SEED_RANGES for task in TASKS]
    pending = [j for j in jobs if not job_complete(j)]
    # Interleave task and seed chunks so the first wave is spread across GPUs
    # rather than placing the three slow Apple shards on a single device.
    q: collections.deque[Job] = collections.deque(pending)
    capacities = {gpu: WORKERS_PER_GPU for gpu in GPUS}
    cooldown_until = {gpu: 0.0 for gpu in GPUS}
    active: dict[int, tuple[subprocess.Popen, int, Job, Path, float]] = {}
    slot_gpu = {slot: gpu for gpu in GPUS for slot in range((gpu - 1) * WORKERS_PER_GPU,
                                                           gpu * WORKERS_PER_GPU)}
    events: list[dict] = []

    def persist() -> None:
        done = sum(1 for j in jobs if job_complete(j))
        data = {
            "protocol_id": PROTOCOL,
            "completed_episodes": sum(
                1 for task in TASKS for arm in ("vanilla", "l11_matched")
                for seed in range(300) if summary_path(task, arm, seed).is_file()
            ),
            "target_episodes": 4200,
            "completed_shards": done,
            "total_shards": len(jobs),
            "queued_shards": len(q),
            "active": [{"gpu": v[1], "job": v[2].name, "pid": v[0].pid} for v in active.values()],
            "gpu_capacity": capacities,
            "updated_unix": time.time(),
            "events": events[-300:],
        }
        (OUT / "coordinator_state.json").write_text(json.dumps(data, indent=2) + "\n")

    with root_log.open("a") as log:
        log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] FORMAL launch: {len(q)} shards, up to 3/GPU\n")

    while q or active:
        occupied_by_gpu = {gpu: sum(v[1] == gpu for v in active.values()) for gpu in GPUS}
        free_slots = [s for s, gpu in slot_gpu.items()
                      if not any(v[1] == gpu and v[0] is not None and v[4] == s for v in active.values())]
        # Build occupied slot set explicitly (slot is stored on tuple position 4 below).
        occupied_slots = {v[4] for v in active.values()}
        free_slots = [s for s in slot_gpu if s not in occupied_slots]
        for slot in free_slots:
            if not q:
                break
            gpu = slot_gpu[slot]
            if occupied_by_gpu[gpu] >= capacities[gpu] or time.monotonic() < cooldown_until[gpu]:
                continue
            job = q.popleft()
            if job_complete(job):
                continue
            attempt = job.attempts + 1
            job = Job(job.task, job.lo, job.hi, attempt)
            worker = f"rtfull_gpu{gpu}_slot{slot}_{job.task}_{job.lo:03d}_{job.hi:03d}_try{attempt}"
            log_path = OUT / "logs" / f"{worker}.log"
            command = [str(PYTHON), str(RUN), "--gpu", str(gpu), "--task", job.task,
                       "--seeds", f"{job.lo}-{job.hi}", "--worker-id", worker,
                       "--output", str(OUT)]
            handle = log_path.open("a")
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            # Store slot in the final tuple field for unambiguous scheduler accounting.
            active[proc.pid] = (proc, gpu, job, (handle, log_path), float(slot))
            occupied_by_gpu[gpu] += 1
            events.append({"event": "start", "gpu": gpu, "slot": slot,
                           "job": job.name, "pid": proc.pid, "attempt": attempt})
            persist()

        time.sleep(5)
        for pid, (proc, gpu, job, handle_info, slot_float) in list(active.items()):
            status = proc.poll()
            if status is None:
                continue
            handle, log_path = handle_info
            handle.close()
            del active[pid]
            if status == 0 and job_complete(job):
                events.append({"event": "complete", "gpu": gpu, "job": job.name,
                               "elapsed_s": round(time.time() - log_path.stat().st_mtime, 1)})
                persist()
                continue
            text = log_path.read_text(errors="replace").lower()
            device_lost = "devicelost" in text or "device lost" in text
            oom = "out of memory" in text
            retry_count = job.attempts
            if device_lost or oom:
                capacities[gpu] = max(1, capacities[gpu] - 1)
                cooldown_until[gpu] = time.monotonic() + 120
            if retry_count < MAX_ATTEMPTS:
                q.appendleft(Job(job.task, job.lo, job.hi, retry_count))
                events.append({"event": "retry", "gpu": gpu, "job": job.name,
                               "status": status, "device_lost": device_lost,
                               "oom": oom, "next_attempt": retry_count + 1,
                               "log": str(log_path)})
                with root_log.open("a") as log:
                    log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] RETRY {job.name} on GPU {gpu}; status={status}; DeviceLost={device_lost}; OOM={oom}\n")
            else:
                payload = {"complete": False, "failed_shard": job.name,
                           "gpu": gpu, "status": status, "log": str(log_path)}
                (OUT / "FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
                raise RuntimeError(payload)
            persist()

    # Keep the formal artifacts resumable and prove exact episode accounting.
    (OUT / "FORMAL_WORKERS_COMPLETE").write_text("All 21 formal shards completed.\n")
    with root_log.open("a") as log:
        log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] All formal shards complete; running final validator\n")
    subprocess.run([str(PYTHON), str(ROOT / "research/semantic_token_cd/validate_rt_full_seven_task.py"),
                    "--output", str(OUT), "--finalize"], cwd=ROOT, env=env, check=True)
    with root_log.open("a") as log:
        log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] COMPLETE: validated 4200 RT-full episodes\n")


if __name__ == "__main__":
    main()
