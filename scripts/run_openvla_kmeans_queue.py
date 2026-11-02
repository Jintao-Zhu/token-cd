#!/usr/bin/env python3
"""Fail-closed, non-destructive K16/K32 queue (launch manually in tmux ovkm_queue).

Requires WORK_PLAN.json, CONTROL_REUSE_AUDIT.json, STATIC_VALIDATION.json and
EXPERIMENT_LOCK.json in the full artifact, plus EXPERIMENT_LOCK.json in smoke.
Both approval JSONs require literal technical_pass: true. STATIC_VALIDATION
code_sha256 must cover REQUIRED_CODE and every supplied entry is verified.

--check-only validates gates/plan/locks without querying GPUs or writing status.
Normal mode records STATUS.json + append-only STATUS_HISTORY.jsonl. After gate
approval it waits >=60s between occupancy checks until all ovtm_* tmux sessions
exit AND GPUs 1/2/3 each use <1024 MiB with no compute processes. No kill, retries,
or partial-block resume. RUN_STARTED.json is exclusive and forbids all reruns
once smoke launches. A failed prelaunch gate may be fixed and manually rerun.
Workers inherit the flock descriptor, retaining exclusion if the supervisor dies.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

from plan_openvla_kmeans_budget import (
    REPO, ARTIFACT, SMOKE_ARTIFACT, PROTOCOL, SMOKE_PROTOCOL,
    SMOKE_TASKS, SMOKE_ARMS, GPUS, build_plan,
)

RUNNER = "research/semantic_token_cd/openvla_kmeans_budget_rollout.py"
VALIDATOR = "scripts/verify_openvla_kmeans_smoke.py"
ACTIVATION = "scripts/activate_simpler_local.sh"
SELF = "scripts/run_openvla_kmeans_queue.py"
REQUIRED_CODE = {RUNNER, VALIDATOR, ACTIVATION, SELF, "scripts/plan_openvla_kmeans_budget.py"}
GATE_PATHS = [ARTIFACT / "CONTROL_REUSE_AUDIT.json", ARTIFACT / "STATIC_VALIDATION.json",
              ARTIFACT / "WORK_PLAN.json", ARTIFACT / "EXPERIMENT_LOCK.json",
              SMOKE_ARTIFACT / "EXPERIMENT_LOCK.json"]


class GateError(RuntimeError):
    pass


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"Expected JSON object: {path}")
    return value


def validate_gates(expected_fingerprint: str | None = None) -> tuple[dict, str]:
    """No shell, writes or GPU imports; fail closed for every missing/malformed gate."""
    documents = {str(path): read_object(path) for path in GATE_PATHS}
    for name in ("CONTROL_REUSE_AUDIT.json", "STATIC_VALIDATION.json"):
        if documents[str(ARTIFACT / name)].get("technical_pass") is not True:
            raise GateError(f"{name}: technical_pass must be literal true")
    plan = documents[str(ARTIFACT / "WORK_PLAN.json")]
    if plan != build_plan():
        raise GateError("WORK_PLAN.json differs from the fixed 1800 episode / 198 chunk LPT plan")
    for root in (ARTIFACT, SMOKE_ARTIFACT):
        if not documents[str(root / "EXPERIMENT_LOCK.json")]:
            raise GateError(f"Empty immutable experiment lock: {root}")
    manifest = documents[str(ARTIFACT / "STATIC_VALIDATION.json")].get("code_sha256")
    if not isinstance(manifest, dict) or not REQUIRED_CODE.issubset(manifest):
        raise GateError(f"code_sha256 must cover at least {sorted(REQUIRED_CODE)}")
    for relative, approved_hash in manifest.items():
        if not isinstance(relative, str) or not isinstance(approved_hash, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", approved_hash):
            raise GateError("Malformed code_sha256 entry")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or str(path) != relative:
            raise GateError(f"Manifest path is not canonical repo-relative: {relative}")
        resolved = (REPO / path).resolve(strict=True)
        try:
            resolved.relative_to(REPO.resolve())
        except ValueError:
            raise GateError(f"Manifest path escapes repository: {relative}")
        if digest(resolved) != approved_hash.lower():
            raise GateError(f"Code hash drift: {relative}")
    # Pin approval bytes, plan and both locks, not just decoded JSON.
    fingerprint = hashlib.sha256(json.dumps({str(p): digest(p) for p in GATE_PATHS}, sort_keys=True).encode()).hexdigest()
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise GateError("Approval, plan or immutable lock changed since initial gate")
    return plan, fingerprint


class Status:
    def __init__(self, path: Path, run_id: str, **initial):
        self.path = path
        self.history = path.with_name(path.stem + "_HISTORY.jsonl")
        self.data = {"run_id": run_id, "pid": os.getpid(), "started_at": stamp(), **initial}

    def update(self, phase: str, **fields):
        self.data.update(phase=phase, updated_at=stamp(), **fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        with self.history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.data, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"[{self.data['updated_at']}] {self.path.name}: {phase}", flush=True)


def inspect_occupancy() -> dict:
    """Read only GPU 1/2/3; unknown telemetry is an error, never 'idle'."""
    tmux = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                          capture_output=True, text=True, timeout=30)
    if tmux.returncode == 0:
        sessions = [line.strip() for line in tmux.stdout.splitlines() if line.strip().startswith("ovtm_")]
    elif tmux.returncode == 1 and ("no server running" in tmux.stderr.lower() or
                                 ("error connecting to" in tmux.stderr.lower() and "no such file or directory" in tmux.stderr.lower())):
        sessions = []
    else:
        raise GateError(f"Cannot inspect tmux: rc={tmux.returncode} {tmux.stderr.strip()}")
    gpu_result = subprocess.run([
        "nvidia-smi", "--id=1,2,3", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=30, check=True)
    devices = {}
    for row in csv.reader(gpu_result.stdout.splitlines()):
        if len(row) != 3:
            raise GateError("Malformed GPU memory telemetry")
        index, gpu_uuid, memory = [field.strip() for field in row]
        index = int(index)
        if index not in GPUS or index in devices or not gpu_uuid:
            raise GateError("Unexpected or duplicated GPU in telemetry")
        used = int(memory)
        if used < 0:
            raise GateError("Negative GPU memory telemetry")
        devices[index] = {"uuid": gpu_uuid, "memory_used_mib": used, "compute_pids": []}
    if set(devices) != set(GPUS):
        raise GateError("Missing GPU telemetry")
    processes = subprocess.run([
        "nvidia-smi", "--id=1,2,3", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=30, check=True)
    by_uuid = {device["uuid"]: device for device in devices.values()}
    for row in csv.reader(processes.stdout.splitlines()):
        if len(row) != 2 or row[0].strip() not in by_uuid:
            raise GateError("Malformed/unexpected compute process telemetry")
        pid = int(row[1].strip())
        if pid <= 0:
            raise GateError("Invalid compute PID")
        by_uuid[row[0].strip()]["compute_pids"].append(pid)
    idle = not sessions and all(d["memory_used_mib"] < 1024 and not d["compute_pids"] for d in devices.values())
    return {"checked_at": stamp(), "taskmean_sessions": sessions, "gpus": devices, "idle": idle}


def wait_idle(status: Status, interval: float, fingerprint: str, phase: str):
    while True:
        validate_gates(fingerprint)
        occupancy = inspect_occupancy()
        status.update(phase, occupancy=occupancy,
                      waiting_for="all ovtm_* sessions gone; GPUs1/2/3 memory <1024 MiB and no compute processes",
                      next_check_after_seconds=None if occupancy["idle"] else interval)
        if occupancy["idle"]:
            return
        time.sleep(interval)


def activated_command(gpu: int, argv: list[str]) -> list[str]:
    if gpu not in GPUS:
        raise GateError(f"GPU {gpu} is forbidden")
    shell = ("set -euo pipefail; source " + shlex.quote(str(REPO / ACTIVATION)) +
             f"; export CUDA_VISIBLE_DEVICES={gpu}; exec " + shlex.join(argv))
    return ["bash", "-c", shell]


def launch(argv: list[str], gpu: int, log: Path, lock_fd: int) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Every child holds the supervisor flock. Never signal/kill unrelated work.
    with log.open("x", encoding="utf-8") as stream:
        return subprocess.Popen(activated_command(gpu, argv), cwd=REPO, env=env,
                                stdout=stream, stderr=subprocess.STDOUT,
                                pass_fds=(lock_fd,), start_new_session=True)


def runner_args(artifact: Path, task: str, arm: str, seeds: str, gpu: int, worker: str, protocol: str) -> list[str]:
    return ["python", RUNNER, "--artifact", str(artifact), "--task", task,
            "--seeds", seeds, "--gpu", str(gpu), "--worker-id", worker,
            "--arms", arm, "--protocol-id", protocol, "--lock-seeds", "200-299"]


def worker_status_path(name: str) -> Path:
    return ARTIFACT / "queue_workers" / f"{name}.json"


def run_worker(args) -> int:
    """Internal subprocess mode; no public independent launching without parent fd."""
    if args.lock_fd is None or not args.fingerprint or not args.run_id:
        raise GateError("Worker requires parent lock descriptor and pinned approvals")
    lock_path = ARTIFACT / ".queue_supervisor.lock"
    if os.fstat(args.lock_fd).st_ino != lock_path.stat().st_ino or os.fstat(args.lock_fd).st_dev != lock_path.stat().st_dev:
        raise GateError("Worker lock descriptor is not the supervisor lock")
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = read_object(ARTIFACT / "RUN_STARTED.json")
    if marker.get("run_id") != args.run_id or marker.get("fingerprint") != args.fingerprint:
        raise GateError("Worker launch marker mismatch")
    status = Status(worker_status_path(args.worker), args.run_id, worker=args.worker, completed_chunks=[], failures=[])
    try:
        plan, _ = validate_gates(args.fingerprint)
        workers = [w for w in plan["assignment"] if w["worker"] == args.worker]
        if len(workers) != 1:
            raise GateError("Unknown worker")
        worker = workers[0]
        status.update("worker_starting", gpu=worker["gpu"], total_chunks=len(worker["chunks"]))
        for chunk in worker["chunks"]:
            validate_gates(args.fingerprint)
            command = runner_args(ARTIFACT, chunk["task"], chunk["arm"], chunk["seeds"],
                                  worker["gpu"], worker["worker"], PROTOCOL)
            log = ARTIFACT / "logs" / f"{worker['worker']}__{chunk['chunk_id']}.log"
            status.update("chunk_starting", current_chunk=chunk, child_pid=None, log=str(log))
            child = launch(command, worker["gpu"], log, args.lock_fd)
            status.update("chunk_running", child_pid=child.pid)
            code = child.wait()
            if code != 0:
                status.data["failures"].append({"chunk_id": chunk["chunk_id"], "returncode": code, "at": stamp()})
                status.update("failed", child_pid=None, returncode=code, error="Chunk failed: worker stopped; no retry")
                return 1
            status.data["completed_chunks"].append(chunk["chunk_id"])
            status.update("chunk_completed", child_pid=None, returncode=0)
        status.update("completed", current_chunk=None)
        return 0
    except BaseException as exc:
        status.update("failed", error=f"{type(exc).__name__}: {exc}", no_children_killed=True)
        return 1


def ensure_pristine():
    if (ARTIFACT / "RUN_STARTED.json").exists():
        raise GateError("Prior launch marker exists: reruns/partial-block resume are forbidden")
    for root in (ARTIFACT, SMOKE_ARTIFACT):
        episodes = root / "episodes"
        if episodes.exists() and any(episodes.rglob("*")):
            raise GateError(f"Existing episode outputs: {episodes}; refusing automatic resume")


def supervise(args) -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT / ".queue_supervisor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another supervisor or its children still hold the exclusive lock", file=sys.stderr)
            return 1
        run_id = uuid.uuid4().hex
        status = Status(ARTIFACT / "STATUS.json", run_id, poll_seconds=args.poll_seconds,
                        workers={}, smoke_completed=[], completed_chunks=0)
        try:
            status.update("checking_gates")
            plan, fingerprint = validate_gates()
            ensure_pristine()
            status.update("gates_passed", gate_fingerprint=fingerprint)
            wait_idle(status, args.poll_seconds, fingerprint, "waiting_for_taskmean_and_idle_gpus")
            validate_gates(fingerprint)
            ensure_pristine()
            # Durable marker before any launch: ambiguous crash => manual review, not resume.
            with (ARTIFACT / "RUN_STARTED.json").open("x", encoding="utf-8") as stream:
                json.dump({"run_id": run_id, "pid": os.getpid(), "created_at": stamp(), "fingerprint": fingerprint}, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            for task in SMOKE_TASKS:
                for arm in SMOKE_ARMS:
                    validate_gates(fingerprint)
                    name = f"ovkm_smoke__{task}__{arm}"
                    status.update("smoke_starting", smoke_current={"task": task, "arm": arm, "seeds": "200", "gpu": 1}, child_pid=None)
                    child = launch(runner_args(SMOKE_ARTIFACT, task, arm, "200", 1, name, SMOKE_PROTOCOL),
                                   1, SMOKE_ARTIFACT / "logs" / f"{name}.log", lock.fileno())
                    status.update("smoke_running", child_pid=child.pid)
                    code = child.wait()
                    if code != 0:
                        raise GateError(f"Smoke {task}/{arm} exited {code}; no retry, no full launch")
                    status.data["smoke_completed"].append({"task": task, "arm": arm, "seed": 200})
                    status.update("smoke_episode_completed", child_pid=None)
            validate_gates(fingerprint)
            child = launch(["python", VALIDATOR, "--artifact", str(SMOKE_ARTIFACT)],
                           1, SMOKE_ARTIFACT / "logs" / "smoke_validation.log", lock.fileno())
            status.update("validating_smoke", child_pid=child.pid)
            code = child.wait()
            status.update("smoke_validation_finished", child_pid=None, smoke_validator_returncode=code)
            if code != 0:
                raise GateError(f"Smoke validator exited {code}; full launch forbidden")
            wait_idle(status, args.poll_seconds, fingerprint, "waiting_for_full_launch_idle_gpus")
            validate_gates(fingerprint)
            children = {}
            # Exactly 12 worker subprocesses: fixed 4 per GPU, one runner at a time each.
            for worker in plan["assignment"]:
                name = worker["worker"]
                child = launch([sys.executable, SELF, "--worker", name, "--lock-fd", str(lock.fileno()),
                                "--fingerprint", fingerprint, "--run-id", run_id],
                               worker["gpu"], ARTIFACT / "logs" / f"{name}__queue.log", lock.fileno())
                children[name] = child
                status.data["workers"][name] = {"pid": child.pid, "gpu": worker["gpu"], "phase": "starting", "completed_chunks": []}
                status.update("launching_workers")
            while True:
                running = False
                failed = False
                count = 0
                for name, child in children.items():
                    code = child.poll()
                    entry = status.data["workers"][name]
                    path = worker_status_path(name)
                    if path.exists():
                        data = read_object(path)
                        if data.get("run_id") != run_id:
                            raise GateError(f"Stale worker status: {name}")
                        entry.update(data)
                    if code is None:
                        running = True
                    else:
                        entry["process_returncode"] = code
                        if code != 0 or entry.get("phase") != "completed":
                            entry["phase"] = "failed"
                            entry.setdefault("error", f"Worker exited {code} without valid completion")
                            failed = True
                    count += len(entry.get("completed_chunks", []))
                status.update("running_workers" if running else ("failed" if failed else "completed"),
                              completed_chunks=count, failed_workers=[name for name, entry in status.data["workers"].items() if entry["phase"] == "failed"],
                              next_check_after_seconds=args.poll_seconds if running else None)
                if not running:
                    if not failed and count != 198:
                        raise GateError(f"Incorrect completion count {count}, expected 198")
                    return 1 if failed else 0
                time.sleep(args.poll_seconds)
        except BaseException as exc:
            status.update("failed", error=f"{type(exc).__name__}: {exc}", no_children_killed=True,
                          note="No retry/resume. Any already running children retain exclusion lock and are not killed.")
            print(status.data["error"], file=sys.stderr)
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0, help="Occupancy/worker monitor interval, >=60 seconds")
    parser.add_argument("--check-only", action="store_true", help="Validate gates, plan, hashes and pristine outputs without GPU queries/writes")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--fingerprint", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 60:
        parser.error("--poll-seconds must be finite and >=60 (no rapid polling)")
    try:
        if args.worker:
            if args.check_only:
                raise GateError("--check-only cannot be combined with worker mode")
            return run_worker(args)
        if args.check_only:
            _, fingerprint = validate_gates()
            ensure_pristine()
            print(json.dumps({"technical_pass": True, "gate_fingerprint": fingerprint,
                              "episodes": 1800, "chunks": 198, "workers": 12, "launched": False}))
            return 0
        return supervise(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
