#!/usr/bin/env python3
"""Run frozen LIBERO checkpoint-transfer N=50 on model GPUs 1-6, EGL GPU7."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = ROOT / "artifacts/libero_checkpoint_transfer_base_openvla_n50_v1_20260925"
WORKER = ROOT / "research/semantic_token_cd/libero_action_value_gate_worker.py"
PROTOCOL = "LIBERO90_CHECKPOINT_TRANSFER_BASE_OPENVLA_VS_LIBERO90_V1"
PREFLIGHT_PROTOCOL = PROTOCOL + "_PREFLIGHT"
GPUS = (1, 2, 3, 4, 5, 6)
WORKERS_PER_GPU = 2
WORKER_SLOTS = [(gpu, slot) for gpu in GPUS for slot in range(1, WORKERS_PER_GPU + 1)]
RENDER_GPU = 7
ALLOCATION_AMENDMENT = "02"
BASE_CHECKPOINT = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source/pretrained/openvla-7b")
REFERENCE_STATS = Path("/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90/dataset_statistics.json")
XID_RE = re.compile(r"NVRM: Xid.*PCI:0000:(?:21|41|61|81|a1|c1|e1):00", re.IGNORECASE)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_status(status: str, started: str, processes: dict[str, subprocess.Popen],
                 exit_codes: dict[str, int | None], reason: str | None = None) -> None:
    atomic_json(ART / "RUN_STATUS.json", {
        "protocol_id": PROTOCOL,
        "status": status,
        "started_local": started,
        "updated_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inference_gpus": list(GPUS),
        "render_gpu": RENDER_GPU,
        "allocation_amendment": ALLOCATION_AMENDMENT,
        "worker_pids": {key: p.pid for key, p in processes.items()},
        "worker_exit_codes": {key: c for key, c in exit_codes.items()},
        "stop_reason": reason,
    })


def terminate_all(processes: dict[str, subprocess.Popen]) -> None:
    for proc in processes.values():
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 20
    for proc in processes.values():
        if proc.poll() is None:
            try:
                proc.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
    for proc in processes.values():
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def prepare_queues() -> None:
    cases = [json.loads(line) for line in (ART / "cases/all.jsonl").read_text().splitlines() if line.strip()]
    ids = [case["case_id"] for case in cases]
    if len(cases) != 50 or len(set(ids)) != 50:
        raise RuntimeError(f"frozen cohort integrity failure: cases={len(cases)} unique={len(set(ids))}")
    if any(case.get("protocol_id") != PROTOCOL for case in cases):
        raise RuntimeError("case protocol mismatch")
    if any(case.get("renderer_backend") != "egl" for case in cases):
        raise RuntimeError("renderer mismatch")
    if any(case.get("arms") != ["vanilla", "matched"] for case in cases):
        raise RuntimeError("arm definition mismatch")
    # Deterministic round-robin assignment keeps all twelve queues balanced.
    queues = {slot: cases[i::len(WORKER_SLOTS)] for i, slot in enumerate(WORKER_SLOTS)}
    flat_ids = [case["case_id"] for slot in WORKER_SLOTS for case in queues[slot]]
    if len(flat_ids) != 50 or set(flat_ids) != set(ids):
        raise RuntimeError("worker queues do not partition the frozen cohort")
    for gpu, worker_slot in WORKER_SLOTS:
        rows = queues[(gpu, worker_slot)]
        path = ART / "cases" / f"checkpoint12_gpu{gpu}_slot{worker_slot}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    atomic_json(ART / "WORKER_QUEUES_12WORKER.json", {
        "protocol_id": PROTOCOL,
        "case_count": 50,
        "unique_case_count": len(set(flat_ids)),
        "workers_per_gpu": WORKERS_PER_GPU,
        "queue_case_ids": {f"gpu{gpu}_slot{slot}": [c["case_id"] for c in queues[(gpu, slot)]]
                           for gpu, slot in WORKER_SLOTS},
    })


def main() -> int:
    os.chdir(ROOT)
    if not (ART / "preflight_corrected/pairs/task18__init049_preflight.json").is_file():
        raise RuntimeError("protocol-correct preflight pair is missing")
    pf = json.loads((ART / "preflight_corrected/pairs/task18__init049_preflight.json").read_text())
    if pf.get("protocol_id") != PREFLIGHT_PROTOCOL or not pf.get("paired_initial_state_match"):
        raise RuntimeError("protocol-correct preflight did not pass")
    if pf.get("initial_state_sha256") != "c97490b46ceff6ad555e184852095f73a8ad95421ce90972538f409c285dd527":
        raise RuntimeError("preflight state differs from frozen expected state")
    if pf.get("initial_rgb_sha256") != "69b8b0eec73941060098efefe8e7dbe52b01ddb22ccf275259faf0291050fd6c":
        raise RuntimeError("preflight RGB differs from frozen expected RGB")
    prepare_queues()

    started_dt = dt.datetime.now().astimezone()
    started = started_dt.isoformat(timespec="seconds")
    journal_since = started_dt.strftime("%Y-%m-%d %H:%M:%S")
    base_env = os.environ.copy()
    base_env["TOKENIZERS_PARALLELISM"] = "false"
    base_env.pop("MUJOCO_EGL_DEVICE_ID", None)
    base_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    processes: dict[str, subprocess.Popen] = {}
    streams = {}
    exits: dict[str, int | None] = {f"gpu{gpu}_slot{slot}": None for gpu, slot in WORKER_SLOTS}
    for gpu, slot in WORKER_SLOTS:
        worker_key = f"gpu{gpu}_slot{slot}"
        log_path = ART / "logs" / f"checkpoint_transfer12_{worker_key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        streams[worker_key] = log_path.open("w", buffering=1)
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        args = [
            sys.executable, str(WORKER),
            "--artifact", str(ART), "--gpu", str(gpu), "--render-gpu", str(RENDER_GPU),
            "--renderer", "egl", "--protocol-id", PROTOCOL,
            "--worker-id", f"base_checkpoint12_{worker_key}_render7",
            "--cases-file", str(ART / "cases" / f"checkpoint12_{worker_key}.jsonl"),
            "--checkpoint", str(BASE_CHECKPOINT),
            "--dataset-statistics", str(REFERENCE_STATS),
            "--checkpoint-revision", "simpler-base-openvla-7b",
        ]
        processes[worker_key] = subprocess.Popen(args, cwd=ROOT, env=env,
                                                 stdout=streams[worker_key], stderr=subprocess.STDOUT)
    write_status("RUNNING", started, processes, exits)
    print(json.dumps({"status": "RUNNING", "started_local": started,
                      "worker_pids": {str(g): p.pid for g, p in processes.items()},
                      "inference_gpus": GPUS, "workers_per_gpu": WORKERS_PER_GPU,
                      "worker_count": len(WORKER_SLOTS), "render_gpu": RENDER_GPU,
                      "cases": 50}), flush=True)
    try:
        while True:
            journal = subprocess.run(
                ["journalctl", "-k", "--since", journal_since, "--no-pager", "-o", "cat"],
                check=False, capture_output=True, text=True, timeout=15,
            )
            xid_lines = [line for line in journal.stdout.splitlines() if XID_RE.search(line)]
            if xid_lines:
                reason = "new_xid: " + " | ".join(xid_lines[-3:])
                terminate_all(processes)
                exits.update({g: p.poll() for g, p in processes.items()})
                write_status("STOPPED_TECHNICAL_XID", started, processes, exits, reason)
                print(json.dumps({"status": "STOPPED_TECHNICAL_XID", "reason": reason}), flush=True)
                return 20
            failed = [(g, p.returncode) for g, p in processes.items()
                      if p.poll() is not None and p.returncode != 0]
            if failed:
                reason = f"worker_nonzero_exit: {failed}"
                terminate_all(processes)
                exits.update({g: p.poll() for g, p in processes.items()})
                write_status("STOPPED_WORKER_FAILURE", started, processes, exits, reason)
                print(json.dumps({"status": "STOPPED_WORKER_FAILURE", "reason": reason}), flush=True)
                return 21
            if all(p.poll() is not None for p in processes.values()):
                exits.update({g: p.poll() for g, p in processes.items()})
                if all(code == 0 for code in exits.values()):
                    write_status("WORKERS_EXITED_AWAITING_AUDIT", started, processes, exits)
                    print(json.dumps({"status": "WORKERS_EXITED_AWAITING_AUDIT",
                                      "worker_exit_codes": exits}), flush=True)
                    return 0
                write_status("STOPPED_WORKER_FAILURE", started, processes, exits,
                             "one or more workers exited nonzero")
                return 21
            write_status("RUNNING", started, processes, {g: p.poll() for g, p in processes.items()})
            time.sleep(10)
    except Exception as exc:
        terminate_all(processes)
        exits.update({g: p.poll() for g, p in processes.items()})
        write_status("STOPPED_SUPERVISOR_ERROR", started, processes, exits,
                     f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
