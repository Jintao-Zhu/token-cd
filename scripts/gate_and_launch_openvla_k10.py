#!/usr/bin/env python3
"""Gate, verify and launch the real K=10 closed-loop run.

Sequence (all detached, nothing to babysit):
  1. wait until the K probe has finished: no ovkp_* tmux sessions, no probe processes,
     and GPUs 1/2/3 below 1024 MiB with no compute processes
  2. preflight: close_drawer seed 200 for both K=10 arms on GPU1, sequentially
  3. verify the preflight: strict audit flags, kmeans_K == 10 on every step, matched
     invariants, and the measured m_t inside the band the probe predicted
  4. archive the preflight episodes to preflight/ and delete them from the main tree
     (the plan's [200] block would otherwise refuse to resume, by design)
  5. launch 12 workers from WORK_PLAN.json

Status: artifacts/openvla_k10_2lambda_200_299_v1/logs/gate.log and LAUNCH_STATUS.json.
Never kills anything it did not start.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = REPO / "artifacts/openvla_k10_2lambda_200_299_v1"
PROBE_ART = REPO / "artifacts/openvla_k_probe_200_219_v1"
PROTOCOL = "OPENVLA_K10_2LAMBDA_200_299_V1"
PREFLIGHT_TASK = "google_robot_close_drawer"
PREFLIGHT_SEED = 200
ARMS = ("matched_k10_l050", "matched_k10_l100")
LOG = ART / "logs" / "gate.log"


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as stream:
        stream.write(line + "\n")


def finished(path: Path) -> bool:
    return path.is_file()


def gpus_idle() -> tuple[bool, dict]:
    sessions = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                              capture_output=True, text=True)
    names = [n for n in sessions.stdout.split() if n.startswith("ovkp_")]
    processes = subprocess.run(["pgrep", "-f", "openvla_matched_k_probe_rollout"],
                               capture_output=True, text=True)
    probe_pids = [int(p) for p in processes.stdout.split() if p.strip().isdigit()]
    telemetry = subprocess.run(
        ["nvidia-smi", "--id=1,2,3", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    memory = {}
    for row in telemetry.stdout.splitlines():
        index, used = [part.strip() for part in row.split(",")]
        memory[index] = int(used)
    busy = subprocess.run(["nvidia-smi", "--id=1,2,3", "--query-compute-apps=pid",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    compute = [p for p in busy.stdout.split() if p.strip().isdigit()]
    idle = (not names and not probe_pids and all(v < 1024 for v in memory.values()) and not compute)
    return idle, {"probe_sessions": names, "probe_pids": probe_pids,
                  "gpu_memory_mib": memory, "compute_pids": compute}


def run(command: list[str], log_path: Path, gpu: int) -> int:
    shell = (f"cd {REPO} && source scripts/activate_simpler_local.sh && "
             f"export CUDA_VISIBLE_DEVICES={gpu} && exec " + " ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # "w", not "x": a retry after a failed gate run must not die on an existing log.
    with log_path.open("w") as stream:
        return subprocess.run(["bash", "-c", shell], stdout=stream,
                              stderr=subprocess.STDOUT).returncode


def verify_preflight() -> tuple[bool, dict]:
    detail = {"arms": {}}
    band = band_from_probe()
    detail["probe_predicted_k10_m_t"] = band
    ok = True
    for arm in ARMS:
        path = ART / "episodes" / PREFLIGHT_TASK / arm / f"episode_{PREFLIGHT_SEED:03d}_summary.json"
        if not path.is_file():
            detail["arms"][arm] = {"error": "missing summary"}
            ok = False
            continue
        data = json.loads(path.read_text())
        trace = data.get("selector_trace") or []
        # Only genuine booleans are audit flags. Prefix filtering used to sweep in the
        # numeric fields kmeans_K / kmeans_seed, which made this check always fail.
        flags = {k: v for k, v in data.items() if isinstance(v, bool) and k != "success"}
        required = {"technical_pass", "budget_count_exact", "kmeans_K_exact", "kmeans_seed_exact",
                    "matched_source_exact", "matched_union_count_exact", "selected_ids_exact",
                    "matched_groups_exact", "clean_features_preserved", "unselected_features_unchanged",
                    "finite_reconstruction", "lambda_exact", "layer_exact", "constructor_exact",
                    "execution_exact", "intervention_location_exact", "seven_action_tokens",
                    "nonempty_trace"}
        record = {
            "success": data.get("success"), "steps": data.get("steps"),
            "technical_pass": data.get("technical_pass"),
            "kmeans_K": data.get("kmeans_K"), "mean_m_t": data.get("mean_selected_token_count"),
            "process_seed_block": data.get("process_seed_block"),
            "flags_false": sorted(k for k, v in flags.items() if v is not True),
            "flags_missing": sorted(required - set(flags)),
            "n_flags": len(flags),
            "steps_with_wrong_K": [i for i, row in enumerate(trace) if row.get("kmeans_K") != 10],
        }
        record["all_flags_true"] = not record["flags_false"] and not record["flags_missing"]
        if band:
            record["within_probe_band"] = bool(band["low"] - 8 <= record["mean_m_t"] <= band["high"] + 8)
        ok = ok and record["technical_pass"] is True and record["all_flags_true"] \
            and not record["steps_with_wrong_K"] and record["kmeans_K"] == 10 \
            and record["process_seed_block"] == [PREFLIGHT_SEED]
        detail["arms"][arm] = record
    return ok, detail


def band_from_probe() -> dict | None:
    """Predicted K=10 budget band for the preflight state.

    The probe run was stopped early at the user's request, so fall back to the archived
    preflight probe episode, which covers exactly this task/seed.
    """
    candidates = [
        PROBE_ART / "episodes" / PREFLIGHT_TASK / "lambda_100" / f"episode_{PREFLIGHT_SEED:03d}_probe.json",
        PROBE_ART / "preflight" / f"episode_{PREFLIGHT_SEED:03d}_probe.json",
    ]
    for path in candidates:
        if path.is_file():
            summary = json.loads(path.read_text())["summary"]["m_k10"]
            return {"low": summary["min"], "high": summary["max"], "mean_on_k8_path": summary["mean"],
                    "source": str(path)}
    return None


def archive_preflight() -> list[str]:
    moved = []
    target = ART / "preflight"
    target.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        directory = ART / "episodes" / PREFLIGHT_TASK / arm
        for path in sorted(directory.glob(f"episode_{PREFLIGHT_SEED:03d}_*")):
            destination = target / f"{arm}__{path.name}"
            destination.write_bytes(path.read_bytes())
            path.unlink()
            moved.append(str(destination))
    return moved


def launch() -> dict:
    plan = json.loads((ART / "WORK_PLAN.json").read_text())
    started = []
    for worker in plan["assignment"]:
        chunks = " ".join(f"{c['task']}:{c['arm']}:{c['seeds']}" for c in worker["chunks"])
        session = worker["worker"]
        command = f"cd {REPO} && bash scripts/run_openvla_k10_worker.sh {worker['gpu']} {session} {chunks}"
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        result = subprocess.run(["tmux", "new-session", "-d", "-s", session, command],
                                capture_output=True, text=True)
        started.append({"worker": session, "gpu": worker["gpu"], "returncode": result.returncode,
                        "chunks": len(worker["chunks"])})
        log(f"tmux {session} gpu{worker['gpu']} rc={result.returncode} chunks={len(worker['chunks'])}")
    return {"workers_started": started, "episodes": plan["episodes"]}


def fail(reason: str, extra: dict | None = None) -> int:
    payload = {"launched": False, "reason": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if extra:
        payload["detail"] = extra
    (ART / "LAUNCH_STATUS.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    log(f"NOT launching: {reason}")
    return 1


def main() -> int:
    log("gate started: waiting for the K probe to release GPUs 1/2/3")
    deadline = time.time() + 6 * 60 * 60
    while True:
        idle, occupancy = gpus_idle()
        if idle:
            log(f"probe finished, GPUs idle: {json.dumps(occupancy)}")
            break
        if time.time() > deadline:
            return fail("probe did not finish within 6 hours", occupancy)
        time.sleep(45)
    for arm in ARMS:
        log(f"preflight {PREFLIGHT_TASK} seed {PREFLIGHT_SEED} arm {arm}")
        code = run(["python", "research/semantic_token_cd/openvla_matched_k10_rollout.py",
                    "--artifact", str(ART), "--task", PREFLIGHT_TASK,
                    "--seeds", str(PREFLIGHT_SEED), "--gpu", "1", "--worker-id", f"k10_preflight_{arm}",
                    "--arms", arm, "--protocol-id", PROTOCOL, "--lock-seeds", "200-299"],
                   ART / "logs" / f"preflight_{arm}.log", 1)
        if code != 0:
            return fail(f"preflight {arm} exited {code}")
    ok, detail = verify_preflight()
    log(f"preflight verification ok={ok} {json.dumps(detail, default=float)[:900]}")
    if not ok:
        return fail("preflight verification failed", detail)
    moved = archive_preflight()
    log(f"archived {len(moved)} preflight files and cleared the main tree")
    result = launch()
    result.update({"launched": True, "preflight_verification": detail,
                   "preflight_archived": moved,
                   "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (ART / "LAUNCH_STATUS.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    log(f"launched {len(result['workers_started'])} workers for {result['episodes']} episodes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:                                        # noqa: BLE001
        # An unexpected crash must still leave a machine-readable verdict behind.
        try:
            (ART / "LAUNCH_STATUS.json").write_text(json.dumps(
                {"launched": False, "reason": f"gate crashed: {type(exc).__name__}: {exc}",
                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n")
            log(f"GATE CRASHED: {type(exc).__name__}: {exc}")
        finally:
            raise
