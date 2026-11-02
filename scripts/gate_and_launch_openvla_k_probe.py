#!/usr/bin/env python3
"""Gate the K-probe launch on the preflight, then start the 12 workers.

Run detached. Steps:
  1. wait for the preflight exit code (written by the worker shell script)
  2. verify the preflight probe file: per-step m_k8 must equal the historical matched
     reference for the same task/seed, and the ladder must be complete
  3. only then start one tmux session per worker, using WORK_PLAN.json
Status is appended to artifacts/openvla_k_probe_200_219_v1/logs/gate.log and, at the end,
written to LAUNCH_STATUS.json. It never kills anything.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = REPO / "artifacts/openvla_k_probe_200_219_v1"
REFERENCE = REPO / "artifacts/openvla_lambda1_extension_200_299_v1"
EXIT_FILE = Path("/tmp/ovkp_preflight.exit")
TASK = "google_robot_close_drawer"
SEED = 200
KS = (8, 9, 10, 11, 12, 14, 16, 20)
LOG = ART / "logs" / "gate.log"


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as stream:
        stream.write(line + "\n")


def verify() -> tuple[bool, dict]:
    probe_path = ART / "episodes" / TASK / "lambda_100" / f"episode_{SEED:03d}_probe.json"
    reference_path = REFERENCE / "episodes" / TASK / "lambda_100" / f"episode_{SEED:03d}_summary.json"
    detail: dict = {"probe": str(probe_path), "reference": str(reference_path)}
    if not probe_path.is_file() or not reference_path.is_file():
        detail["error"] = "missing probe or reference file"
        return False, detail
    probe = json.loads(probe_path.read_text())
    reference = json.loads(reference_path.read_text())
    reference_m = [int(row.get("m_t", -1)) for row in reference["selector_trace"]]
    per_step = probe.get("per_step", [])
    detail["steps_reference"] = len(reference_m)
    detail["steps_probe"] = len(per_step)
    if len(per_step) != len(reference_m):
        detail["error"] = "step count mismatch"
        return False, detail
    probe_m = [row.get("m_k8") for row in per_step]
    detail["k8_exact_steps"] = sum(1 for a, b in zip(reference_m, probe_m) if a == b)
    detail["k8_mismatch_steps"] = [i for i, (a, b) in enumerate(zip(reference_m, probe_m)) if a != b][:10]
    missing = [k for k in KS if not any(row.get(f"m_k{k}") is not None for row in per_step)]
    detail["missing_ks"] = missing
    detail["ladder_means"] = {f"k{k}": probe["summary"][f"m_k{k}"]["mean"] for k in KS}
    ok = (not detail["k8_mismatch_steps"] and not missing
          and probe_m == reference_m and detail["steps_probe"] > 0)
    return ok, detail


def launch() -> dict:
    plan = json.loads((ART / "WORK_PLAN.json").read_text())
    started = []
    for worker in plan["assignment"]:
        chunks = " ".join(f"{c['task']}:{c['seeds']}" for c in worker["chunks"])
        session = worker["worker"]
        command = (f"cd {REPO} && bash scripts/run_openvla_k_probe_worker.sh "
                   f"{worker['gpu']} {session} {chunks}")
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        result = subprocess.run(["tmux", "new-session", "-d", "-s", session, command],
                                capture_output=True, text=True)
        started.append({"worker": session, "gpu": worker["gpu"],
                        "chunks": [f"{c['task']}:{c['seeds']}" for c in worker["chunks"]],
                        "returncode": result.returncode, "stderr": result.stderr.strip()})
        log(f"tmux {session} on gpu{worker['gpu']} rc={result.returncode} "
            f"chunks={len(worker['chunks'])}")
    return {"workers_started": started, "episodes": plan["episodes"], "chunks": plan["chunks"]}


def main() -> int:
    log("gate started; waiting for preflight exit file")
    deadline = time.time() + 60 * 60
    while not EXIT_FILE.is_file():
        if time.time() > deadline:
            log("preflight did not finish within 60 minutes; NOT launching")
            (ART / "LAUNCH_STATUS.json").write_text(json.dumps(
                {"launched": False, "reason": "preflight timeout"}, indent=2) + "\n")
            return 1
        time.sleep(15)
    code = int(EXIT_FILE.read_text().strip() or -1)
    log(f"preflight exit code = {code}")
    if code != 0:
        log("preflight failed; NOT launching")
        (ART / "LAUNCH_STATUS.json").write_text(json.dumps(
            {"launched": False, "reason": f"preflight exit {code}"}, indent=2) + "\n")
        return 1
    ok, detail = verify()
    log(f"preflight verification ok={ok} {json.dumps(detail)[:600]}")
    if not ok:
        log("preflight verification failed; NOT launching")
        (ART / "LAUNCH_STATUS.json").write_text(json.dumps(
            {"launched": False, "reason": "verification failed", "detail": detail}, indent=2) + "\n")
        return 1
    result = launch()
    result.update({"launched": True, "preflight_verification": detail,
                   "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (ART / "LAUNCH_STATUS.json").write_text(json.dumps(result, indent=2) + "\n")
    log(f"launched {len(result['workers_started'])} workers for {result['episodes']} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
