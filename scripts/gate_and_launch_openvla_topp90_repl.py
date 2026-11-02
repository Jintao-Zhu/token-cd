#!/usr/bin/env python3
"""Gate + launch for the TopP90-vs-matched replication (lambda=1.0, seeds 100-199).

1. wait for the preflight exit file
2. verify both arms' preflight episode: technical_pass, correct rule, matched/TopP invariants
3. archive the preflight episodes out of the main tree (the [100] block reruns them)
4. launch 12 workers from WORK_PLAN.json
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = REPO / "artifacts/openvla_topp90_repl_100_199_v1"
TASK = "google_robot_close_drawer"
SEED = 100
ARMS = ("matched_l100", "top_p090_l100")
EXIT_FILE = Path("/tmp/ovrep_pre.exit")
LOG = ART / "logs" / "gate.log"


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as stream:
        stream.write(line + "\n")


def verify() -> tuple[bool, dict]:
    detail, ok = {}, True
    for arm in ARMS:
        path = ART / "episodes" / TASK / arm / f"episode_{SEED:03d}_summary.json"
        if not path.is_file():
            detail[arm] = {"error": "missing summary"}
            ok = False
            continue
        data = json.loads(path.read_text())
        trace = data.get("selector_trace") or []
        flags = {k: v for k, v in data.items() if isinstance(v, bool) and k != "success"}
        if arm.startswith("matched"):
            rule_checks = {
                "matched_rule": all(x.get("coverage_mode") == "own_state_matched_budget" for x in trace),
                "matched_union": all(len(set(x.get("reference_shr_token_ids", []))) == x.get("m_t")
                                     for x in trace),
            }
        else:
            rule_checks = {
                "top_p_rule": all(x.get("coverage_mode") == "visual_top_p_0.90_clip_1_256" for x in trace),
                "top_p_threshold": all(abs(float(x.get("top_p_threshold") or -1) - 0.9) < 1e-12
                                       for x in trace),
                "top_p_unclipped": all(int(x.get("selection_min_count", -1)) == 1
                                       and int(x.get("selection_max_count", -1)) == 256 for x in trace),
                "top_p_no_clip": all(not x.get("lower_bound_triggered")
                                     and not x.get("upper_bound_triggered") for x in trace),
                "attn_layer_11": all(x.get("attention_layers") == [11] for x in trace),
            }
        entry = {"success": data.get("success"), "steps": data.get("steps"),
                 "technical_pass": data.get("technical_pass"),
                 "mean_m_t": data.get("mean_selected_token_count"),
                 "budget_rule": data.get("budget_rule"), "lambda": data.get("lambda"),
                 "flags_false": sorted(k for k, v in flags.items() if v is not True),
                 "rule_checks": rule_checks, "steps_total": len(trace)}
        entry["ok"] = (entry["technical_pass"] is True and not entry["flags_false"]
                       and all(rule_checks.values()) and entry["steps_total"] > 0
                       and abs(float(entry["lambda"] or -1) - 1.0) < 1e-12)
        ok = ok and entry["ok"]
        detail[arm] = entry
    return ok, detail


def archive() -> list[str]:
    moved, target = [], ART / "preflight"
    target.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        directory = ART / "episodes" / TASK / arm
        for path in sorted(directory.glob(f"episode_{SEED:03d}_*")):
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
        command = (f"cd {REPO} && bash scripts/run_openvla_topp90_repl_worker.sh "
                   f"{worker['gpu']} {session} {chunks}")
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        result = subprocess.run(["tmux", "new-session", "-d", "-s", session, command],
                                capture_output=True, text=True)
        started.append({"worker": session, "gpu": worker["gpu"], "chunks": len(worker["chunks"]),
                        "returncode": result.returncode})
        log(f"tmux {session} gpu{worker['gpu']} rc={result.returncode}")
    return {"workers_started": started, "episodes": plan["episodes"]}


def main() -> int:
    log("gate started; waiting for preflight")
    deadline = time.time() + 3600
    while not EXIT_FILE.is_file():
        if time.time() > deadline:
            (ART / "LAUNCH_STATUS.json").write_text(json.dumps(
                {"launched": False, "reason": "preflight timeout"}, indent=2) + "\n")
            return 1
        time.sleep(15)
    code = int(EXIT_FILE.read_text().strip() or -1)
    log(f"preflight exit={code}")
    if code != 0:
        (ART / "LAUNCH_STATUS.json").write_text(json.dumps(
            {"launched": False, "reason": f"preflight exit {code}"}, indent=2) + "\n")
        return 1
    ok, detail = verify()
    log(f"verify ok={ok} {json.dumps(detail, default=float)[:700]}")
    if not ok:
        (ART / "LAUNCH_STATUS.json").write_text(json.dumps(
            {"launched": False, "reason": "verification failed", "detail": detail},
            indent=2, default=float) + "\n")
        return 1
    moved = archive()
    log(f"archived {len(moved)} preflight files")
    result = launch()
    result.update({"launched": True, "preflight": detail, "archived": moved,
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
        (ART / "LAUNCH_STATUS.json").write_text(json.dumps(
            {"launched": False, "reason": f"gate crashed: {type(exc).__name__}: {exc}"},
            indent=2) + "\n")
        raise
