#!/usr/bin/env python3
"""Overnight supervisor: keep both experiments alive, analyse when complete, write a morning report."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

WS = Path("/home/leju-suzhou/zjt_ws/token-cd")
PY = Path("/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python")
SC_ROOT = WS / "artifacts/l11_matched_state_coupling_v1"
PA_ROOT = WS / "artifacts/l11_budget_response_curve_v1"
LOG = WS / "artifacts/NIGHTLY_SUPERVISOR.log"
REPORT = WS / "artifacts/NIGHTLY_REPORT.md"
POLL = 300


def say(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def sh(cmd: list[str]) -> int:
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def count(pattern: str) -> int:
    return len(list(Path("/").glob(pattern.lstrip("/"))))


def running(pat: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return bool(r.stdout.strip())


def run(cmd: list[str], logname: str) -> int:
    with (WS / "artifacts" / logname).open("a") as fh:
        fh.write(f"\n===== {datetime.now():%H:%M:%S} {' '.join(cmd)} =====\n")
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, cwd=str(WS))
    return p.returncode


def tmux_has(name: str) -> bool:
    r = subprocess.run(["tmux", "has-session", "-t", name], capture_output=True, text=True)
    return r.returncode == 0


def main() -> None:
    say("supervisor start")
    sc_done = pa_done = False
    sc_analyzed = pa_analyzed = False
    restarts = {"sc": 0, "pa": 0}

    while not (sc_done and pa_done):
        # ---------- state coupling ----------
        n_sc = len(list((SC_ROOT / "episodes").glob("*/*/episode_*_summary.json")))
        if n_sc >= 400:
            if not sc_analyzed:
                say(f"state-coupling complete ({n_sc}/400) -> analysing")
                rc = run([str(PY), "research/semantic_token_cd/analyze_l11_matched_state_coupling.py",
                          "--artifact", str(SC_ROOT), "--matched-artifact",
                          str(WS / "artifacts/prompt_attn_l11_token_count_v1")], "sc_analysis.log")
                say(f"state-coupling analysis rc={rc}")
                sc_analyzed = True
            sc_done = True
        else:
            if not running("matched_shuffle_rollout") and not tmux_has("statecoupling"):
                restarts["sc"] += 1
                if restarts["sc"] <= 6:
                    say(f"state-coupling stalled at {n_sc}/400 -> restart #{restarts['sc']}")
                    tmux_leave = subprocess.run(
                        ["tmux", "new-session", "-d", "-s", "statecoupling", "-c", str(WS),
                         "bash research/semantic_token_cd/orchestrate_l11_matched_state_coupling_2pergpu.sh; exec bash"],
                        capture_output=True, text=True)
                    say(f"tmux restart rc={tmux_leave.returncode}")
                else:
                    say("state-coupling restart budget exhausted")
                    sc_done = True

        # ---------- phase A ----------
        n_pa = len(list((PA_ROOT / "phase_a").glob("*/seed_*.json")))
        if n_pa >= 200:
            if not pa_analyzed:
                say(f"Phase A complete ({n_pa}/200 seeds) -> analysing")
                rc = run([str(PY), "research/semantic_token_cd/analyze_phase_a_budget_response.py",
                          "--artifact", str(PA_ROOT)], "pa_analysis.log")
                say(f"Phase A analysis rc={rc}")
                pa_analyzed = True
            pa_done = True
        else:
            if not running("phase_a_l11_budget_response_scan") and not tmux_has("phaseA"):
                restarts["pa"] += 1
                if restarts["pa"] <= 6:
                    say(f"Phase A stalled at {n_pa}/200 -> restart #{restarts['pa']}")
                    subprocess.run(
                        ["tmux", "new-session", "-d", "-s", "phaseA", "-c", str(WS),
                         "bash research/semantic_token_cd/orchestrate_phase_a.sh; exec bash"],
                        capture_output=True, text=True)
                else:
                    say("Phase A restart budget exhausted")
                    pa_done = True

        if not (sc_done and pa_done):
            time.sleep(POLL)

    # ---------- morning report ----------
    parts = [f"# 夜间实验报告  {datetime.now():%Y-%m-%d %H:%M}\n"]
    for title, p in (("实验一：Matched 状态耦合消融",
                      SC_ROOT / "STATE_COUPLING_REPORT.md"),
                     ("实验二：Phase A 有效反事实预算",
                      PA_ROOT / "PHASE_A_REPORT.md")):
        parts.append(f"\n---\n\n## {title}\n")
        parts.append(p.read_text() if p.exists() else "_(报告未生成)_\n")
    parts.append(f"\n---\n\nsupervisor 结束于 {datetime.now():%Y-%m-%d %H:%M:%S}；"
                 f"重启次数 sc={restarts['sc']} pa={restarts['pa']}\n")
    REPORT.write_text("".join(parts))
    say(f"morning report written -> {REPORT}")


if __name__ == "__main__":
    main()
