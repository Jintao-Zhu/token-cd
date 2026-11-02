#!/usr/bin/env python3
"""Resume failed optimized Apple/WidowX shards and finalize nine-task report."""
from __future__ import annotations

import json
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/leju-suzhou/zjt_ws")
TOKEN_CD = ROOT / "token-cd"
RUN = TOKEN_CD / "runs/pi0fast_jit_header_extra5_100seed_rtfull_v1"
TASKS = (
    "google_robot_place_apple_in_closed_top_drawer", "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube",
)
FAMILIES = ("google", "widowx", "widowx", "widowx", "widowx")
GPUS = (5, 5, 6, 6, 7)
PORTS = (19051, 19052, 19061, 19062, 19071)
SHARDS = ((0, 33), (34, 66), (67, 99))


def alive(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def start(session: str, command: str) -> None:
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)


def listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def main() -> None:
    restarts: dict[str, int] = {}
    while True:
        complete = 0
        stuck = []
        for i, task in enumerate(TASKS):
            for shard, (lo, hi) in enumerate(SHARDS):
                paths = [RUN / "episodes" / task / f"l6_matched_050_seed{seed:03d}.json"
                         for seed in range(lo, hi + 1)]
                count = 0
                for path in paths:
                    if path.is_file():
                        try:
                            with path.open() as file:
                                count += '"status": "complete"' in file.read(100)
                        except OSError:
                            pass
                complete += count
                if count == len(paths):
                    continue
                srv = f"pi0hdr9_srv_{i}"
                wrk = f"pi0hdr9_w_{i}_{shard}"
                if not alive(srv):
                    restarts[srv] = restarts.get(srv, 0) + 1
                    if restarts[srv] <= 2:
                        command = (f"bash {TOKEN_CD}/scripts/run_pi0fast_jit_extra_server.sh "
                                   f"{GPUS[i]} {PORTS[i]} {FAMILIES[i]}")
                        start(srv, command)
                    else:
                        stuck.append(srv)
                    continue
                ready = listening(PORTS[i])
                if ready and not alive(wrk):
                    restarts[wrk] = restarts.get(wrk, 0) + 1
                    if restarts[wrk] <= 2:
                        command = (f"bash {TOKEN_CD}/scripts/run_pi0fast_jit_extra_worker.sh "
                                   f"{GPUS[i]} {PORTS[i]} {task} {FAMILIES[i]} {lo} {hi}")
                        start(wrk, command)
                    else:
                        stuck.append(wrk)
        status = {"timestamp": datetime.now().isoformat(), "completed": complete,
                  "planned": 500, "unrecoverable_sessions": stuck, "restarts": restarts}
        (RUN / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        if stuck:
            (RUN / "NEEDS_ATTENTION").write_text(json.dumps(status, indent=2) + "\n")
            raise SystemExit(f"run needs attention: {stuck}")
        if complete == 500:
            subprocess.run(["python3", str(TOKEN_CD / "scripts/report_pi0fast_jit_9task.py")], check=True)
            for i in range(5):
                srv = f"pi0hdr9_srv_{i}"
                if alive(srv):
                    subprocess.run(["tmux", "kill-session", "-t", srv], check=True)
            (RUN / "COMPLETE").write_text(datetime.now().isoformat() + "\n")
            print("all 500 extra optimized episodes finished and services stopped", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
