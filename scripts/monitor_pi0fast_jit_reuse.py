#!/usr/bin/env python3
"""Keep the four-task optimized run resumable and finalize it when complete."""
from __future__ import annotations

import json
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/leju-suzhou/zjt_ws")
TOKEN_CD = ROOT / "token-cd"
RUN = TOKEN_CD / "runs/pi0fast_jit_header_google4_100seed_rtfull_v1"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
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
    restart_count: dict[str, int] = {}
    while True:
        completed = 0
        missing = []
        for i, task in enumerate(TASKS):
            gpu = i + 1
            for shard, (start_seed, end_seed) in enumerate(SHARDS):
                files = [RUN / "comparisons" / task / f"seed{s:03d}.json"
                         for s in range(start_seed, end_seed + 1)]
                # The comparison JSON is written only after the optimized episode
                # completes; unlike episode JSON, it has no status field.
                count = sum(path.is_file() for path in files)
                completed += count
                if count == len(files):
                    continue
                port = 18761 + i + 100 * shard
                server = f"pi0hdr100_srv_{i}" if shard == 0 else f"pi0hdr100_srv_{i}_{shard}"
                worker = f"pi0hdr100_w_{i}_{shard}"
                if not alive(server):
                    key = f"server_{i}_{shard}"
                    restart_count[key] = restart_count.get(key, 0) + 1
                    if restart_count[key] <= 2:
                        command = (f"env PI0FAST_ARTIFACT_DIR={RUN} PI0FAST_TIMING=1 "
                                   f"PI0FAST_CD_JIT_HEADER=1 bash {TOKEN_CD}/scripts/run_pi0fast_l6_server.sh "
                                   f"{gpu} {port} jit100_{shard}")
                        start(server, command)
                    else:
                        missing.append(server)
                    continue
                if not alive(worker) and listening(port):
                    key = f"worker_{i}_{shard}"
                    restart_count[key] = restart_count.get(key, 0) + 1
                    if restart_count[key] <= 2:
                        command = (f"bash {TOKEN_CD}/scripts/run_pi0fast_jit_header_reuse_worker.sh "
                                   f"{gpu} {task} {port} {start_seed} {end_seed} {i}_{shard}")
                        start(worker, command)
                    else:
                        missing.append(worker)
        status = {
            "timestamp": datetime.now().isoformat(),
            "completed": completed,
            "planned": 400,
            "unrecoverable_sessions": missing,
            "restarts": restart_count,
        }
        (RUN / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        if missing:
            (RUN / "NEEDS_ATTENTION").write_text(json.dumps(status, indent=2) + "\n")
            raise SystemExit(f"run needs attention: {missing}")
        if completed == 400:
            subprocess.run(["python3", str(TOKEN_CD / "scripts/report_pi0fast_jit_reuse.py")], check=True)
            subprocess.run(["python3", str(TOKEN_CD / "scripts/report_pi0fast_jit_9task.py")], check=True)
            for i in range(4):
                for shard in range(3):
                    session = f"pi0hdr100_srv_{i}" if shard == 0 else f"pi0hdr100_srv_{i}_{shard}"
                    if alive(session):
                        subprocess.run(["tmux", "kill-session", "-t", session], check=True)
            (RUN / "COMPLETE").write_text(datetime.now().isoformat() + "\n")
            print("all 400 optimized episodes finished and services stopped", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
