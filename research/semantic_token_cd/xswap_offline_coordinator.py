"""Run xswap_offline_decode for all four tasks over GPU 2,3 (one job per task)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from research.semantic_token_cd.xswap_protocol import ARTIFACT, TASKS

GPUS = (2, 3)


def main() -> None:
    repo = Path("/home/leju-suzhou/zjt_ws/token-cd")
    logs = Path("/home/leju-suzhou/zjt_ws/token-cd/logs/xswap/offline")
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": f"{repo}:{pcd}:{env.get('PYTHONPATH', '')}"})
    # assign first two tasks to GPU2, rest to GPU3; then re-run any failures.
    assignment = {task: GPUS[i % 2] for i, task in enumerate(TASKS)}
    processes = []
    for task in TASKS:
        if (ARTIFACT / "runs/offline" / task / "offline_summary.json").exists():
            print(json.dumps({"offline_skip_done": task}), flush=True)
            continue
        gpu = assignment[task]
        log = logs / f"{task.removeprefix('google_robot_')}_gpu{gpu}.log"
        handle = log.open("a")
        cmd = [sys.executable, str(repo / "research/semantic_token_cd/xswap_offline_decode.py"),
               "--task", task, "--gpu", str(gpu), "--worker-id", f"offline_{task}"]
        processes.append((task, subprocess.Popen(cmd, cwd=repo, env=env, stdout=handle,
                                                 stderr=subprocess.STDOUT, start_new_session=True), log))
        print(json.dumps({"offline_started": task, "gpu": gpu}), flush=True)
        time.sleep(2)
    failed = []
    for task, proc, log in processes:
        rc = proc.wait()
        print(json.dumps({"offline_finished": task, "rc": rc}), flush=True)
        if rc != 0:
            failed.append(task)
    # retry failures on the other GPU once
    for task in failed:
        gpu = 2 if assignment[task] == 3 else 3
        log = logs / f"{task.removeprefix('google_robot_')}_retry_gpu{gpu}.log"
        handle = log.open("a")
        cmd = [sys.executable, str(repo / "research/semantic_token_cd/xswap_offline_decode.py"),
               "--task", task, "--gpu", str(gpu), "--worker-id", f"offline_retry_{task}"]
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
        rc = proc.wait()
        print(json.dumps({"offline_retry_finished": task, "rc": rc}), flush=True)
    print("OFFLINE_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
