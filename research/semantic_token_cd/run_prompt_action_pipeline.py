"""Sequential, resumable Stage A -> Stage B -> closed-loop pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from research.semantic_token_cd.prompt_action_complement_protocol import ARTIFACT, REPO, atomic_json


PYTHONPATH = f"{REPO}:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"


def run_group(name, script, specs):
    logs = ARTIFACT / name / "logs"; logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update({"PYTHONPATH": PYTHONPATH, "HF_HUB_OFFLINE": "1",
                                         "TOKENIZERS_PARALLELISM": "false"})
    active = []
    for index, (task, gpu, shard, nshards) in enumerate(specs):
        log = logs / f"{task.removeprefix('google_robot_')}_shard{shard}of{nshards}_gpu{gpu}.log"
        handle = log.open("a")
        command = [sys.executable, str(REPO / script), "--task", task, "--gpu", str(gpu),
                   "--shard-index", str(shard), "--num-shards", str(nshards)]
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        active.append((process, handle, log)); print(json.dumps({"stage": name, "started": command}), flush=True)
    failed = []
    for process, handle, log in active:
        code = process.wait(); handle.close()
        if code: failed.append((code, str(log)))
    if failed: raise RuntimeError(f"{name} worker failures: {failed}")


def main():
    preflight = json.loads((ARTIFACT / "preflight/report.json").read_text())
    if not preflight["audit"]["technical_pass"]: raise RuntimeError("preflight did not pass")
    specs = [
        ("google_robot_open_drawer", 2, 0, 1),
        ("google_robot_pick_coke_can", 2, 0, 2),
        ("google_robot_move_near", 2, 0, 2),
        ("google_robot_close_drawer", 3, 0, 1),
        ("google_robot_pick_coke_can", 3, 1, 2),
        ("google_robot_move_near", 3, 1, 2),
    ]
    if not (ARTIFACT / "stage_a/COMPLETE").exists():
        run_group("stage_a", "research/semantic_token_cd/prompt_action_stage_a.py", specs)
        subprocess.run([sys.executable, str(REPO / "research/semantic_token_cd/analyze_prompt_action_stages.py"), "a"],
                       cwd=REPO, env={**os.environ, "PYTHONPATH": PYTHONPATH}, check=True)
    if not (ARTIFACT / "stage_b/COMPLETE").exists():
        run_group("stage_b", "research/semantic_token_cd/prompt_action_stage_b.py", specs)
        subprocess.run([sys.executable, str(REPO / "research/semantic_token_cd/analyze_prompt_action_stages.py"), "b"],
                       cwd=REPO, env={**os.environ, "PYTHONPATH": PYTHONPATH}, check=True)
    if not (ARTIFACT / "closed_loop/COMPLETE").exists():
        subprocess.run([sys.executable, str(REPO / "research/semantic_token_cd/prompt_action_complement_coordinator.py")],
                       cwd=REPO, env={**os.environ, "PYTHONPATH": PYTHONPATH}, check=True)
    subprocess.run([sys.executable, str(REPO / "research/semantic_token_cd/analyze_prompt_action_closed.py")],
                   cwd=REPO, env={**os.environ, "PYTHONPATH": PYTHONPATH}, check=True)
    (ARTIFACT / "COMPLETE").write_text("all stages and final report complete\n")


if __name__ == "__main__": main()
