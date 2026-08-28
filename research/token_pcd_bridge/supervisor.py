from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from research.token_pcd_stage_a.core import TASKS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--protocol-id", default="TOKEN_PCD_BRIDGE_CLOSED_LOOP_PILOT_R1")
    args = parser.parse_args(); workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    pcd_root = workspace / "official-reproductions/pcd_openvla_simpler_box_31b027e"
    python = pcd_root / "env/venv/bin/python"; log_dir = artifact / "logs"; log_dir.mkdir(exist_ok=True)
    status_path = artifact / "status.json"
    for index, task in enumerate(TASKS):
        completed = artifact / "episodes" / task / "pairing_manifest.json"
        if completed.exists(): continue
        write_json(status_path, {"status": "RUNNING", "task": task, "task_index": index,
                                 "completed_tasks": index, "updated_at": datetime.now().astimezone().isoformat()})
        command = [str(python), "-m", "research.token_pcd_bridge.runner", "--pcd-root", str(pcd_root),
                   "--artifact", str(artifact), "--task", task,
                   "--seeds", ",".join(map(str, range(args.seed_start, args.seed_start + args.seed_count))),
                   "--protocol-id", args.protocol_id]
        with (log_dir / f"{task}.log").open("x") as handle:
            subprocess.run(command, cwd=workspace, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(python), "-m", "research.token_pcd_bridge.analyze", "--artifact", str(artifact),
                    "--seed-start", str(args.seed_start), "--seed-count", str(args.seed_count)],
                   cwd=workspace, env=os.environ.copy(), check=True,
                   stdout=(log_dir / "analysis.log").open("x"), stderr=subprocess.STDOUT)
    decision = json.loads((artifact / "bridge_decision.json").read_text())
    write_json(status_path, {"status": "COMPLETE", "decision": decision["status"], "completed_tasks": 9,
                             "updated_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__": main()
