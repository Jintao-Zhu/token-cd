#!/usr/bin/env python3
"""Launch budget robustness workers on GPUs 2 and 3."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_action_layer_budget_robustness_v2"
WORKER = REPO / "research/semantic_token_cd/prompt_action_layer_budget_robustness_worker.py"
ANALYZER = REPO / "research/semantic_token_cd/analyze_prompt_action_layer_budget_robustness.py"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_id": "PROMPT_ACTION_LAYER_BUDGET_ROBUSTNESS_V2",
        "date": "2026-09-15",
        "expected_states": 90,
        "layers": [9, 11, 14],
        "budget_scales": [0.75, 1.0, 1.25],
        "gpus": [2, 3],
        "processes_per_gpu": 1,
        "random_control": "one deterministic statewise permutation; nested prefixes across scales",
        "split": "20 exploration episodes / 10 frozen validation episodes",
        "manual_target_boxes_used": False,
        "reconstruction": "harmonic beta=0",
    }
    lock = ROOT / "CONFIG_LOCK.json"
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise RuntimeError("CONFIG_LOCK differs")
    lock.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    env = os.environ.copy()
    pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{REPO}:{pcd}:{env.get('PYTHONPATH', '')}",
    })
    processes = []
    logs = []
    for shard, gpu in enumerate((2, 3)):
        log = (ROOT / f"worker_gpu{gpu}.log").open("a")
        logs.append(log)
        processes.append(subprocess.Popen([
            sys.executable, str(WORKER), "--artifact", str(ROOT), "--gpu", str(gpu),
            "--shard-index", str(shard), "--num-shards", "2",
        ], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    if any(code != 0 for code in codes):
        raise SystemExit(f"workers failed: {codes}")
    subprocess.run([sys.executable, str(ANALYZER)], cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
