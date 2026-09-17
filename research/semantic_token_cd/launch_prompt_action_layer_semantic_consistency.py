#!/usr/bin/env python3
"""Run semantic invariance/selectivity scan on GPUs 2 and 3."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_action_layer_semantic_consistency_v1"
WORKER = REPO / "research/semantic_token_cd/prompt_action_layer_semantic_consistency_worker.py"
ANALYZER = REPO / "research/semantic_token_cd/analyze_prompt_action_layer_semantic_consistency.py"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_id": "PROMPT_ACTION_LAYER_SEMANTIC_CONSISTENCY_V1",
        "date": "2026-09-15", "expected_states": 90, "gpus": [2, 3],
        "manual_target_boxes_used": False,
        "question": "same-semantics paraphrases should preserve intervention residual; changed target should alter it",
        "split": "20 exploration episodes / 10 frozen validation episodes",
        "mask_budget": "cached matched m_t", "reconstruction": "harmonic beta=0",
    }
    path = ROOT / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("CONFIG_LOCK differs")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    env = os.environ.copy(); pcd = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    env.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": f"{REPO}:{pcd}:{env.get('PYTHONPATH','')}"})
    processes, logs = [], []
    for shard, gpu in enumerate((2, 3)):
        log = (ROOT / f"worker_gpu{gpu}.log").open("a"); logs.append(log)
        processes.append(subprocess.Popen([
            sys.executable, str(WORKER), "--artifact", str(ROOT), "--gpu", str(gpu),
            "--shard-index", str(shard), "--num-shards", "2",
        ], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for log in logs: log.close()
    if any(code != 0 for code in codes): raise SystemExit(f"workers failed: {codes}")
    subprocess.run([sys.executable, str(ANALYZER)], cwd=REPO, check=True)


if __name__ == "__main__": main()
