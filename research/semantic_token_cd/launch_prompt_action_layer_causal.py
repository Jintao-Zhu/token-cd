#!/usr/bin/env python3
"""Launch the box-free action-level layer screen on physical GPUs 2 and 3."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "artifacts/prompt_action_layer_causal_v1"
WORKER = REPO / "research/semantic_token_cd/prompt_action_layer_causal_worker.py"
ANALYZE = REPO / "research/semantic_token_cd/analyze_prompt_action_layer_causal.py"


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_id": "PROMPT_ACTION_LAYER_CAUSAL_V1",
        "date": "2026-09-15",
        "purpose": "box-free action-level screen of OpenVLA layers 0..31",
        "expected_states": 9,
        "states": "9 existing same-image target switches; 6 exploration / 3 frozen validation",
        "manual_target_boxes_used": False,
        "mask_budget": "cached own-state matched m_t",
        "reconstruction": "16x16 four-neighbor harmonic beta=0",
        "prompt_direction": "shared-original-action-prefix log-softmax residual between original and changed-target Prompt",
        "intervention_direction": "shared-prefix log-softmax residual between clean and reconstructed negative branch",
        "primary_metric": "mean cosine over action dimensions 0..5",
        "primary_ranking": "exploration mean causal margin = Top-m alignment - max(Random, Bottom, wrong-Prompt Top-m)",
        "validation": "candidate weights and ranking frozen before reading 3 validation states",
        "gpus": [2, 3],
    }
    config_path = ARTIFACT / "CONFIG_LOCK.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("existing CONFIG_LOCK differs")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    processes = []
    logs = []
    environment = os.environ.copy()
    pcd_source = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{REPO}:{pcd_source}:{environment.get('PYTHONPATH', '')}",
    })
    for shard, gpu in enumerate((2, 3)):
        log_path = ARTIFACT / f"worker_gpu{gpu}.log"
        log = log_path.open("a")
        command = [
            sys.executable,
            str(WORKER),
            "--artifact", str(ARTIFACT),
            "--gpu", str(gpu),
            "--shard-index", str(shard),
            "--num-shards", "2",
            "--only-original-controls",
        ]
        processes.append(subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT
        ))
        logs.append(log)
    return_codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    if any(code != 0 for code in return_codes):
        raise SystemExit(f"workers failed: {return_codes}")
    subprocess.run([sys.executable, str(ANALYZE), "--artifact", str(ARTIFACT)], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
