from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from datetime import datetime
from pathlib import Path

from research.coreact_w1_slg_rollout.sampler import ACTIVE_STEPS, ARMS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=False)
    for name in ("episodes", "invalid_pairs", "logs", "status"):
        (artifact / name).mkdir()
    offline = workspace / "artifacts/coreact_slg_self_weak_low_noise_v1_20260816_234129"
    checkpoint = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model"
    lock = json.loads((offline / "selected_self_weak.lock.json").read_text())
    if lock["candidate"] != "W1_skip_last_1" or lock["low_steps"] != [8, 9] or lock["high_steps"] != [0, 1]:
        raise RuntimeError("offline selection lock does not match preregistered rollout")
    protocol = {
        "experiment": "W1 Low-Noise SLG Closed-Loop Success Validation",
        "created_at": datetime.now().astimezone().isoformat(),
        "scientific_question": "Does confirmed low-noise W1 Strong-Weak refinement improve closed-loop task success?",
        "strong": {"checkpoint": str(checkpoint), "expert_layers": 16, "checkpoint_model_sha256": sha256(checkpoint / "model.safetensors")},
        "weak": {"operator": "skip only final action-expert block", "scale_contract": [1.0] * 15 + [0.0], "shared_checkpoint": True},
        "guidance": {"lambda": 0.5, "trust_region_kappa": 0.25, "low_steps": [8, 9], "full_steps": list(range(10)), "high_steps": [0, 1]},
        "arms": list(ARMS),
        "rollout": {"suite": "libero_spatial", "tasks": list(range(10)), "init_state_ids": list(range(50)), "paired_states": 500, "episodes": 2000, "maximum_control_steps": 520, "chunk_size": 50, "executed_actions": 10, "flow_steps": 10},
        "seed_rules": {"reset": "860000000 + task*1000 + init*10", "flow_noise_base": "870000000 + task*1000 + init*10; add replan index"},
        "offline_lock": {"path": str(offline / "selected_self_weak.lock.json"), "sha256": sha256(offline / "selected_self_weak.lock.json"), "decision": "SLG_SELF_WEAK_LOW_NOISE_COMPATIBILITY_CONFIRMED"},
        "interpretation": {"primary": "low_w1 - strong", "go_pp": 5.0, "ci_lower_strictly_positive": True, "rescue_gt_harm": True, "tasks_nonworse": 7, "secondary": ["low_w1 > full_w1", "low_w1 > high_w1"]},
        "prohibited": ["lambda tuning", "window tuning", "per-task settings", "dimension gating", "W2 rescue", "entropy gating", "CFG"],
        "environment": {"host": socket.gethostname(), "platform": platform.platform()},
    }
    (artifact / "protocol.lock.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    rows = []
    for task in range(10):
        for init in range(50):
            pair = f"task{task:02d}__init{init:02d}"
            reset_seed = 860_000_000 + task * 1000 + init * 10
            noise_seed = 870_000_000 + task * 1000 + init * 10
            for arm in ARMS:
                rows.append({
                    "episode_id": f"{pair}__{arm}", "pair_id": pair, "suite": "libero_spatial",
                    "task_id": task, "init_state_id": init, "arm": arm,
                    "reset_seed": reset_seed, "flow_noise_seed_base": noise_seed,
                    "active_flow_steps": sorted(ACTIVE_STEPS[arm]),
                })
    with (artifact / "episode_manifest.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (artifact / "manifest.lock.sha256").write_text(sha256(artifact / "episode_manifest.jsonl") + "  episode_manifest.jsonl\n")
    (artifact / "status.json").write_text(json.dumps({"status": "PREPARED_DRY_RUN_PENDING", "episodes_complete": 0, "episodes_planned": 2000}, indent=2) + "\n")
    print(json.dumps({"artifact": str(artifact), "pairs": 500, "episodes": 2000}, indent=2))


if __name__ == "__main__":
    main()
