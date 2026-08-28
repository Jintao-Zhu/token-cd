from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from research.coreact_w1_slg_rollout.sampler import ACTIVE_STEPS


ARMS = ("lambda_0", "lambda_005", "lambda_010", "lambda_025", "lambda_050")
DOSES = {"lambda_0": 0.0, "lambda_005": 0.05, "lambda_010": 0.1, "lambda_025": 0.25, "lambda_050": 0.5}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, required=True); p.add_argument("--artifact", type=Path, required=True)
    a = p.parse_args(); w, out = a.workspace.resolve(), a.artifact.resolve(); out.mkdir(parents=True, exist_ok=False)
    for name in ("episodes", "invalid_pairs", "logs", "status", "regression"):
        (out / name).mkdir()
    checkpoint = w / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model"
    offline = w / "artifacts/coreact_slg_self_weak_low_noise_v1_20260816_234129/selected_self_weak.lock.json"
    protocol = {
        "experiment": "W1 Low-Noise Guidance Dose-Response Closed-Loop Test",
        "created_at": datetime.now().astimezone().isoformat(),
        "strong_checkpoint": str(checkpoint), "weak": "W1 skip final action-expert block",
        "window": {"low_steps": [8, 9], "scheduler": "10 steps, tau=1.0..0.1", "trust_region_kappa": 0.25},
        "doses": DOSES, "arms": list(ARMS), "lambda_050_role": "known high-dose control, rerun in this artifact",
        "rollout": {"suite": "libero_spatial", "tasks": list(range(10)), "init_state_ids": list(range(50)), "pairs": 500, "episodes": 2500, "horizon": 520, "chunk": 50, "execute": 10},
        "splits": {"selection": "init_state_id 0..24 per task", "confirmation": "init_state_id 25..49 per task", "locked_before_new_dose_rollouts": True},
        "old_artifact_not_reused": True, "offline_lock_sha256": sha256(offline),
        "prohibited": ["dose additions", "task-specific dose", "dynamic dose", "dimension gating", "window changes", "CFG", "one-shot"],
    }
    (out / "protocol.lock.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    rows = []
    for task in range(10):
        for init in range(50):
            pair = f"task{task:02d}__init{init:02d}"; reset = 880_000_000 + task * 1000 + init * 10; noise = 890_000_000 + task * 1000 + init * 10
            for arm in ARMS:
                rows.append({"episode_id": f"{pair}__{arm}", "pair_id": pair, "arm": arm, "task_id": task, "init_state_id": init, "split": "selection" if init < 25 else "confirmation", "reset_seed": reset, "flow_noise_seed_base": noise, "lambda": DOSES[arm], "active_flow_steps": [8, 9] if DOSES[arm] else []})
    with (out / "episode_manifest.jsonl").open("x") as f:
        f.write("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    (out / "manifest.sha256").write_text(sha256(out / "episode_manifest.jsonl") + "  episode_manifest.jsonl\n")
    (out / "status.json").write_text(json.dumps({"status": "PREPARED_REGRESSION_PENDING", "episodes_complete": 0, "episodes_planned": 2500}, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "pairs": 500, "episodes": 2500, "selection": 250, "confirmation": 250}, indent=2))


if __name__ == "__main__": main()
