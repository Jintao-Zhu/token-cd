from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=False)
    for name in ("snapshots", "audits", "logs", "alternatives", "screen", "confirmation", "invalid_units"):
        (artifact / name).mkdir()
    checkpoint = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model"
    protocol = {
        "experiment": "Causal Hard-Failure Data Existence Gate",
        "created_at": datetime.now().astimezone().isoformat(),
        "suite": "libero_spatial",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
        "collection": {"tasks": list(range(10)), "init_state_ids": list(range(50)), "rollouts": 500, "horizon": 520, "progress_points": [0.25, 0.50, 0.75, 0.90]},
        "native_candidates": {"count": 8, "noise_seed_rule": "fixed pre-registered per snapshot"},
        "expert_candidates": {"count": 4, "same_task": True, "non_same_trajectory": True, "retrieval": "physical-state distance"},
        "screening": {"downstream_seeds": 1, "candidate_selection": "minimum executed first10 normalized distance among rescues"},
        "confirmation": {"fresh_downstream_seeds": 5, "p_bad_max": 0.4, "p_rescue_min": 0.8, "delta_u_min": 0.4, "hard_r_max": 1.0},
        "forbidden": ["failure weak training", "guidance lambda tuning", "outcome-based retrieval", "extra rollout collection", "task-specific thresholds"],
    }
    (artifact / "protocol.lock.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (artifact / "status.json").write_text(json.dumps({"status": "STRONG_FAILURE_COLLECTION_RUNNING"}, indent=2) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
