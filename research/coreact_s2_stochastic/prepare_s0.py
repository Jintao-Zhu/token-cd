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
    for name in ("snapshots", "audits", "logs", "s0", "invalid_units"):
        (artifact / name).mkdir()
    checkpoint = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model"
    protocol = {
        "experiment": "S2-Style Stochastic Weak Guidance for Flow-VLA",
        "created_at": datetime.now().astimezone().isoformat(),
        "suite": "libero_spatial",
        "strong_checkpoint": str(checkpoint),
        "strong_model_sha256": sha(checkpoint / "model.safetensors"),
        "weak_candidates": {"S1": {"drop_count": 1, "eligible_blocks": list(range(1, 16))}, "S2": {"drop_count": 2, "eligible_blocks": list(range(1, 16)), "without_replacement": True}},
        "flow": {"num_steps": 10, "active_steps": list(range(1, 9)), "boundary_steps": [0, 9], "order": "tau 1.0 to 0.1", "omega": 0.25, "trust_region_kappa": 0.25},
        "s0": {"tasks": list(range(10)), "init_state_ids": list(range(50, 55)), "progress": [0.30, 0.65], "states": 100, "masks_per_state_timestep": 12, "mask_seed_rule": "202608180000 + state_ordinal*10000 + timestep*100 + mask_index"},
        "prohibited": ["training", "rollout before S0 gate", "omega sweep", "timestep sweep", "outcome-based mask selection", "expert-manifold GO gate"],
    }
    (artifact / "protocol.lock.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (artifact / "status.json").write_text(json.dumps({"status": "S0_SNAPSHOT_CAPTURE_RUNNING"}, indent=2) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
