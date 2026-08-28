from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=False)
    for name in ("snapshots", "audits", "logs", "u1", "invalid_units"):
        (artifact / name).mkdir()
    checkpoint = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model"
    protocol = {
        "experiment": "Local Success Direction / Action Utility Existence Test Phase U1",
        "created_at": datetime.now().astimezone().isoformat(),
        "suite": "libero_spatial",
        "strong_checkpoint": str(checkpoint),
        "strong_model_sha256": sha256(checkpoint / "model.safetensors"),
        "snapshots": {"tasks": list(range(10)), "init_state_ids": list(range(45, 50)), "progress": [0.30, 0.65], "count": 100, "horizon": 520},
        "pca": {"source": "libero_spatial training demonstrations only", "dimensions": "first10x7 normalized actions", "axes": 4, "directions": 8},
        "evaluation": {"seeds_per_snapshot": 5, "selection_seeds": [0, 1, 2], "held_seeds": [3, 4], "episodes": 4500, "one_shot": True},
        "radius": "per-unit norm of executed W1(+0.05) first10 action delta",
        "tie_break": "lowest direction number",
    }
    (artifact / "protocol.lock.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (artifact / "status.json").write_text(json.dumps({"status": "U1_SNAPSHOT_CAPTURE_RUNNING"}, indent=2) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
