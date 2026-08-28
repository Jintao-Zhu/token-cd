#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = (args.output or workspace / "artifacts" / f"coreact_why_ag_fails_manifold_v1_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if output.exists():
        raise FileExistsError(output)
    for name in ("neighbors", "states", "logs", "status", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)

    step1 = workspace / "artifacts/coreact_why_ag_fails_error_decomposition_v1_20260816_090151"
    capacity = workspace / "artifacts/coreact_capacity_weak_v1_20260815_202000"
    source = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
    manifest = step1 / "state_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != 250 or {row["split"] for row in rows} != {"selection"}:
        raise RuntimeError("Step 2 must use the frozen 250-state Step 1 manifest")
    (output / "state_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    checkpoints = {
        "strong_16l": source / "training_run/trajectory/checkpoints/015000/pretrained_model",
        "weak_a_8l": capacity / "training/weak_a_8l/checkpoints/015000/pretrained_model",
        "weak_b_4l": capacity / "training/weak_b_4l/checkpoints/015000/pretrained_model",
    }
    protocol = {
        "experiment_name": "coreact_why_ag_fails_action_manifold_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "STEP2_ACTION_MANIFOLD_DIAGNOSIS_ONLY",
        "models": {name: {"checkpoint": str(path), "model_sha256": sha256(path / "model.safetensors")} for name, path in checkpoints.items()},
        "data": {"manifest": str(manifest), "manifest_sha256": sha256(manifest), "states": 250, "tasks": 10, "noise_seeds_per_state": 3, "timesteps": 10},
        "retrieval": {
            "task_constraint": "same task only",
            "k_primary": 16,
            "anchor_trajectory_excluded": True,
            "temporal_deduplication": "one physically nearest eligible frame per non-anchor trajectory before selecting K",
            "candidate_horizon_rule": "candidate remaining actions >= anchor valid horizon",
            "features": {"object_pose": "dynamic object root-body xyz + canonicalized quaternion", "eef_pose": "stored ee_pos + ee_ori", "gripper": "stored gripper_states", "robot_state": "stored joint_states"},
            "standardization": "per task and scalar feature dimension over all expert frames; std floor 1e-6",
            "distance": "sum over feature groups of mean squared standardized coordinate distance",
            "vlm_embedding_forbidden": True,
        },
        "posterior": {"neighbor_prior": "uniform over frozen K=16", "log_likelihood": "-||x_t-(1-t)A_j||^2/(2*t^2) over anchor-valid normalized action coordinates", "target": "u_j=(x_t-A_j)/t"},
        "guidance": {"lambda": 0.5, "trust_region_kappa": 0.25, "implementation": "identical to capacity-Weak offline point_metrics"},
        "decision": {"misclassification": ">=10pp improvement over single-demo and at least one primary metric >=60%", "not_explained": "primary manifold metrics remain <=50%"},
        "prohibited": ["training", "rollout", "lambda tuning", "K tuning", "task-specific retrieval", "VLM retrieval", "Strong/Weak-dependent posterior weights"],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (output / "status/current.json").write_text(json.dumps({"stage": "PREPARED"}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
