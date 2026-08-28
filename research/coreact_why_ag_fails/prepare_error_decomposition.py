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


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--output", type=Path); args = parser.parse_args()
    workspace = args.workspace.resolve(); output = (args.output or workspace/"artifacts"/f"coreact_why_ag_fails_error_decomposition_v1_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if output.exists(): raise FileExistsError(output)
    for name in ("states", "logs", "status", "figures"): (output/name).mkdir(parents=True, exist_ok=True)
    source = workspace/"artifacts/coreact_capacity_weak_v1_20260815_202000"
    manifest = [json.loads(line) for line in (source/"state_manifest.jsonl").read_text().splitlines() if json.loads(line)["split"] == "selection"]
    if len(manifest) != 250: raise RuntimeError(len(manifest))
    (output/"state_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in manifest))
    checkpoints = {
        "strong_16l": workspace/"artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model",
        "weak_a_8l": source/"training/weak_a_8l/checkpoints/015000/pretrained_model",
        "weak_b_4l": source/"training/weak_b_4l/checkpoints/015000/pretrained_model",
    }
    protocol = {
        "experiment_name": "coreact_why_ag_fails_error_decomposition_v1", "created_at": datetime.now().astimezone().isoformat(), "stage": "STEP1_ERROR_DECOMPOSITION_ONLY",
        "models": {name: {"checkpoint": str(path), "model_sha256": sha256(path/"model.safetensors")} for name,path in checkpoints.items()},
        "data": {"source_manifest": str(source/"state_manifest.jsonl"), "source_manifest_sha256": sha256(source/"state_manifest.jsonl"), "split": "selection_only", "states": 250, "tasks": 10, "noise_seeds_per_state": 3, "flow_timesteps": 10},
        "definitions": {"eS": "vS-u", "eW": "vW-u", "alpha": "dot(eW,eS)/(norm(eS)^2)", "e_perp": "eW-alpha*eS", "orthogonal_ratio": "norm(e_perp)/norm(eW)"},
        "scopes": ["full_50", "front_10_executed", "tail_40", "translation", "rotation", "gripper", "chunk_positions_0_49"],
        "aggregation": "state is the independent unit; noise/timestep are repeated measurements within state",
        "prohibited": ["new training", "rollout", "manifold target", "cross-swap", "timestep/window selection", "lambda tuning", "5k/10k checkpoint selection"],
    }
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True)+"\n"); (output/"status/current.json").write_text(json.dumps({"stage":"PREPARED","states":250},indent=2)+"\n"); print(output)


if __name__ == "__main__": main()
