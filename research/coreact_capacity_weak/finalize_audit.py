#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from research.coreact_capacity_weak.prepare import architecture_audit, file_sha256, parameter_audit, tensors_sha256


PROGRESS = (("early", 0.15), ("middle", 0.45), ("late", 0.75))


def load_policy(path):
    config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    config.device = "cuda"
    config.compile_model = False
    return SmolVLAPolicy.from_pretrained(path, config=config, local_files_only=True)


def audit_loaded(policy, path):
    combined = policy.model.vlm_with_expert
    return {
        "checkpoint": str(path),
        "architecture": architecture_audit(policy),
        "parameters": parameter_audit(policy),
        "model_file_sha256": file_sha256(path / "model.safetensors"),
        "loaded_vlm_sha256": tensors_sha256(combined.vlm.state_dict().items()),
        "loaded_action_expert_sha256": tensors_sha256(combined.lm_expert.state_dict().items()),
        "all_finite": all(torch.isfinite(value).all() for value in policy.state_dict().values() if value.is_floating_point()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    source = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
    strong_path = source / "training_run/trajectory/checkpoints/015000/pretrained_model"
    architecture = json.loads((artifact / "architecture_audit.json").read_text())

    policies = [("strong_16l", strong_path)] + [(name, artifact / "initializations" / name) for name in ("weak_a_8l", "weak_b_4l")]
    reload_audits = {}
    for name, path in policies:
        policy = load_policy(path)
        reload_audits[name] = audit_loaded(policy, path)
        if name != "strong_16l":
            if reload_audits[name]["loaded_vlm_sha256"] != architecture[name]["vlm_initialization_sha256"]:
                raise RuntimeError(f"{name}: VLM checkpoint reload mismatch")
            if reload_audits[name]["loaded_action_expert_sha256"] != architecture[name]["action_expert_initialization_sha256"]:
                raise RuntimeError(f"{name}: action expert checkpoint reload mismatch")
        del policy
        torch.cuda.empty_cache()
    if reload_audits["strong_16l"]["architecture"]["expert_to_vlm_stage_mapping"] != list(range(16)):
        raise RuntimeError("Strong native mapping mismatch")
    architecture["strong_16l"] = reload_audits["strong_16l"]
    architecture["reload_integrity"] = reload_audits
    (artifact / "architecture_audit.json").write_text(json.dumps(architecture, indent=2, sort_keys=True) + "\n")

    previous_manifest = workspace / "artifacts/coreact_flow_timestep_compatibility_f0_v1_20260815_185400/state_manifest.jsonl"
    previous = [json.loads(line) for line in previous_manifest.read_text().splitlines()]
    previous_by_demo = {(row["task_id"], row["demo_ordinal"]): row for row in previous}
    states = []
    for task_id in range(10):
        path = Path(previous_by_demo[(task_id, 0)]["demo_path"])
        with h5py.File(path, "r") as handle:
            demos = sorted(handle["data"], key=lambda value: int(value.split("_")[-1]))
            for ordinal, demo_id in enumerate(demos):
                split = "selection" if ordinal % 2 == 0 else "confirmation"
                split_rank = ordinal // 2
                phase, progress = PROGRESS[split_rank % 3]
                length = len(handle["data"][demo_id]["actions"])
                frame = math.floor((length - 1) * progress)
                previous_frame = previous_by_demo[(task_id, ordinal)]["resolved_frame"]
                if frame == previous_frame:
                    raise RuntimeError(f"fresh-state collision task={task_id} demo={ordinal}")
                states.append({
                    "state_id": f"task{task_id:02d}__demo{ordinal:02d}__frame{frame:04d}",
                    "split": split,
                    "task_id": task_id,
                    "language": previous_by_demo[(task_id, ordinal)]["language"],
                    "demo_id": demo_id,
                    "demo_ordinal": ordinal,
                    "episode_length": length,
                    "phase": phase,
                    "target_progress": progress,
                    "resolved_frame": frame,
                    "demo_path": str(path),
                    "noise_seeds": [202608160000 + task_id * 10000 + ordinal * 10 + index for index in range(3)],
                })
    if sum(row["split"] == "selection" for row in states) != 250 or sum(row["split"] == "confirmation" for row in states) != 250:
        raise RuntimeError("split size mismatch")
    (artifact / "state_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in states))

    protocol = json.loads((artifact / "protocol.json").read_text())
    protocol["strong"]["architecture"] = reload_audits["strong_16l"]["architecture"]
    protocol["strong"]["parameters"] = reload_audits["strong_16l"]["parameters"]
    protocol["offline"]["manifest"] = str(artifact / "state_manifest.jsonl")
    protocol["offline"]["fresh_progress_schedule"] = [0.15, 0.45, 0.75]
    protocol["offline"]["fresh_relative_to_previous_f0"] = True
    protocol["audit_finalized"] = True
    (artifact / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (artifact / "status/current.json").write_text(json.dumps({"stage": "AUDIT_PASS", "training": "authorized"}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "states": len(states), "strong_parameters": reload_audits["strong_16l"]["parameters"]["all"]}, sort_keys=True))


if __name__ == "__main__":
    main()
