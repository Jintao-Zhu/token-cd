#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from research.coreact_capacity_weak.prepare import architecture_audit, copy_processors, file_sha256, parameter_audit, tensors_sha256


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    source = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
    strong = source / "training_run/trajectory/checkpoints/015000/pretrained_model"
    existing = json.loads((artifact / "architecture_audit.json").read_text())
    for name, depth in (("weak_a_8l", 8), ("weak_b_4l", 4)):
        destination = artifact / "initializations" / name
        if destination.exists(): raise FileExistsError(destination)
        torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
        config = PreTrainedConfig.from_pretrained(strong, local_files_only=True); config.pretrained_path=None; config.num_expert_layers=depth; config.device="cuda"; config.compile_model=False
        policy = SmolVLAPolicy(config); policy.save_pretrained(destination); processors=copy_processors(strong, destination)
        combined = policy.model.vlm_with_expert
        audit = {"name":name,"initialization_seed":1729,"architecture":architecture_audit(policy),"parameters":parameter_audit(policy),"processor_files":processors,"model_sha256":file_sha256(destination/"model.safetensors"),"vlm_initialization_sha256":tensors_sha256(combined.vlm.state_dict().items()),"action_expert_initialization_sha256":tensors_sha256(combined.lm_expert.state_dict().items()),"implementation_consistency_fix":True}
        expected=list(range(0,16,16//depth))
        if audit["architecture"]["expert_to_vlm_stage_mapping"] != expected: raise RuntimeError(audit)
        existing[name]=audit
        del policy; torch.cuda.empty_cache()
    if existing["weak_a_8l"]["vlm_initialization_sha256"] != existing["weak_b_4l"]["vlm_initialization_sha256"]: raise RuntimeError("VLM init mismatch")
    (artifact/"architecture_audit.json").write_text(json.dumps(existing,indent=2,sort_keys=True)+"\n")
    print(json.dumps({name:existing[name]["architecture"]["expert_to_vlm_stage_mapping"] for name in ("weak_a_8l","weak_b_4l")},sort_keys=True))


if __name__ == "__main__": main()
