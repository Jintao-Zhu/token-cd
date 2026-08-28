#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from research.coreact_capacity_weak.prepare import architecture_audit, parameter_audit
from research.coreact_capacity_weak.regression_capture import inputs, tensor_hash, velocity


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--artifact",type=Path,required=True); args=parser.parse_args(); artifact=args.artifact.resolve(); results={}
    for name, expected in (("weak_a_8l",[0,2,4,6,8,10,12,14]),("weak_b_4l",[0,4,8,12])):
        path=artifact/"initializations"/name; config=PreTrainedConfig.from_pretrained(path,local_files_only=True); config.device="cuda"; config.compile_model=False
        policy=SmolVLAPolicy.from_pretrained(path,config=config,local_files_only=True).eval(); values=inputs(policy.model,torch.device("cuda")); output=velocity(policy.model,*values); architecture=architecture_audit(policy)
        result={"output_shape":list(output.shape),"output_sha256":tensor_hash(output),"finite":bool(torch.isfinite(output).all()),"mapping":architecture["expert_to_vlm_stage_mapping"],"parameters":parameter_audit(policy),"pass":bool(list(output.shape)==[1,50,32] and torch.isfinite(output).all() and architecture["expert_to_vlm_stage_mapping"]==expected)}
        if not result["pass"]: raise RuntimeError(result)
        results[name]=result; del policy; torch.cuda.empty_cache()
    target=artifact/"implementation_gate"; (target/"reduced_forward.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n"); print(json.dumps(results,sort_keys=True))


if __name__=="__main__":main()
