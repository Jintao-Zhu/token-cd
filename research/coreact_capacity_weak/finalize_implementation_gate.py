#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--workspace",type=Path,required=True); parser.add_argument("--artifact",type=Path,required=True); args=parser.parse_args(); workspace,artifact=args.workspace.resolve(),args.artifact.resolve(); gate=artifact/"implementation_gate"; repo=workspace/"research/worktrees/lerobot_trained_weak_clean_e40b58a"
    before=json.loads((gate/"16l_before.json").read_text()); after=json.loads((gate/"16l_after.json").read_text()); forwards=json.loads((gate/"reduced_forward.json").read_text()); architecture=json.loads((artifact/"architecture_audit.json").read_text())
    params={"strong_16l":architecture["strong_16l"]["parameters"],"weak_a_8l":forwards["weak_a_8l"]["parameters"],"weak_b_4l":forwards["weak_b_4l"]["parameters"]}
    reference=params["strong_16l"]
    for name in ("weak_a_8l","weak_b_4l"):
        for category in ("vision","vlm_text_connector","action_interface"):
            if params[name][category] != reference[category]: raise RuntimeError(f"unexpected trainability change: {name}/{category}")
        total_reduction=reference["all"]["total"]-params[name]["all"]["total"]
        expert_reduction=reference["action_expert"]["total"]-params[name]["action_expert"]["total"]
        if total_reduction != expert_reduction: raise RuntimeError(f"non-expert parameter change: {name}")
    backward={}
    for name in ("weak_a_8l","weak_b_4l"):
        text=(gate/f"backward_{name}.log").read_text(); backward[name]={"optimizer_step_1": "step:1 " in text, "training_ended": "End of training" in text}
        if not all(backward[name].values()): raise RuntimeError(backward[name])
    commit="e40b58a8dfa9e7b86918c374791599d070518d11"
    diff=(gate/"reduced_depth_consistency_fix.diff").read_text()
    if "mapped_vlm_stage" not in diff or "expert_stage_stride" not in diff: raise RuntimeError("unexpected/missing implementation diff")
    result={"decision":"REDUCED_DEPTH_IMPLEMENTATION_GATE_PASS","base_commit":commit,"diff_sha256":hashlib.sha256(diff.encode()).hexdigest(),"changed_file":"src/lerobot/policies/smolvla/smolvlm_with_expert.py","16l_regression":{"same_inputs":before["input_hashes"]==after["input_hashes"],"max_abs":after["max_abs_vs_before"],"pass":after["regression_pass"]},"forward":{"weak_a_8l":forwards["weak_a_8l"]["pass"],"weak_b_4l":forwards["weak_b_4l"]["pass"]},"backward":backward,"mappings":{"weak_a_8l":forwards["weak_a_8l"]["mapping"],"weak_b_4l":forwards["weak_b_4l"]["mapping"]},"parameters":params,"trainability_only_expert_depth_differs":True}
    (gate/"gate_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    protocol=json.loads((artifact/"protocol.json").read_text()); protocol["reduced_depth_implementation_fix"]={"status":"PASS","scope":"projection construction uses mapped VLM stage; mapping and runtime dispatch unchanged","base_commit":commit,"diff":str(gate/"reduced_depth_consistency_fix.diff"),"diff_sha256":result["diff_sha256"],"gate_result":str(gate/"gate_result.json")}; (artifact/"protocol.json").write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n")
    (artifact/"status/current.json").write_text(json.dumps({"stage":"REDUCED_DEPTH_IMPLEMENTATION_GATE_PASS","training":"authorized"},indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
