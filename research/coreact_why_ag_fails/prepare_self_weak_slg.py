#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--output",type=Path);args=parser.parse_args()
    workspace=args.workspace.resolve();artifact=(args.output or workspace/"artifacts"/f"coreact_slg_self_weak_low_noise_v1_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if artifact.exists():raise FileExistsError(artifact)
    for name in ("selection","confirmation","logs","status","figures"):(artifact/name).mkdir(parents=True,exist_ok=True)
    manifold=workspace/"artifacts/coreact_why_ag_fails_manifold_v1_20260816_225256";source=workspace/"artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522";checkpoint=source/"training_run/trajectory/checkpoints/015000/pretrained_model"
    manifest=manifold/"state_manifest.jsonl";rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows)!=250:raise RuntimeError("selection manifest is not 250 states")
    (artifact/"state_manifest.jsonl").write_text(manifest.read_text())
    protocol={
      "experiment":"Flow-VLA SGG-style Self-Weak Low-Noise Compatibility Test","stage":"SELECTION_ONLY_CONFIRMATION_SEALED","created_at":datetime.now().astimezone().isoformat(),
      "strong":{"checkpoint":str(checkpoint),"model_sha256":sha256(checkpoint/"model.safetensors"),"expert_layers":16},
      "self_weak":{"W1_skip_last_1":{"affected_expert_layer_indices_zero_based":[15],"residual_scale":0.0},"W2_skip_last_2":{"affected_expert_layer_indices_zero_based":[14,15],"residual_scale":0.0},"operator":"h_out=h_in+alpha*(h_full-h_in)","shared_checkpoint":True,"VLM_vision_condition_unchanged":True},
      "scheduler":{"x_t":"t*epsilon+(1-t)*action","target":"epsilon-action","highest_noise":{"flow_step":0,"t":1.0},"lowest_noise":{"flow_step":9,"t":0.1}},
      "selection":{"states":250,"manifest":str(manifest),"manifest_sha256":sha256(manifest),"manifold_parent":str(manifold),"neighbor_manifest_sha256":sha256(manifold/"neighbor_manifest.jsonl"),"k":16,"noise_seeds":3,"timesteps":10},
      "windows":{"last_2":{"low_steps":[8,9],"high_placebo_steps":[0,1]},"last_3":{"low_steps":[7,8,9],"high_placebo_steps":[0,1,2]},"last_4":{"low_steps":[6,7,8,9],"high_placebo_steps":[0,1,2,3]}},
      "guidance":{"lambda":0.5,"trust_region_kappa":0.25,"unchanged":True},
      "gate":{"delta_d_applied_positive":0.55,"marginal_g_positive":0.60,"marginal_ci_lower_strictly_above":0.50,"tasks_marginal_above_half":7,"low_minus_full_pp":10,"low_minus_high_pp":10,"low_delta_d_strictly_above_high":True},
      "prohibited":["training","rollout","lambda tuning","new skip pattern","noncontiguous window","per-task window","independent Weak rescue","neighbor retrieval changes"]}
    (artifact/"protocol.json").write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n");(artifact/"status/current.json").write_text(json.dumps({"stage":"PREPARED_SELECTION"},indent=2)+"\n");print(artifact)


if __name__=="__main__":main()
