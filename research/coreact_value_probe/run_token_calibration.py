#!/usr/bin/env python3
"""Run paired task-8 rollouts for signed value-selected token masks."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import joblib
import numpy as np
import torch

from research.coreact_closed_loop.run_pilot import vector_info_value, write_json
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_value_probe.token_utility import sample_selected_mask, signed_token_sets


CONDITIONS = ("vanilla", "mask_strong_positive", "mask_strong_negative", "mask_near_zero")


def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def run_one(workspace, artifact, cfg, policy, pre, post, probe, means, spec):
    directory = artifact / "episodes" / spec["episode_id"]; output = directory / "episode.json"
    if output.exists(): return json.loads(output.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    env, env_pre, env_post = make_task_env(spec["suite"], spec["task_id"], cfg)
    queue=[]; rows=[]; traces=[]; success=False; replans=0; latencies=[]
    try:
        env.envs[0].init_state_id=spec["init_state_id"]
        observation,_=env.reset(seed=spec["reset_seed"])
        for step in range(280):
            if not queue:
                prepared=prepare(policy,pre,env_pre,observation,spec["language"])
                gen=torch.Generator(device=prepared["state"].device).manual_seed(spec["action_noise_seed"]*1000+replans)
                noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=gen,device=prepared["state"].device,dtype=prepared["state"].dtype)
                started=time.perf_counter()
                with torch.inference_mode():
                    if spec["condition"]=="vanilla" or replans>=3:
                        chunk=policy.model.sample_actions(prepared["images"],prepared["image_masks"],prepared["lang_tokens"],prepared["lang_masks"],prepared["state"],noise=noise); trace=None
                    else:
                        selection=signed_token_sets(policy.model,prepared,noise,means,probe)
                        chunk,trace=sample_selected_mask(policy.model,selection,selection["sets"][spec["condition"]],noise,means)
                torch.cuda.synchronize();latencies.append(time.perf_counter()-started)
                queue.extend(chunk[:,:10,:7].transpose(0,1));replans+=1
                if trace is not None: traces.append({"replan":replans-1,**trace})
            model_action=queue.pop(0)
            if not torch.isfinite(model_action).all(): raise RuntimeError("nonfinite action")
            physical=post(model_action);legal=env_post({"action":physical})["action"]
            observation,_,terminated,_,info=env.step(legal.detach().cpu().numpy())
            success=bool(vector_info_value(info,"is_success"));rows.append({"step":step,"replan":replans-1,"success":success,"action":model_action[0].detach().float().cpu().tolist()})
            if bool(terminated[0]) or success: break
        with (directory/"steps.jsonl").open("x") as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
        record={**spec,"status":"complete","success":success,"control_steps":len(rows),"replans":replans,"masked_replans":min(replans,3) if spec["condition"]!="vanilla" else 0,"latency_median":float(np.median(latencies)),"selection_traces":traces,"step_log":str((directory/"steps.jsonl").relative_to(artifact))}
        write_json(output,record);return record
    finally: env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--artifact",type=Path,required=True);p.add_argument("--shard-index",type=int,default=0);p.add_argument("--shard-count",type=int,default=1);a=p.parse_args()
    workspace,artifact=a.workspace.resolve(),a.artifact.resolve()
    if not json.loads((artifact/"integrity_report.json").read_text())["pass"]:raise RuntimeError("integrity gate failed")
    cfg,policy,pre,post=load_policy_and_processors(workspace);probe=joblib.load(workspace/"artifacts/coreact_success_value_probe_v1_20260808_145958/linear_probe.joblib")
    means=torch.load(workspace/"artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",weights_only=True,map_location="cpu")["visual_position_mean"]
    specs=[r for i,r in enumerate(read_jsonl(artifact/"episode_manifest.jsonl")) if i%a.shard_count==a.shard_index]
    for i,spec in enumerate(specs,1):
        r=run_one(workspace,artifact,cfg,policy,pre,post,probe,means,spec);print(f"[{a.shard_index}] {i}/{len(specs)} {spec['episode_id']} success={r['success']}",flush=True)
if __name__=="__main__":main()

