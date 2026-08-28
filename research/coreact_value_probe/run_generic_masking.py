#!/usr/bin/env python3
"""Append-only four-condition generic masking runner."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np,torch
from research.coreact_closed_loop.guidance import GuidanceConfig
from research.coreact_closed_loop.run_pilot import vector_info_value,write_json
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_revision.masked_sampler import sample_masked_actions
def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--artifact",type=Path,required=True);p.add_argument("--shard-index",type=int,default=0);p.add_argument("--shard-count",type=int,default=1);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();cfg,policy,pre,post=load_policy_and_processors(w);means=torch.load(w/"artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",weights_only=True,map_location="cpu")["visual_position_mean"]
 specs=[r for i,r in enumerate(read(o/"episode_manifest.jsonl")) if i%a.shard_count==a.shard_index]
 for ordinal,s in enumerate(specs,1):
  d=o/"episodes"/s["episode_id"];out=d/"episode.json"
  if out.exists():continue
  d.mkdir(parents=True,exist_ok=True);env,ep,epo=make_task_env(s["suite"],s["task_id"],cfg);queue=[];actions=[];steps=[];traces=[];lat=[];success=False;replans=0;mask_delta=[];prev_action=None;boundaries=[];eef_start=None;eef_end=None
  try:
   env.envs[0].init_state_id=s["init_state_id"];obs,_=env.reset(seed=s["reset_seed"]);inner=env.envs[0]._env.env
   for control in range(280):
    new_chunk=not queue
    if not queue:
     prepared=prepare(policy,pre,ep,obs,s["language"]);gen=torch.Generator(device=prepared["state"].device).manual_seed(s["action_noise_seed"]*1000+replans);noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=gen,device=prepared["state"].device,dtype=prepared["state"].dtype);start=time.perf_counter()
     with torch.inference_mode():
      if s["condition"]=="vanilla" or replans>=3:chunk=policy.model.sample_actions(prepared["images"],prepared["image_masks"],prepared["lang_tokens"],prepared["lang_masks"],prepared["state"],noise=noise);trace=None
      else:
       mode={"attention_top8":"top","random8":"random","bottom8":"bottom"}[s["condition"]];chunk,trace=sample_masked_actions(policy.model,prepared["images"],prepared["image_masks"],prepared["lang_tokens"],prepared["lang_masks"],prepared["state"],noise,means,config=GuidanceConfig(selection=mode,group_count=8,action_dim=7,num_steps=10),selection_seed=s["selection_seed"]*1000+replans);clean=policy.model.sample_actions(prepared["images"],prepared["image_masks"],prepared["lang_tokens"],prepared["lang_masks"],prepared["state"],noise=noise);mask_delta.append(float(torch.linalg.vector_norm(chunk[:,:10,:7]-clean[:,:10,:7])))
     torch.cuda.synchronize();lat.append(time.perf_counter()-start);queue.extend(chunk[:,:10,:7].transpose(0,1));
     if trace is not None:traces.append({"replan":replans,**trace})
     replans+=1
    model_action=queue.pop(0);finite=bool(torch.isfinite(model_action).all());
    if not finite:raise RuntimeError("nonfinite action")
    if new_chunk and prev_action is not None:boundaries.append(float(torch.linalg.vector_norm(model_action[0]-prev_action)))
    prev_action=model_action[0].detach().clone();physical=post(model_action);legal=epo({"action":physical})["action"];obs,_,terminated,_,info=env.step(legal.detach().cpu().numpy());success=bool(vector_info_value(info,"is_success"));actions.append(model_action[0].detach().float().cpu().numpy());
    pos=np.asarray(inner.sim.data.site_xpos[inner.robots[0].eef_site_id]);eef_start=pos.copy() if eef_start is None else eef_start;eef_end=pos.copy();steps.append({"control_step":control,"replan":replans-1,"success":success,"model_action":model_action[0].detach().float().cpu().tolist()})
    if bool(terminated[0]) or success:break
   aa=np.stack(actions);tv=float(np.linalg.norm(np.diff(aa,axis=0),axis=1).sum()) if len(aa)>1 else 0.;write_json(out,{**s,"status":"complete","success":success,"control_steps":len(steps),"replans":replans,"action_total_variation":tv,"chunk_boundary_discontinuity":float(np.mean(boundaries)) if boundaries else 0.,"eef_displacement":float(np.linalg.norm(eef_end-eef_start)) if eef_start is not None else 0.,"latency_median":float(np.median(lat)) if lat else 0.,"mask_action_delta_norms":mask_delta,"selection_traces":traces,"step_log":str((d/"steps.jsonl").relative_to(o))});(d/"steps.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in steps));print(f"[{a.shard_index}] {ordinal}/{len(specs)} {s['episode_id']} success={success}",flush=True)
  finally:env.close()
if __name__=="__main__":main()
