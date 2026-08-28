#!/usr/bin/env python3
"""Capture target-free cross-flow action-consistency features."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import h5py,numpy as np,torch
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from research.coreact_closed_loop.runtime import env_config,load_policy_and_processors
from research.coreact_exploration.instrumentation import grouped_intervention,predict_teacher_forced_velocity
from research.coreact_exploration.run_development_gates import CAMERA_IDS,replacements_for
from research.coreact_region.segmented_runtime import batched_observation,make_segmented_env
from research.coreact_revision.run_region_sign_capture_v3 import TAUS,NOISE_SEEDS,action_chunk,rewrite_demo_xml

CONDITION_SUBSETS = {
 "full6": (0, 1, 2, 3, 4, 5),
 "tau05_noise2": (2, 3),
 "tau3_seed1729": (0, 2, 4),
 "tau3_seed9473": (1, 3, 5),
 "extremes4": (0, 1, 4, 5),
}

def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def append(p,x):
 with Path(p).open("a") as f:f.write(json.dumps(x,sort_keys=True)+"\n")
def dispersion(x):
 mean=x.mean(0);return float(((x-mean)**2).mean()/(mean.square().mean()+1e-8))
def pairwise_direction_cosine(x):
 x=x.flatten(1);x=x/(x.norm(dim=1,keepdim=True)+1e-8);m=x@x.T;n=len(x);return float((m.sum()-m.diagonal().sum())/max(1,n*(n-1)))

def subset_features(clean_stack,masked_stack,indices):
 idx=torch.as_tensor(indices,dtype=torch.long)
 clean=clean_stack.index_select(0,idx);masked=masked_stack.index_select(0,idx)
 clean_mean=clean.mean(0);masked_mean=masked.mean(0);delta=masked-clean
 clean_disp=dispersion(clean);masked_disp=dispersion(masked)
 return {
  "clean_cross_condition_dispersion":clean_disp,
  "masked_cross_condition_dispersion":masked_disp,
  "masked_minus_clean_dispersion":masked_disp-clean_disp,
  "clean_mask_consensus_distance":float((masked_mean-clean_mean).square().mean()/(clean_mean.square().mean()+1e-8)),
  "intervention_direction_consistency":pairwise_direction_cosine(delta),
  "mean_intervention_norm":float(delta.norm(dim=1).mean()),
 }

def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--source-artifact",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--means",type=Path,required=True);a=p.parse_args();w,s,o=a.workspace.resolve(),a.source_artifact.resolve(),a.output.resolve();o.parent.mkdir(parents=True,exist_ok=True)
 manifest={(r["task_id"],r["demo_id"],r["frame_id"]):r for r in read(s/"state_manifest.jsonl")};names={r["task_id"]:r["name"] for r in json.loads((s/"task_manifest.json").read_text())};done={(r["task_id"],r["demo_id"],r["frame_id"],r["group_id"]) for r in read(o)} if o.exists() else set()
 if len(manifest)!=300:raise RuntimeError(f"expected 300 states, got {len(manifest)}")
 cfg,policy,pre,_=load_policy_and_processors(w);means=torch.load(a.means,weights_only=True,map_location="cpu");dataset=w/"LIBERO/libero/datasets/libero_spatial"
 for task in sorted(names):
  env=make_segmented_env("libero_spatial",task);env_pre,_=make_env_pre_post_processors(env_cfg=env_config("libero_spatial",task),policy_cfg=cfg);rows=[r for k,r in manifest.items() if k[0]==task]
  try:
   with h5py.File(dataset/f"{names[task]}_demo.hdf5","r") as h5:
    by=defaultdict(list)
    for r in rows:by[r["demo_id"]].append(r)
    for demo in sorted(by,key=lambda x:int(x.split("_")[-1])):
     ep=h5["data"][demo];states=np.asarray(ep["states"]);actions=np.asarray(ep["actions"]);xml=ep.attrs["model_file"];xml=xml.decode() if isinstance(xml,bytes) else xml
     env._env.reset();env._env.reset_from_xml_string(__import__("libero.libero.envs.utils",fromlist=["postprocess_model_xml"]).postprocess_model_xml(rewrite_demo_xml(xml,w),{},demo_generation=False));env._env.env.sim.reset()
     for row in sorted(by[demo],key=lambda x:x["frame_id"]):
      frame=row["frame_id"];groups=row["selected_groups"]
      if all((task,demo,frame,f"prefix-{i}") in done for i in groups):continue
      raw=env._env.regenerate_obs_from_state(states[frame]);obs=env._format_raw_obs(raw);act,pad=action_chunk(actions,frame);batch=preprocess_observation(batched_observation(obs));batch["task"]=[row["language"]];batch["action"]=act.unsqueeze(0);batch["action_is_pad"]=pad.unsqueeze(0);batch=pre(env_pre(batch));images,masks=policy.prepare_images(batch);state=policy.prepare_state(batch);actions_t=policy.prepare_action(batch);clean_vectors=[];masked_vectors=[[] for _ in groups]
      for tau_v in TAUS:
       for seed in NOISE_SEEDS:
        gen=torch.Generator(device=actions_t.device).manual_seed(seed);noise=torch.randn(actions_t.shape,generator=gen,device=actions_t.device,dtype=actions_t.dtype);tau=torch.tensor([tau_v],device=actions_t.device,dtype=actions_t.dtype)
        with torch.inference_mode():base=predict_teacher_forced_velocity(policy.model,images,masks,batch["observation.language.tokens"],batch["observation.language.attention_mask"],state,actions_t,tau,noise,record_attention=False,camera_ids=CAMERA_IDS)
        def intervene(prefix,maps):return grouped_intervention(prefix,maps[0],[[i] for i in groups],replacements_for(maps[0],means,mode="position"))
        with torch.inference_mode():neg=predict_teacher_forced_velocity(policy.model,images,masks,batch["observation.language.tokens"],batch["observation.language.attention_mask"],state,actions_t,tau,noise,prefix_intervention=intervene,record_attention=False,camera_ids=CAMERA_IDS)
        dim=act.shape[-1];valid=(~batch["action_is_pad"]).unsqueeze(-1).expand(-1,-1,dim)[0];clean=(base["x_tau"][:,:,:dim]-tau_v*base["v_pred"][:,:,:dim])[0][valid].detach().cpu();clean_vectors.append(clean)
        estimates=neg["x_tau"][:,:,:dim]-tau_v*neg["v_pred"][:,:,:dim]
        for j in range(len(groups)):masked_vectors[j].append(estimates[j][valid].detach().cpu())
      clean_stack=torch.stack(clean_vectors)
      for index,vectors in zip(groups,masked_vectors,strict=True):
       masked=torch.stack(vectors);output={"task_id":task,"demo_id":demo,"frame_id":frame,"group_id":f"prefix-{index}","conditions":6}
       for subset,indices in CONDITION_SUBSETS.items():
        output.update({f"{subset}_{name}":value for name,value in subset_features(clean_stack,masked,indices).items()})
       append(o,output)
      print(f"task={task} {demo} frame={frame} complete",flush=True)
  finally:env.close()
 print(f"task {task} complete",flush=True)
 print(f"captured {len(read(o))} consistency rows",flush=True)
if __name__=="__main__":main()
