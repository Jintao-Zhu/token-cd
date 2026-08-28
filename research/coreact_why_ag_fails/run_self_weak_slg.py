#!/usr/bin/env python3
from __future__ import annotations

import argparse,json,os
from pathlib import Path
import h5py,numpy as np,torch
from lerobot.envs.factory import make_env_pre_post_processors
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_capacity_weak.metrics import exact_training_pair
from research.coreact_closed_loop.runtime import env_config
from research.coreact_expert_direction.direction_audit import action_chunk,prepare_demo_state,rewrite_demo_xml
from research.coreact_quality_negative_branch.quality_branch import BRANCHES,branch_scale_matrix,velocity_from_embeddings
from research.coreact_region.segmented_runtime import make_segmented_env
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_why_ag_fails.run_manifold_diagnosis import normalize_candidates

TIMES=tuple(round(1.-.1*i,1) for i in range(10));PRIMARY=BRANCHES[:2];EPS=1e-12


def metrics(strong,weak,candidate_actions,x_t,tau,valid):
 flat=valid[:,None].expand_as(strong);s,w=strong[flat].float(),weak[flat].float();d=s-w
 acts=candidate_actions[:,valid,:7].float().reshape(len(candidate_actions),-1);x=x_t[valid,:7].float().reshape(-1);targets=(x[None]-acts)/tau
 ll=-torch.sum((x[None]-(1.-tau)*acts)**2,1)/(2.*tau*tau);weights=torch.softmax(ll-torch.max(ll),0);marginal=torch.sum(weights[:,None]*targets,0);nearest=targets[torch.argmin(torch.sum((targets-s[None])**2,1))]
 gm=torch.dot(d,marginal-s);gn=torch.dot(d,nearest-s);raw=s+.5*d;raw_correction=.5*(strong-weak);clip=torch.clamp(.25*torch.linalg.vector_norm(strong)/(torch.linalg.vector_norm(raw_correction)+EPS),max=1.);applied=s+.5*d*clip
 ds=torch.min(torch.sum((targets-s[None])**2,1));dw=torch.min(torch.sum((targets-w[None])**2,1));dr=torch.min(torch.sum((targets-raw[None])**2,1));da=torch.min(torch.sum((targets-applied[None])**2,1));sm=torch.sum((s-marginal)**2);wm=torch.sum((w-marginal)**2)
 return {"marginal_g_positive":bool(gm>0),"marginal_cosine":float(gm/(torch.linalg.vector_norm(d)*torch.linalg.vector_norm(marginal-s)+EPS)),"nearest_g_positive":bool(gn>0),"nearest_cosine":float(gn/(torch.linalg.vector_norm(d)*torch.linalg.vector_norm(nearest-s)+EPS)),"delta_d_raw":float(ds-dr),"delta_d_raw_positive":bool(dr<ds),"delta_d_applied":float(ds-da),"delta_d_applied_positive":bool(da<ds),"weak_to_strong_manifold_distance_ratio":float(dw/(ds+EPS)),"weak_to_strong_marginal_error_ratio":float(wm/(sm+EPS)),"direction_relative_norm":float(torch.linalg.vector_norm(d)/(torch.linalg.vector_norm(s)+EPS)),"clip_scale":float(clip),"clipped":bool(clip<1.-1e-7),"posterior_ess":float(1./torch.sum(weights**2))}


@torch.no_grad()
def evaluate(model,preprocessor,batch,raw_candidates,raw_anchor,state):
 actions=batch["actions"];candidates=normalize_candidates(preprocessor,raw_candidates,actions.device,actions.dtype,actions.shape[-1]);anchor=normalize_candidates(preprocessor,raw_anchor[None],actions.device,actions.dtype,actions.shape[-1]);normalization_max_abs=float((anchor-actions).abs().max());
 if normalization_max_abs>=1e-6:raise RuntimeError(f"normalization parity {normalization_max_abs}")
 noises=torch.cat([torch.randn(actions.shape,generator=torch.Generator(device=actions.device).manual_seed(seed),device=actions.device,dtype=actions.dtype) for seed in state["noise_seeds"]]);actions3=actions.expand(3,-1,-1);valid=~batch["action_is_pad"][0];prefix,pad,att=model.embed_prefix(batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],state=batch["state"]);scales=branch_scale_matrix(model.vlm_with_expert.num_expert_layers,3,device=actions.device,dtype=actions.dtype)[:len(PRIMARY)*3];rows=[];parity=0.
 for step,tau in enumerate(TIMES):
  timestep=torch.full((3,),tau,device=actions.device,dtype=actions.dtype);x_t,_=exact_training_pair(actions3,noises,timestep);strong=velocity_from_embeddings(model,prefix.expand(3,-1,-1),pad.expand(3,-1),att.expand(3,-1),x_t,timestep)[:,:,:7];weak=velocity_from_embeddings(model,prefix.expand(6,-1,-1),pad.expand(6,-1),att.expand(6,-1),torch.cat([x_t,x_t]),timestep.repeat(2),expert_residual_scales=scales).reshape(2,3,50,-1)[:,:,:,:7]
  if step==0:
   same=velocity_from_embeddings(model,prefix.expand(3,-1,-1),pad.expand(3,-1),att.expand(3,-1),x_t,timestep,expert_residual_scales=torch.ones(3,model.vlm_with_expert.num_expert_layers,device=actions.device,dtype=actions.dtype))[:,:,:7];parity=float((same-strong).abs().max())
  for bi,branch in enumerate(PRIMARY):
   for ni in range(3):rows.append({"candidate":branch.name,"flow_step":step,"timestep":tau,"noise_ordinal":ni,**metrics(strong[ni],weak[bi,ni],candidates,x_t[ni,:,:7],tau,valid)})
 expected={"W1_skip_last_1":[1.]*15+[0.],"W2_skip_last_2":[1.]*14+[0.,0.]};actual={branch.name:scales[i*3].float().cpu().tolist() for i,branch in enumerate(PRIMARY)}
 if actual!=expected:raise RuntimeError(f"scale contract mismatch {actual}")
 return rows,{"finite":all(np.isfinite(v) for row in rows for v in row.values() if isinstance(v,float)),"rows":len(rows),"normalization_max_abs":normalization_max_abs,"all_ones_strong_parity_max_abs":parity,"num_expert_layers":model.vlm_with_expert.num_expert_layers,"scale_contract":actual}


def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--artifact",type=Path,required=True);p.add_argument("--task-ids",type=int,nargs="+",default=list(range(10)));p.add_argument("--max-states-per-task",type=int);a=p.parse_args();w,art=a.workspace.resolve(),a.artifact.resolve();parent=w/"artifacts/coreact_why_ag_fails_manifold_v1_20260816_225256";states=[json.loads(x) for x in (art/"state_manifest.jsonl").read_text().splitlines() if json.loads(x)["task_id"] in a.task_ids]
 if a.max_states_per_task is not None:states=[r for t in a.task_ids for r in [x for x in states if x["task_id"]==t][:a.max_states_per_task]]
 source=w/"artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522";os.chdir(w/"LIBERO");config,policy,pre,_=load_policy(source/"training_run/trajectory/checkpoints/015000/pretrained_model");model=policy.model
 for task in a.task_ids:
  task_states=[x for x in states if x["task_id"]==task]
  if not task_states:continue
  env_pre,_=make_env_pre_post_processors(env_cfg=env_config("libero_spatial",task),policy_cfg=config);env=make_segmented_env("libero_spatial",task)
  try:
   with h5py.File(task_states[0]["demo_path"],"r") as h:
    for i,state in enumerate(task_states,1):
     out=art/"selection"/f"{state['state_id']}.json"
     if out.exists():continue
     ep=h["data"][state["demo_id"]];ss,aa=np.asarray(ep["states"]),np.asarray(ep["actions"]);xml=ep.attrs["model_file"];xml=xml.decode() if isinstance(xml,bytes) else xml;env._env.reset();env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml,w),{},demo_generation=False));env._env.env.sim.reset();raw=env._env.regenerate_obs_from_state(ss[state["resolved_frame"]]);batch=prepare_demo_state(policy,pre,env_pre,env,raw,state["language"],aa,state["resolved_frame"]);neighbors=np.load(parent/"neighbors"/f"{state['state_id']}.npz")["actions"];anchor=action_chunk(aa,state["resolved_frame"])[0].numpy();rows,integrity=evaluate(model,pre,batch,neighbors,anchor,state);tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps({"state":state,"rows":rows,"integrity":integrity},sort_keys=True)+"\n");tmp.replace(out);print(json.dumps({"task":task,"state":i,"of":len(task_states),"rows":len(rows)}),flush=True)
  finally:env.close()


if __name__=="__main__":main()
