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
from research.coreact_quality_negative_branch.quality_branch import velocity_from_embeddings
from research.coreact_region.segmented_runtime import make_segmented_env
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_why_ag_fails.run_manifold_diagnosis import normalize_candidates
from research.coreact_why_ag_fails.run_self_weak_slg import metrics

STEPS=(0,1,8,9);TIMES=tuple(round(1.-.1*i,1) for i in range(10))
@torch.no_grad()
def evaluate(model,pre,batch,raw_candidates,raw_anchor,state):
 actions=batch['actions'];candidates=normalize_candidates(pre,raw_candidates,actions.device,actions.dtype,actions.shape[-1]);anchor=normalize_candidates(pre,raw_anchor[None],actions.device,actions.dtype,actions.shape[-1]);norm=float((anchor-actions).abs().max());
 if norm>=1e-6:raise RuntimeError(f'normalization parity {norm}')
 noises=torch.cat([torch.randn(actions.shape,generator=torch.Generator(device=actions.device).manual_seed(seed),device=actions.device,dtype=actions.dtype) for seed in state['noise_seeds']]);actions3=actions.expand(3,-1,-1);valid=~batch['action_is_pad'][0];prefix,pad,att=model.embed_prefix(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],state=batch['state']);scale=torch.ones(3,16,device=actions.device,dtype=actions.dtype);scale[:,-1]=0;rows=[];parity=0.
 for step in STEPS:
  tau=TIMES[step];time=torch.full((3,),tau,device=actions.device,dtype=actions.dtype);x_t,_=exact_training_pair(actions3,noises,time);strong=velocity_from_embeddings(model,prefix.expand(3,-1,-1),pad.expand(3,-1),att.expand(3,-1),x_t,time)[:,:,:7];weak=velocity_from_embeddings(model,prefix.expand(3,-1,-1),pad.expand(3,-1),att.expand(3,-1),x_t,time,expert_residual_scales=scale)[:,:,:7]
  if step==0:
   same=velocity_from_embeddings(model,prefix.expand(3,-1,-1),pad.expand(3,-1),att.expand(3,-1),x_t,time,expert_residual_scales=torch.ones_like(scale))[:,:,:7];parity=float((same-strong).abs().max())
  for ni in range(3):rows.append({'candidate':'W1_skip_last_1','flow_step':step,'timestep':tau,'noise_ordinal':ni,**metrics(strong[ni],weak[ni],candidates,x_t[ni,:,:7],tau,valid)})
 return rows,{'finite':all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,float)),'rows':len(rows),'normalization_max_abs':norm,'all_ones_strong_parity_max_abs':parity,'scale_contract':scale[0].cpu().tolist()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--task-ids',type=int,nargs='+',default=list(range(10)));p.add_argument('--max-states-per-task',type=int);a=p.parse_args();w,art=a.workspace.resolve(),a.artifact.resolve();root=art/'confirmation_manifold';states=[json.loads(x) for x in (root/'state_manifest.jsonl').read_text().splitlines() if json.loads(x)['task_id'] in a.task_ids]
 if a.max_states_per_task is not None:states=[r for t in a.task_ids for r in [x for x in states if x['task_id']==t][:a.max_states_per_task]]
 source=w/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522';os.chdir(w/'LIBERO');config,policy,pre,_=load_policy(source/'training_run/trajectory/checkpoints/015000/pretrained_model');model=policy.model
 for task in a.task_ids:
  ts=[x for x in states if x['task_id']==task]
  if not ts:continue
  env_pre,_=make_env_pre_post_processors(env_cfg=env_config('libero_spatial',task),policy_cfg=config);env=make_segmented_env('libero_spatial',task)
  try:
   with h5py.File(ts[0]['demo_path'],'r') as h:
    for i,state in enumerate(ts,1):
     out=art/'confirmation'/f"{state['state_id']}.json"
     if out.exists():continue
     ep=h['data'][state['demo_id']];ss,aa=np.asarray(ep['states']),np.asarray(ep['actions']);xml=ep.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else xml;env._env.reset();env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml,w),{},demo_generation=False));env._env.env.sim.reset();raw=env._env.regenerate_obs_from_state(ss[state['resolved_frame']]);batch=prepare_demo_state(policy,pre,env_pre,env,raw,state['language'],aa,state['resolved_frame']);neighbors=np.load(root/'neighbors'/f"{state['state_id']}.npz")['actions'];anchor=action_chunk(aa,state['resolved_frame'])[0].numpy();rows,integrity=evaluate(model,pre,batch,neighbors,anchor,state);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'rows':rows,'integrity':integrity},sort_keys=True)+'\n');tmp.replace(out);print(json.dumps({'task':task,'state':i,'of':len(ts),'rows':len(rows)}),flush=True)
  finally:env.close()
if __name__=='__main__':main()
