from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np,torch
from research.coreact_closed_loop.guidance import GuidanceConfig
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_closed_loop.run_pilot import read_jsonl,vector_info_value,write_json
from research.coreact_direct_delete import sample_deleted_actions

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1); a=p.parse_args(); w,art=a.workspace.resolve(),a.artifact.resolve(); rows=[r for i,r in enumerate(read_jsonl(art/'episode_manifest.jsonl')) if i%a.shard_count==a.shard_index]; cfg,policy,pre,post=load_policy_and_processors(w); means=torch.load(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu')
 for spec in rows:
  out=art/'episodes'/f"{spec['episode_id']}.json"
  if out.exists(): continue
  env,ep,eo=make_task_env(spec['suite'],spec['task_id'],cfg); actions=[]; traces=[]; replans=0; lat=[]; success=False
  try:
   inner=env.envs[0]; inner.init_state_id=spec['init_state_id']; obs,_=env.reset(seed=spec['reset_seed']); queue=[]
   for step in range(280):
    if not queue:
     x=prepare(policy,pre,ep,obs,spec['language']); gen=torch.Generator(device=x['state'].device).manual_seed(spec['action_noise_seed']*1000+replans); noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=gen,device=x['state'].device,dtype=x['state'].dtype); t=time.perf_counter()
     with torch.inference_mode():
      if spec['condition']=='vanilla': chunk=policy.model.sample_actions(x['images'],x['image_masks'],x['lang_tokens'],x['lang_masks'],x['state'],noise=noise); trace=None
      else: chunk,trace=sample_deleted_actions(policy.model,x['images'],x['image_masks'],x['lang_tokens'],x['lang_masks'],x['state'],noise,config=GuidanceConfig(group_count=8,guidance_scale=.5,trust_region_kappa=.25,action_dim=7,num_steps=10,direction='away'),selection_seed=spec['selection_seed']*1000+replans,mode='mask_only' if spec['condition']=='top8_mask_only' else spec['condition'])
     lat.append(time.perf_counter()-t); queue.extend(chunk[:,:10,:7].transpose(0,1)); replans+=1
     if trace is not None: traces.append(trace)
    action=queue.pop(0); finite=bool(torch.isfinite(action).all());
    if not finite: raise RuntimeError('nonfinite action')
    legal=eo({'action':post(action)})['action']; step_result=env.step(legal.detach().cpu().numpy()); obs,_,terminated,_,info=step_result if len(step_result)==5 else (*step_result[:3], step_result[3], step_result[3]); actions.append(action[0].detach().float().cpu().tolist()); success=bool(vector_info_value(info,'is_success'))
    if bool(terminated[0]) or success: break
  finally: env.close()
  arr=np.asarray(actions); rec={**spec,'success':success,'control_steps':len(actions),'replans':replans,'mask_replans':0 if spec['condition']=='vanilla' else len(traces),'all_replans_masked':spec['condition']=='vanilla' or len(traces)==replans,'all_actions_finite':bool(np.isfinite(arr).all()),'action_total_variation':float(np.abs(np.diff(arr,axis=0)).sum()) if len(arr)>1 else 0.,'mean_replan_latency_s':float(np.mean(lat)) if lat else 0.,'replan_traces':traces}
  if not rec['all_replans_masked'] or not rec['all_actions_finite']: raise RuntimeError('integrity failure')
  write_json(out,rec); print(json.dumps({'episode_id':spec['episode_id'],'complete':True}),flush=True)
if __name__=='__main__': main()
