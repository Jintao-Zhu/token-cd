#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,statistics,sys
from collections import defaultdict
from pathlib import Path
import torch
WS=Path(__file__).resolve().parents[2];sys.path[:0]=[str(WS/'LIBERO'),str(WS/'lerobot/src'),str(WS)]
from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_self_guidance.reference_snapshot_gate import fingerprints
from research.coreact_dynamic_core.dynamic_guidance import sample_dynamic_core_actions
ARMS=('vanilla','random8','attention8','instability8','combined8');MAP={'random8':'random','attention8':'attention','instability8':'instability','combined8':'combined'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--max-new-pairs',type=int);p.add_argument('--pair-id');a=p.parse_args();w=a.workspace.resolve();art=a.artifact.resolve();os.environ.setdefault('HF_HOME',str(w/'task1/.hf-cache'));os.environ.setdefault('TRANSFORMERS_CACHE',str(w/'task1/.hf-cache/hub'));os.environ['MUJOCO_GL']='egl'
 specs=[json.loads(x) for x in (art/'episode_manifest.jsonl').read_text().splitlines()];pairs=defaultdict(dict)
 for x in specs:pairs[x['pair_id']][x['arm']]=x
 if len(pairs)!=100 or any(set(x)!=set(ARMS) for x in pairs.values()):raise RuntimeError('manifest')
 cfg,policy,pre,post=load_policy_and_processors(w);means=torch.load(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu')['visual_position_mean'];protocol_sha=sha(art/'protocol.lock.yaml');from libero.libero import benchmark;suite=benchmark.get_benchmark_dict()['libero_spatial']();complete=0;new=0
 for pair_id,ps in sorted(pairs.items()):
  if a.pair_id is not None and pair_id!=a.pair_id:continue
  outs=[art/'episodes'/f"{ps[x]['episode_id']}.json" for x in ARMS]
  if all(x.exists() for x in outs):complete+=1;continue
  if any(x.exists() for x in outs):raise RuntimeError(f'partial pair {pair_id}')
  initial={};real_reset={};records={};canonical_observation=None
  for arm in ARMS:
   s=ps[arm];language=suite.get_task(s['task_id']).language;env,env_pre,env_post=make_task_env('libero_spatial',s['task_id'],cfg);queue=[];actions=[];traces=[];replans=0;success=False;reason='horizon';noise_hashes=[]
   try:
    inner=env.envs[0];inner.init_state_id=s['init_state_id'];real_obs,_=env.reset(seed=s['reset_seed']);real_batch=prepare(policy,pre,env_pre,real_obs,language);real_reset[arm]=fingerprints(env,real_obs,real_batch,[])
    if canonical_observation is None:canonical_observation=copy.deepcopy(real_obs)
    obs=copy.deepcopy(canonical_observation);first_fp=None;first_clean=None
    for control in range(280):
     if not queue:
      batch=prepare(policy,pre,env_pre,obs,language);gen=torch.Generator(device=batch['state'].device).manual_seed(s['noise_seed_base']+replans);noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype);noise_hashes.append(tensor_sha256(noise))
      if first_fp is None:first_fp=fingerprints(env,obs,batch,[])
      with torch.inference_mode():
       if arm=='vanilla':chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise);trace=None;clean_hash=tensor_sha256(chunk)
       else:chunk,trace=sample_dynamic_core_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,means,selector=MAP[arm],selection_seed=s['selection_seed_base']+replans);clean_hash=trace['clean_flow_final_sha256'];traces.append({'replan':replans,**trace})
      if first_clean is None:first_clean=clean_hash
      if not torch.isfinite(chunk).all():raise RuntimeError('nonfinite chunk')
      queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)];replans+=1
     ma=queue.pop(0);legal=env_post({'action':post(ma)})['action'];obs,_,terminated,_,info=env.step(legal.detach().cpu().numpy());actions.append(ma[0].detach().float().cpu());success=bool(vector_info_value(info,'is_success'))
     if success:reason='success';break
     if bool(terminated[0]):reason='terminated';break
   finally:env.close()
   initial[arm]={'fingerprints':first_fp,'noise_sha256':noise_hashes[0],'clean_chunk_sha256':first_clean};records[arm]={**s,'status':'complete','success':success,'termination_reason':reason,'control_steps':len(actions),'replans':replans,'all_actions_finite':bool(actions and all(torch.isfinite(x).all() for x in actions)),'initial_gate':initial[arm],'noise_sha256_by_replan':noise_hashes,'replan_traces':traces,'protocol_sha256':protocol_sha}
  physics_fields=('simulator_state','qpos','qvel','object_pose','robot_observation')
  for field in physics_fields:
   if len({real_reset[x][field] for x in ARMS})!=1:
    atomic(art/'invalid_pairs'/f'{pair_id}.json',{'pair_id':pair_id,'mismatch':f'reset_{field}','real_reset':real_reset,'initial':initial});raise RuntimeError(f'{pair_id} reset_{field}')
  for field in ('fingerprints','noise_sha256','clean_chunk_sha256'):
   if len({json.dumps(initial[x][field],sort_keys=True) for x in ARMS})!=1:
    atomic(art/'invalid_pairs'/f'{pair_id}.json',{'pair_id':pair_id,'mismatch':field,'real_reset':real_reset,'initial':initial});raise RuntimeError(f'{pair_id} {field}')
  raw_camera_mismatch={field:len({real_reset[x][field] for x in ARMS})!=1 for field in ('camera1','camera2','full_observation','preprocessing')}
  for arm in ARMS:records[arm]['real_reset_fingerprints']=real_reset[arm];records[arm]['raw_reset_mismatch_flags']=raw_camera_mismatch;records[arm]['canonical_first_observation_arm']='vanilla';atomic(art/'episodes'/f"{ps[arm]['episode_id']}.json",records[arm])
  complete+=1;new+=1;print(json.dumps({'pairs_complete':complete,'planned':100,'pair_id':pair_id}),flush=True)
  if a.max_new_pairs is not None and new>=a.max_new_pairs:break
 if complete==100:(art/'status'/'rollout.complete').write_text('100/100 pairs; 500/500 episodes\n')
if __name__=='__main__':main()
