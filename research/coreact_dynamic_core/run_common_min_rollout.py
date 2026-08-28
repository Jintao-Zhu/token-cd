#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys
from collections import defaultdict
from pathlib import Path
import torch,yaml
WS=Path(__file__).resolve().parents[2];sys.path[:0]=[str(WS/'LIBERO'),str(WS/'lerobot/src'),str(WS)]
from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_self_guidance.reference_snapshot_gate import fingerprints
from research.coreact_dynamic_core.dynamic_guidance import sample_common_min_actions
ARMS=('vanilla','random8_common','attention8_common','instability8_common');MAP={'random8_common':'random','attention8_common':'attention','instability8_common':'instability'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--max-new-pairs',type=int);a=p.parse_args();w=a.workspace.resolve();art=a.artifact.resolve();protocol=yaml.safe_load((art/'protocol.lock.yaml').read_text());cap=protocol['operator']['magnitude_cap'];os.environ.setdefault('HF_HOME',str(w/'task1/.hf-cache'));os.environ.setdefault('TRANSFORMERS_CACHE',str(w/'task1/.hf-cache/hub'));os.environ['MUJOCO_GL']='egl'
 specs=[json.loads(x) for x in (art/'episode_manifest.jsonl').read_text().splitlines()];pairs=defaultdict(dict)
 for x in specs:pairs[x['pair_id']][x['arm']]=x
 if len(pairs)!=50 or any(set(x)!=set(ARMS) for x in pairs.values()):raise RuntimeError('manifest')
 cfg,policy,pre,post=load_policy_and_processors(w);means=torch.load(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu')['visual_position_mean'];protocol_sha=sha(art/'protocol.lock.yaml');from libero.libero import benchmark;suite=benchmark.get_benchmark_dict()['libero_spatial']();complete=new=0
 for pair_id,ps in sorted(pairs.items()):
  outs=[art/'episodes'/f"{ps[x]['episode_id']}.json" for x in ARMS]
  if all(x.exists() for x in outs):complete+=1;continue
  if any(x.exists() for x in outs):raise RuntimeError(f'partial pair {pair_id}')
  initial={};real_reset={};records={};canonical_observation=None
  for arm in ARMS:
   s=ps[arm];language=suite.get_task(4).language;env,env_pre,env_post=make_task_env('libero_spatial',4,cfg);queue=[];actions=[];traces=[];replans=0;success=False;reason='horizon';noise_hashes=[]
   try:
    env.envs[0].init_state_id=s['init_state_id'];real_obs,_=env.reset(seed=s['reset_seed']);real_batch=prepare(policy,pre,env_pre,real_obs,language);real_reset[arm]=fingerprints(env,real_obs,real_batch,[])
    if canonical_observation is None:canonical_observation=copy.deepcopy(real_obs)
    obs=copy.deepcopy(canonical_observation);first_fp=first_clean=None
    for _ in range(280):
     if not queue:
      batch=prepare(policy,pre,env_pre,obs,language);g=torch.Generator(device=batch['state'].device).manual_seed(s['noise_seed_base']+replans);noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=g,device=batch['state'].device,dtype=batch['state'].dtype);noise_hashes.append(tensor_sha256(noise))
      if first_fp is None:first_fp=fingerprints(env,obs,batch,[])
      with torch.inference_mode():
       if arm=='vanilla':chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise);clean_hash=tensor_sha256(chunk)
       else:
        chunks,joint=sample_common_min_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,means,selection_seed=s['selection_seed_base']+replans,magnitude_cap=cap);name=MAP[arm];chunk=chunks[name];trace=joint[name];clean_hash=trace['clean_flow_final_sha256'];traces.append({'replan':replans,**trace})
      if first_clean is None:first_clean=clean_hash
      if not torch.isfinite(chunk).all():raise RuntimeError('nonfinite chunk')
      queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)];replans+=1
     model_action=queue.pop(0);legal=env_post({'action':post(model_action)})['action'];obs,_,terminated,_,info=env.step(legal.detach().cpu().numpy());actions.append(model_action[0].detach().float().cpu());success=bool(vector_info_value(info,'is_success'))
     if success:reason='success';break
     if bool(terminated[0]):reason='terminated';break
   finally:env.close()
   initial[arm]={'fingerprints':first_fp,'noise_sha256':noise_hashes[0],'clean_chunk_sha256':first_clean};records[arm]={**s,'status':'complete','success':success,'termination_reason':reason,'control_steps':len(actions),'replans':replans,'all_actions_finite':bool(actions and all(torch.isfinite(x).all() for x in actions)),'initial_gate':initial[arm],'noise_sha256_by_replan':noise_hashes,'replan_traces':traces,'protocol_sha256':protocol_sha}
  for field in ('simulator_state','qpos','qvel','object_pose','robot_observation'):
   if len({real_reset[x][field] for x in ARMS})!=1:atomic(art/'invalid_pairs'/f'{pair_id}.json',{'pair_id':pair_id,'mismatch':f'reset_{field}','real_reset':real_reset,'initial':initial});raise RuntimeError(f'{pair_id} reset_{field}')
  for field in ('fingerprints','noise_sha256','clean_chunk_sha256'):
   if len({json.dumps(initial[x][field],sort_keys=True) for x in ARMS})!=1:atomic(art/'invalid_pairs'/f'{pair_id}.json',{'pair_id':pair_id,'mismatch':field,'real_reset':real_reset,'initial':initial});raise RuntimeError(f'{pair_id} {field}')
  raw_mismatch={field:len({real_reset[x][field] for x in ARMS})!=1 for field in ('camera1','camera2','full_observation','preprocessing')}
  for arm in ARMS:records[arm]['real_reset_fingerprints']=real_reset[arm];records[arm]['raw_reset_mismatch_flags']=raw_mismatch;records[arm]['canonical_first_observation_arm']='vanilla';atomic(art/'episodes'/f"{ps[arm]['episode_id']}.json",records[arm])
  complete+=1;new+=1;print(json.dumps({'pairs_complete':complete,'planned':50,'pair_id':pair_id}),flush=True)
  if a.max_new_pairs is not None and new>=a.max_new_pairs:break
 if complete==50:(art/'status'/'rollout.complete').write_text('50/50 pairs; 200/200 episodes\n')
if __name__=='__main__':main()
