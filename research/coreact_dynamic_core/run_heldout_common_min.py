from __future__ import annotations
import argparse,copy,hashlib,json,os,sys
from collections import defaultdict
from pathlib import Path
import torch
WS=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(WS/'LIBERO'),str(WS/'lerobot/src'),str(WS)]
from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_self_guidance.reference_snapshot_gate import fingerprints
from research.coreact_dynamic_core.dynamic_guidance import sample_common_min_actions
from libero.libero import benchmark
ARMS=('vanilla','random8_common','attention8_common'); MAP={'random8_common':'random','attention8_common':'attention'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def atomic(p,x):
 t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); t.replace(p)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--max-new-pairs',type=int); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1); a=p.parse_args(); w=a.workspace.resolve(); art=a.artifact.resolve(); protocol=__import__('yaml').safe_load((art/'protocol.lock.yaml').read_text()); cap=protocol['operator']['magnitude_cap']; os.environ.setdefault('MUJOCO_GL','egl')
 specs=[json.loads(x) for x in (art/'episode_manifest.jsonl').read_text().splitlines()]; pairs=defaultdict(dict)
 for s in specs:pairs[s['pair_id']][s['arm']]=s
 if any(set(v)!=set(ARMS) for v in pairs.values()): raise RuntimeError('manifest arms mismatch')
 for d in ('episodes','invalid_pairs','status','logs'):(art/d).mkdir(exist_ok=True)
 cfg,policy,pre,post=load_policy_and_processors(w); means=torch.load(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu')['visual_position_mean']; protocol_sha=sha(art/'protocol.lock.yaml'); complete=0; new=0
 selected_pairs=[item for index,item in enumerate(sorted(pairs.items())) if index % a.shard_count == a.shard_index]
 for pair_id,ps in selected_pairs:
  outs=[art/'episodes'/f"{ps[x]['episode_id']}.json" for x in ARMS]
  if all(x.exists() for x in outs):complete+=1;continue
  if any(x.exists() for x in outs):raise RuntimeError(f'partial pair {pair_id}')
  initial={}; reset_fp={}; records={}; canonical=None
  for arm in ARMS:
   s=ps[arm]; env,envpre,envpost=make_task_env('libero_spatial',s['task_id'],cfg); obs=None; queue=[]; actions=[]; traces=[]; replan=0; success=False; reason='horizon'; noise_hashes=[]
   try:
    language=benchmark.get_benchmark_dict()['libero_spatial']().get_task(s['task_id']).language; env.envs[0].init_state_id=s['init_state_id']; real_obs,_=env.reset(seed=s['reset_seed']); real_batch=prepare(policy,pre,envpre,real_obs,language); reset_fp[arm]=fingerprints(env,real_obs,real_batch,[])
    if canonical is None:canonical=copy.deepcopy(real_obs)
    obs=copy.deepcopy(canonical); first_fp=None; first_clean=None
    for _ in range(280):
     if not queue:
      batch=prepare(policy,pre,envpre,obs,language); gen=torch.Generator(device=batch['state'].device).manual_seed(s['noise_seed_base']+replan); noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype); noise_hashes.append(tensor_sha256(noise));
      if first_fp is None:first_fp=fingerprints(env,obs,batch,[])
      with torch.inference_mode():
       if arm=='vanilla': chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise); trace=None; clean_hash=tensor_sha256(chunk)
       else:
        chunks,joints=sample_common_min_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,means,selection_seed=s['selection_seed_base']+replan,magnitude_cap=cap); name=MAP[arm]; chunk=chunks[name]; trace=joints[name]; clean_hash=trace['clean_flow_final_sha256']; traces.append({'replan':replan,**trace})
      if first_clean is None:first_clean=clean_hash
      if not torch.isfinite(chunk).all():raise RuntimeError('nonfinite chunk')
      queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)]; replan+=1
     action=queue.pop(0); legal=envpost({'action':post(action)})['action']; obs,_,terminated,_,info=env.step(legal.detach().cpu().numpy()); actions.append(action[0].detach().float().cpu()); success=bool(vector_info_value(info,'is_success'))
     if success:reason='success';break
     if bool(terminated[0]):reason='terminated';break
   finally:env.close()
   initial[arm]={'fingerprints':first_fp,'noise_sha256':noise_hashes[0],'clean_chunk_sha256':first_clean}; records[arm]={**s,'status':'complete','success':success,'termination_reason':reason,'control_steps':len(actions),'replans':replan,'all_actions_finite':bool(actions and all(torch.isfinite(x).all() for x in actions)),'initial_gate':initial[arm],'noise_sha256_by_replan':noise_hashes,'replan_traces':traces,'protocol_sha256':protocol_sha,'real_reset_fingerprints':reset_fp[arm]}
  mismatch=None
  for field in ('simulator_state','qpos','qvel','object_pose','robot_observation'):
   if len({reset_fp[x][field] for x in ARMS})!=1:mismatch='reset_'+field;break
  for field in ('fingerprints','noise_sha256','clean_chunk_sha256'):
   if mismatch is None and len({json.dumps(initial[x][field],sort_keys=True) for x in ARMS})!=1:mismatch=field;break
  if mismatch:atomic(art/'invalid_pairs'/f'{pair_id}.json',{'pair_id':pair_id,'mismatch':mismatch});raise RuntimeError(f'{pair_id} {mismatch}')
  for arm in ARMS:atomic(art/'episodes'/f"{ps[arm]['episode_id']}.json",records[arm])
  complete+=1;new+=1;print(json.dumps({'pairs_complete':complete,'planned':len(pairs),'pair_id':pair_id}),flush=True)
  if a.max_new_pairs is not None and new>=a.max_new_pairs:break
 if complete==len(pairs):(art/'status'/'rollout.complete').write_text(f'{complete}/{len(pairs)} pairs; {complete*len(ARMS)}/{len(pairs)*len(ARMS)} episodes\n')
if __name__=='__main__':main()
