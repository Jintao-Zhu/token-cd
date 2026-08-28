#!/usr/bin/env python3
"""Append-only six-arm Task-4 early trajectory accumulation runner."""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
WS=Path(__file__).resolve().parents[2];sys.path[:0]=[str(WS/'LIBERO'),str(WS/'lerobot/src'),str(WS)]
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_self_guidance.reference_snapshot_gate import digest,fingerprints
from research.coreact_self_guidance.timestep_sampler import sample_timestep_shift_actions,treatment_free_timestep_geometry
ARMS=('V_vanilla','W2_full','W2_early25','W2_start25','W2_early40','W2_start40')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(path,x):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(path)
def vectors(trace):
 keys=('_x_tau','_clean_velocity','_shifted_velocity','_correction_vector','_applied_correction_vector');out={k:[] for k in keys}
 for s in trace['step_traces']:
  for k in keys:out[k].append(s.pop(k))
 return {k:torch.stack(v) for k,v in out.items()}
def use_w2(arm,step,c25,c40):
 return arm=='W2_full' or (arm=='W2_early25' and step<c25) or (arm=='W2_start25' and step>=c25) or (arm=='W2_early40' and step<c40) or (arm=='W2_start40' and step>=c40)
def cutoff(L,p):return max(10,min(270,10*math.floor((p*L)/10)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--max-new-pairs',type=int);a=ap.parse_args();ws=a.workspace.resolve();art=a.artifact.resolve()
 os.environ.setdefault('HF_HOME',str(ws/'task1/.hf-cache'));os.environ.setdefault('TRANSFORMERS_CACHE',str(ws/'task1/.hf-cache/hub'));os.environ['MUJOCO_GL']='egl'
 specs=[json.loads(x) for x in (art/'episode_manifest.jsonl').read_text().splitlines()];pairs=defaultdict(dict)
 for x in specs:pairs[x['pair_id']][x['arm']]=x
 if len(pairs)!=50 or any(set(x)!=set(ARMS) for x in pairs.values()):raise RuntimeError('manifest')
 config,policy,pre,post=load_policy_and_processors(ws);protocol_sha=sha(art/'protocol.lock.yaml')
 from libero.libero import benchmark
 language=benchmark.get_benchmark_dict()['libero_spatial']().get_task(4).language;done=0
 for pair_id,ps in sorted(pairs.items()):
  outs=[art/'episodes'/f"{ps[x]['episode_id']}.json" for x in ARMS]
  if all(x.exists() for x in outs):done+=1;continue
  if any(x.exists() for x in outs):raise RuntimeError(f'partial pair {pair_id}')
  branch={};records={};sides={};vanilla_L=None;c25=c40=None
  for arm in ARMS:
   spec=ps[arm];env,env_pre,env_post=make_task_env('libero_spatial',4,config);queue=[];actions=[];replans=0;success=False;reason='horizon';mode_by_replan=[];traces=[];vec=[]
   try:
    inner=env.envs[0];inner.init_state_id=spec['init_state_id'];obs,_=env.reset(seed=spec['reset_seed']);batch=prepare(policy,pre,env_pre,obs,language);gen=torch.Generator(device=batch['state'].device).manual_seed(spec['noise_seed_base']);noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype);geom=treatment_free_timestep_geometry(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,shift=0.2);branch[arm]={'fingerprints':fingerprints(env,obs,batch,[]),'noise':digest(noise),'geometry':{k:digest(v) for k,v in geom.items()}}
    for control in range(280):
     if not queue:
      if replans>0:
       batch=prepare(policy,pre,env_pre,obs,language);gen=torch.Generator(device=batch['state'].device).manual_seed(spec['noise_seed_base']+replans);noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype)
      active=False if arm=='V_vanilla' else use_w2(arm,control,c25,c40);mode_by_replan.append({'replan':replans,'control_step':control,'operator':'W2' if active else 'Vanilla'})
      with torch.inference_mode():
       if active:
        chunk,trace=sample_timestep_shift_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,shift=0.2,w=2.0,pure_negative=False,record_correction_vectors=True);vec.append(vectors(trace));traces.append(trace)
       else:chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise)
      if not torch.isfinite(chunk).all():raise RuntimeError('nonfinite chunk')
      queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)];replans+=1
     ma=queue.pop(0);legal=env_post({'action':post(ma)})['action'];obs,_,terminated,_,info=env.step(legal.detach().cpu().numpy());actions.append(ma[0].detach().float().cpu());success=bool(vector_info_value(info,'is_success'))
     if success:reason='success';break
     if bool(terminated[0]):reason='terminated';break
   finally:env.close()
   if arm=='V_vanilla':vanilla_L=len(actions);c25=cutoff(vanilla_L,.25);c40=cutoff(vanilla_L,.40)
   records[arm]={**spec,'status':'complete','success':success,'termination_reason':reason,'control_steps':len(actions),'replans':replans,'vanilla_reference_length':vanilla_L if arm=='V_vanilla' else None,'cutoff25':c25,'cutoff40':c40,'all_actions_finite':bool(actions and all(torch.isfinite(x).all() for x in actions)),'initial_gate':branch[arm],'protocol_sha256':protocol_sha,'operator_by_replan':mode_by_replan}
   sides[arm]={'initial_geometry':geom,'trajectory_vectors':vec,'traces':traces,'operator_by_replan':mode_by_replan}
  for field in ('fingerprints','noise','geometry'):
   if len({json.dumps(branch[x][field],sort_keys=True) for x in ARMS})!=1:
    atomic(art/'invalid_pairs'/f'{pair_id}.json',{'pair_id':pair_id,'mismatch':field,'branch':branch});raise RuntimeError(f'{pair_id} {field}')
  for arm in ARMS:
   records[arm]['vanilla_reference_length']=vanilla_L;records[arm]['cutoff25']=c25;records[arm]['cutoff40']=c40;side=art/'geometry'/f"{ps[arm]['episode_id']}.pt";torch.save(sides[arm],side);records[arm]['geometry_path']=str(side.relative_to(art));records[arm]['geometry_sha256']=sha(side);atomic(art/'episodes'/f"{ps[arm]['episode_id']}.json",records[arm])
  done+=1;print(json.dumps({'pairs_complete':done,'planned':50,'pair_id':pair_id,'L':vanilla_L,'cutoff25':c25,'cutoff40':c40}),flush=True)
  if a.max_new_pairs is not None and done>=a.max_new_pairs:break
 if done==50:(art/'status'/'rollout.complete').write_text('50/50 pairs; 300/300 episodes\n')
if __name__=='__main__':
 main()
