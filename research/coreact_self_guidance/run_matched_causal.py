"""Append-only state-level matched causal rollout runner."""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
_WS=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(_WS/'LIBERO'),str(_WS/'lerobot/src'),str(_WS)]
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_self_guidance.reference_snapshot_gate import digest, fingerprints
from research.coreact_self_guidance.timestep_sampler import sample_timestep_shift_actions, treatment_free_timestep_geometry

ARMS=("V_vanilla","N0_shift_only","W05_interpolation","W20_extrapolation")

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def atomic_json(path,payload):
 tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); tmp.replace(path)

def trace_vectors(trace):
 keys=("_x_tau","_clean_velocity","_shifted_velocity","_correction_vector","_applied_correction_vector"); out={k:[] for k in keys}
 for s in trace['step_traces']:
  for k in keys: out[k].append(s.pop(k))
 return {k:torch.stack(v) for k,v in out.items()}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--reference-artifact',type=Path,required=True); p.add_argument('--arms',default=','.join(ARMS)); a=p.parse_args(); ws=a.workspace.resolve(); art=a.artifact.resolve(); refs=a.reference_artifact.resolve(); arms=tuple(x for x in a.arms.split(',') if x)
 os.environ.setdefault('HF_HOME','/home/zjt/.cache/huggingface'); os.environ.setdefault('TRANSFORMERS_CACHE','/home/zjt/.cache/huggingface/hub'); os.environ['MUJOCO_GL']='egl'
 if json.loads((refs/'decision.json').read_text())['decision']!='REFERENCE_SNAPSHOT_GATE_PASS_READY_FOR_MATCHED_CAUSAL_ROLLOUT': raise RuntimeError('reference gate not passed')
 rows=[json.loads(x) for x in (art/'episode_manifest.jsonl').read_text().splitlines()]; groups={}
 for r in rows: groups.setdefault((r['snapshot_id'],r['noise_seed']),{})[r['arm']]=r
 if any(set(x)!=set(arms) for x in groups.values()): raise RuntimeError('manifest grouping failure')
 for d in ('episodes','geometry','invalid_units','status','logs'): (art/d).mkdir(exist_ok=True)
 config,policy,pre,post=load_policy_and_processors(ws); protocol_sha=sha(art/'protocol.lock.yaml'); ref_sha=sha(refs/'reference_gate.lock.yaml'); completed=0
 for (snapshot_id,noise_seed),specs in sorted(groups.items()):
  outputs=[art/'episodes'/f"{specs[x]['episode_id']}.json" for x in arms]
  if all(x.exists() for x in outputs): completed+=1; continue
  if any(x.exists() for x in outputs): raise RuntimeError(f'partial causal unit requires audit: {snapshot_id}/{noise_seed}')
  snapshot_file_id=snapshot_id.replace('__p','__bin')
  snap=torch.load(refs/'snapshots'/f'{snapshot_file_id}.pt',weights_only=False,map_location='cpu'); meta=snap['metadata']; prefix=snap['action_prefix'].numpy(); branch={}; records={}; sidecars={}; invalid=None
  for arm in arms:
   spec=specs[arm]; env,env_pre,env_post=make_task_env('libero_spatial',spec['task_id'],config); actions=[]; queue=[]; traces=[]; vectors=[]; replans=0; success=False; reason='horizon'
   try:
    inner=env.envs[0]; inner.init_state_id=spec['init_state_id']; obs,_=env.reset(seed=int(meta['reset_seed']))
    for x in prefix: obs,_,terminated,_,info=env.step(np.asarray(x,dtype=np.float32)[None,:])
    batch=prepare(policy,pre,env_pre,obs,meta['instruction']); fp=fingerprints(env,obs,batch,list(prefix)); base_seed=int(noise_seed)*1000; gen=torch.Generator(device=batch['state'].device).manual_seed(base_seed); noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype)
    geometry=treatment_free_timestep_geometry(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,shift=0.2)
    branch[arm]={'fingerprints':fp,'reference_equal':fp==meta['reference_fingerprints'],'noise':digest(noise),'geometry':{k:digest(v) for k,v in geometry.items()}}
    remaining=max(0,280-int(meta['resolved_control_step']))
    for control in range(remaining):
     if not queue:
      if replans>0:
       batch=prepare(policy,pre,env_pre,obs,meta['instruction']); gen=torch.Generator(device=batch['state'].device).manual_seed(base_seed+replans); noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype)
      with torch.inference_mode():
       if arm=='V_vanilla': chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise); trace=None
       else:
        w={'N0_shift_only':0.0,'W05_interpolation':0.5,'W20_extrapolation':2.0}[arm]; chunk,trace=sample_timestep_shift_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,shift=0.2,w=w,pure_negative=arm=='N0_shift_only',record_correction_vectors=True); vectors.append(trace_vectors(trace)); traces.append(trace)
      if not torch.isfinite(chunk).all(): raise RuntimeError('nonfinite chunk')
      queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)]; replans+=1
     ma=queue.pop(0); legal=env_post({'action':post(ma)})['action']; obs,_,terminated,_,info=env.step(legal.detach().cpu().numpy()); actions.append(ma[0].detach().float().cpu()); success=bool(vector_info_value(info,'is_success'))
     if success: reason='success'; break
     if bool(terminated[0]): reason='terminated'; break
   finally: env.close()
   records[arm]={**spec,'status':'complete','success':success,'termination_reason':reason,'branch_control_steps':len(actions),'total_control_step':int(meta['resolved_control_step'])+len(actions),'replans':replans,'all_actions_finite':bool(all(torch.isfinite(x).all() for x in actions)),'protocol_sha256':protocol_sha,'reference_gate_sha256':ref_sha,'branch_point':branch[arm]}
   sidecars[arm]={'selector_geometry':geometry,'trajectory_vectors':vectors,'traces':traces}
  # Exact branch-point equality across all four independently replayed arms.
  for field in ('fingerprints','noise','geometry'):
   if len({json.dumps(branch[a][field],sort_keys=True) for a in arms})!=1: invalid=field; break
  if invalid is None and not all(branch[a]['reference_equal'] for a in arms): invalid='reference_fingerprint'
  if invalid:
   atomic_json(art/'invalid_units'/f'{snapshot_id}__noise{noise_seed}.json',{'snapshot_id':snapshot_id,'noise_seed':noise_seed,'first_mismatch':invalid,'branch':branch}); raise RuntimeError(f'causal unit mismatch {snapshot_id}/{noise_seed}: {invalid}')
  for arm in arms:
   side=art/'geometry'/f"{specs[arm]['episode_id']}.pt"; torch.save(sidecars[arm],side); records[arm]['geometry_path']=str(side.relative_to(art)); records[arm]['geometry_sha256']=sha(side); atomic_json(art/'episodes'/f"{specs[arm]['episode_id']}.json",records[arm])
  completed+=1; print(json.dumps({'causal_units_complete':completed,'planned':500,'snapshot_id':snapshot_id,'noise_seed':noise_seed}),flush=True)
 (art/'status'/'rollout.complete').write_text(f"{len(groups)}/{len(groups)} causal units; {len(groups)*len(arms)}/{len(groups)*len(arms)} episodes\n")

if __name__=='__main__': main()
