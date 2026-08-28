from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

WS=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(WS/'LIBERO'),str(WS/'lerobot/src'),str(WS)]
from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_self_guidance.reference_snapshot_gate import digest, fingerprints

ARMS=('V_vanilla','T_attention8_one_shot','A_attention8_one_shot')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def write_json(path,payload):
 tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
def json_safe(value):
 if isinstance(value,torch.Tensor): return {'tensor_sha256':tensor_sha256(value),'shape':list(value.shape),'dtype':str(value.dtype)}
 if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
 return value

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--reference-artifact',type=Path,required=True); p.add_argument('--dry-run',action='store_true'); p.add_argument('--snapshot',default='task04__init00__bin01'); p.add_argument('--noise-seed',type=int,default=2026080901)
 a=p.parse_args(); ws=a.workspace.resolve(); art=a.artifact.resolve(); ref=a.reference_artifact.resolve(); os.environ.setdefault('HF_HOME','/home/zjt/.cache/huggingface'); os.environ['MUJOCO_GL']='egl'
 manifest=[json.loads(x) for x in (art/'episode_manifest.jsonl').read_text().splitlines()]
 groups=defaultdict(dict)
 for r in manifest: groups[(r['snapshot_id'],r['noise_seed'])][r['arm']]=r
 if set(groups[(a.snapshot,a.noise_seed)]) != set(ARMS): raise RuntimeError('missing dry-run arms')
 config,policy,pre,post=load_policy_and_processors(ws); means=torch.load(ws/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu')['visual_position_mean']
 toward=GuidanceConfig(selection='top',group_count=8,guidance_scale=.5,trust_region_kappa=.25,action_dim=7,num_steps=10,direction='toward',record_correction_vectors=True)
 away=GuidanceConfig(selection='top',group_count=8,guidance_scale=.5,trust_region_kappa=.25,action_dim=7,num_steps=10,direction='away',record_correction_vectors=True)
 snap=torch.load(ref/'snapshots'/(a.snapshot.replace('__p','__bin')+'.pt'),weights_only=False,map_location='cpu'); meta=snap['metadata']; prefix=snap['action_prefix'].numpy(); results={}; traces={}
 for arm in ARMS:
  spec=groups[(a.snapshot,a.noise_seed)][arm]; env,envpre,envpost=make_task_env('libero_spatial',spec['task_id'],config); obs=None; success=False; reason='horizon'; actions=[]; queue=[]; replan=0
  try:
   env.envs[0].init_state_id=spec['init_state_id']; obs,_=env.reset(seed=int(meta['reset_seed']))
   for x in prefix: obs,_,term,_,info=env.step(np.asarray(x,dtype=np.float32)[None,:])
   batch=prepare(policy,pre,envpre,obs,meta['instruction']); fp=fingerprints(env,obs,batch,list(prefix)); base_seed=int(a.noise_seed)*1000; gen=torch.Generator(device=batch['state'].device).manual_seed(base_seed); noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype)
   trace={'method':'vanilla','selected_indices':[],'changed_indices':[],'noise_sha256':tensor_sha256(noise),'step_traces':[]}
   with torch.inference_mode():
    if arm=='T_attention8_one_shot': chunk,trace=sample_coreact_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,means,config=toward,selection_seed=base_seed)
    elif arm=='A_attention8_one_shot': chunk,trace=sample_coreact_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,means,config=away,selection_seed=base_seed)
    else: chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise)
   if not torch.isfinite(chunk).all(): raise RuntimeError('nonfinite chunk')
   traces[arm]=trace; first_noise=tensor_sha256(noise); queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)]; replan=1
   remaining=max(0,280-int(meta['resolved_control_step']))
   for control in range(remaining):
    if not queue:
     batch=prepare(policy,pre,envpre,obs,meta['instruction']); gen=torch.Generator(device=batch['state'].device).manual_seed(base_seed+replan); noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype)
     with torch.inference_mode(): chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise)
     queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)]; replan+=1
    action=queue.pop(0); legal=envpost({'action':post(action)})['action']; obs,_,term,_,info=env.step(legal.detach().cpu().numpy()); actions.append(action[0].detach().float().cpu()); success=bool(vector_info_value(info,'is_success'))
    if success: reason='success'; break
    if bool(term[0]): reason='terminated'; break
  finally: env.close()
  results[arm]={'status':'complete','success':success,'termination_reason':reason,'replans':replan,'branch_fingerprint':fp,'reference_equal':fp==meta['reference_fingerprints'],'initial_noise_sha256':first_noise,'trace':json_safe(trace),'spec':spec}
 # Treatment integrity: branch fingerprints and noise must match; T/A geometry must be symmetric.
 if len({json.dumps(results[x]['branch_fingerprint'],sort_keys=True) for x in ARMS}) != 1: raise RuntimeError('CAUSAL_DRYRUN_FAILED: branch fingerprint mismatch')
 if len({results[x]['initial_noise_sha256'] for x in ARMS}) != 1: raise RuntimeError('CAUSAL_DRYRUN_FAILED: noise mismatch')
 t=results['T_attention8_one_shot']['trace']; aa=results['A_attention8_one_shot']['trace']
 if t.get('selected_indices') != aa.get('selected_indices') or t.get('changed_indices') != aa.get('changed_indices'): raise RuntimeError('CAUSAL_DRYRUN_FAILED: selector mismatch')
 # Only the first flow step is pre-treatment. Later x_tau values legitimately
 # diverge because the two signed operators have already been applied.
 if not t.get('step_traces') or not aa.get('step_traces'):
  raise RuntimeError('CAUSAL_DRYRUN_FAILED: missing flow trace')
 st,sa=t['step_traces'][0],aa['step_traces'][0]
 if st.get('positive_velocity_sha256') != sa.get('positive_velocity_sha256'):
  raise RuntimeError('CAUSAL_DRYRUN_FAILED: initial clean velocity mismatch')
 if st.get('raw_guidance_sha256') != sa.get('raw_guidance_sha256'):
  raise RuntimeError('CAUSAL_DRYRUN_FAILED: initial raw correction mismatch')
 if abs(float(st.get('applied_guidance_norm',0))-float(sa.get('applied_guidance_norm',0)))>1e-5:
  raise RuntimeError('CAUSAL_DRYRUN_FAILED: initial magnitude mismatch')
 out=art/'status'/'dry_run.pass.json' if a.dry_run else art/'status'/f'unit_{a.snapshot}_{a.noise_seed}.json'; write_json(out,{'decision':'CAUSAL_FOUR_ARM_DRYRUN_VALIDATED' if a.dry_run else 'UNIT_PASS','results':results})
 print(json.dumps({'decision':'CAUSAL_FOUR_ARM_DRYRUN_VALIDATED' if a.dry_run else 'UNIT_PASS','snapshot':a.snapshot,'noise_seed':a.noise_seed},indent=2))

if __name__=='__main__': main()
