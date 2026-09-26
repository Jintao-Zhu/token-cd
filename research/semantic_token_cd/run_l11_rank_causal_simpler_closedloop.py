#!/usr/bin/env python3
"""Paired SIMPLER RT0 closed-loop arms for the L11 rank causal study."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pickle
from pathlib import Path
import numpy as np

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
ART=ROOT/'artifacts/l11_rank_causal_effect_study_v1_20260926'
ARMS=(('vanilla',None,None),('b1',0,None),('b4',3,None),('b8',7,None),('random32',None,'random'))

def atomic(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--task',required=True); ap.add_argument('--seeds',default='100-129'); ap.add_argument('--artifact',type=Path,default=ART/'closedloop_simpler'); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
 if a.gpu not in (1,2,3): raise ValueError('only GPUs 1,2,3 are authorized')
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu); os.environ['TOKENIZERS_PARALLELISM']='false'; os.environ.setdefault('HF_HUB_OFFLINE','1')
 import sys
 PCD=Path('/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source')
 for p in (ROOT/'task1/shim_site',ROOT,PCD):
  if str(p) not in sys.path: sys.path.insert(0,str(p))
 import torch
 torch.set_num_threads(1)
 from properties import get_policy_config
 from simpler_env.policies.openvla.openvla_model import OpenVLAInference
 from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
 from research.semantic_token_cd.distractor_rollout import restore_snapshot,snapshot_sha,jsonable
 from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
 from research.semantic_token_cd.prompt_attn_shr_rollout import _init_common
 from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX
 from research.semantic_token_cd.spatial_grid_rollout import run_episode
 from research.semantic_token_cd.replay_simpler_l11_rank_effects import make_rt_depth0_environment,CANONICAL,TASKS
 if a.task not in TASKS: raise ValueError(a.task)
 seeds=[]
 for part in a.seeds.split(','):
  part=part.strip()
  if '-' in part:
   lo,hi=map(int,part.split('-',1));seeds.extend(range(lo,hi+1))
  elif part:seeds.append(int(part))
 seeds=sorted(set(seeds))
 if a.smoke:seeds=seeds[:1]
 ckpt=str(PCD/'pretrained/openvla-7b'); cfg=get_policy_config('openvla',ckpt,a.task,{},False); base=OpenVLAInference(**cfg)
 policies={}
 vanilla=copy.copy(base); vanilla.__class__=AuditedVanillaInference; vanilla._episode_trace=[];vanilla._episode_logits=[];policies['vanilla']=vanilla
 for name,rank_bin,random_kind in ARMS[1:]:
  p=copy.copy(base);p.__class__=PromptAttentionSHRInference;_init_common(p,0.5)
  p.beta=0.0;p.selector_mode='random_matched' if random_kind else 'prompt_attention';p.attention_layers=(11,);p.selection_count=32;p.selection_rank_bin=rank_bin;p.selection_budget_schedule=None;p.selection_top_p=None;p.save_prompt_attention=False;p.task_index=TASK_INDEX[a.task]
  policies[name]=p
 env=make_rt_depth0_environment(a.task);outroot=a.artifact.resolve()/a.task;outroot.mkdir(parents=True,exist_ok=True)
 for seed in seeds:
  src=CANONICAL/'episodes'/a.task/'vanilla'/f'episode_{seed:03d}_summary.json';sp=CANONICAL/'snapshots'/a.task/f'seed_{seed:03d}.pkl'
  ref=json.loads(src.read_text());snapshot=pickle.load(sp.open('rb'));canonical=snapshot_sha(snapshot)
  if canonical!=ref['canonical_snapshot_sha256']:raise RuntimeError(f'snapshot mismatch {a.task}:{seed}')
  pair={}
  for arm,_,_ in ARMS:
   path=outroot/arm/f'seed_{seed:03d}.json'; arrpath=outroot/arm/f'seed_{seed:03d}_actions.npz'
   if path.exists() and arrpath.exists():
    d=json.loads(path.read_text());pair[arm]=d;continue
   obs,state_sha,rgb_sha=restore_snapshot(env,seed,snapshot)
   if state_sha!=ref['initial_state_sha256']:raise RuntimeError(f'initial state mismatch {a.task}:{seed}:{arm}')
   instr=env.unwrapped.get_language_instruction()
   if instr!=ref['instruction']:raise RuntimeError(f'instruction mismatch {a.task}:{seed}')
   p=policies[arm];p.task_index=TASK_INDEX[a.task];p.reset(instr,seed=seed);p._episode_trace=[];p._episode_logits=[]
   result,steps,reason,actions,jerk=run_episode(env,p,instr,obs)
   trace=jsonable(p._episode_trace)
   if arm!='vanilla':
    expected_bin=int(arm[1:])-1 if arm.startswith('b') else None
    for row in trace:
     ids=row.get('selected_token_ids',[])
     if len(ids)!=32 or len(set(ids))!=32: raise RuntimeError(f'{arm} selected non-K32 mask for {a.task}:{seed}')
     if expected_bin is not None and row.get('selection_rank_bin')!=expected_bin: raise RuntimeError(f'{arm} rank-bin metadata mismatch')
   payload={'protocol_id':'L11_RANK_CAUSAL_EFFECT_CLOSEDLOOP_V1','benchmark':'SIMPLER','task':a.task,'seed':seed,'arm':arm,'selector':'vanilla' if arm=='vanilla' else ('Random32' if arm=='random32' else f'L11-B{int(arm[1:])}'),'selected_k':0 if arm=='vanilla' else 32,'rank_bin':int(arm[1:]) if arm.startswith('b') else None,'success':bool(result['success']),'steps':int(steps),'failure_reason':reason,'runtime_action_jerk':float(jerk),'canonical_snapshot_sha256':canonical,'initial_state_sha256':state_sha,'initial_rgb_sha256_rt0':rgb_sha,'source_renderer_initial_rgb_sha256':ref['initial_rgb_sha256'],'renderer':'SAPIEN RT depth0, spp=32, denoiser=true','renderer_gpu':a.gpu,'inference_gpu':a.gpu,'lambda':0.5 if arm!='vanilla' else 0.0,'harmonic_beta':0.0,'sampled_states':len(p._episode_trace),'trace':trace}
   arrpath.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(arrpath,executed_actions=actions)
   atomic(path,payload);pair[arm]=payload
   print(json.dumps({'task':a.task,'seed':seed,'arm':arm,'success':payload['success'],'steps':steps},sort_keys=True),flush=True)
  hashes={(d['canonical_snapshot_sha256'],d['initial_state_sha256'],d['initial_rgb_sha256_rt0']) for d in pair.values()}
  if set(pair)!=set(x[0] for x in ARMS) or len(hashes)!=1:raise RuntimeError(f'paired-arm initial state mismatch {a.task}:{seed}')
 env.close()
if __name__=='__main__':main()
