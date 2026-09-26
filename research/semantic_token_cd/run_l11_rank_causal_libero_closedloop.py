#!/usr/bin/env python3
"""Paired LIBERO-90 OSMesa closed-loop arms for L11 rank causal study."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
ART=ROOT/'artifacts/l11_rank_causal_effect_study_v1_20260926'
CHECKPOINT=Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
CODE_DIR=ROOT/'third_party/openvla/prismatic/extern/hf'
TASK_IDS=(3,10,49,72,73)
ARMS=(('vanilla',None,False),('b1',0,False),('b4',3,False),('b8',7,False),('random32',None,True))

def atomic(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def random_seed(task,episode,step):
 key=f'LIBERO-90\0{task}\0{int(episode)}\0{int(step)}\0Random32'.encode()
 return int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
def parse_ids(spec):
 out=[]
 for part in spec.split(','):
  part=part.strip()
  if '-' in part:
   lo,hi=map(int,part.split('-',1));out.extend(range(lo,hi+1))
  elif part:out.append(int(part))
 return sorted(set(out))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--task-id',type=int,required=True,choices=TASK_IDS);ap.add_argument('--episodes',default='0-29');ap.add_argument('--artifact',type=Path,default=ART/'closedloop_libero');ap.add_argument('--smoke',action='store_true');ap.add_argument('--max-steps',type=int,default=400);a=ap.parse_args()
 if a.gpu not in (1,2,3):raise ValueError('only GPUs 1,2,3 are authorized')
 os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa';os.environ.pop('MUJOCO_EGL_DEVICE_ID',None);os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
 torch.set_num_threads(1)
 from research.semantic_token_cd.libero_matched_rollout import predict_matched
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 from research.ar_token_counterfactual.libero_runtime import build_prompt,load_policy,predict_action,prepare_agentview,prepare_env_action,set_determinism
 episodes=parse_ids(a.episodes)
 if a.smoke:episodes=episodes[:1]
 suite=benchmark.get_benchmark_dict()['libero_90']();task=suite.get_task(a.task_id);task_name=task.name
 if not episodes or max(episodes)>=len(suite.get_task_init_states(a.task_id)):raise ValueError(f'episode id out of range for {task_name}')
 bddl=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
 init_states=suite.get_task_init_states(a.task_id)
 set_determinism(7)
 model,processor=load_policy(CHECKPOINT,CODE_DIR,device='cuda:0',dataset_statistics_path=CHECKPOINT/'dataset_statistics.json',unnorm_key='libero_90_no_noops')
 root=a.artifact.resolve()/task_name;root.mkdir(parents=True,exist_ok=True)
 for ep in episodes:
  paired={}
  for arm,rank_bin,is_random in ARMS:
   out=root/arm/f'episode_{ep:03d}.json';arrpath=root/arm/f'episode_{ep:03d}_actions.npz'
   if out.exists() and arrpath.exists():paired[arm]=json.loads(out.read_text());continue
   env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256)
   case_seed=20_261_001+a.task_id*100_000+ep
   set_determinism(case_seed);env.seed(0);env.reset();obs=env.set_init_state(init_states[ep])
   for _ in range(10):
    obs,_,done,_=env.step([0,0,0,0,0,0,-1])
    if done:raise RuntimeError(f'{task_name}:{ep}: terminated during settle')
   state=np.asarray(env.get_sim_state()).copy();state_sha=sha(state);rgb,image=prepare_agentview(obs);rgb_sha=sha(np.asarray(image.convert('RGB'),dtype=np.uint8))
   trace=[];actions=[];done=False
   for step in range(a.max_steps):
    _rgb,image=prepare_agentview(obs)
    if arm=='vanilla':raw=predict_action(model,processor,image,task.language,unnorm_key='libero_90_no_noops');meta=None
    else:
     raw,meta=predict_matched(model,processor,image,task.language,entity_mode='source_target_libero90',query_mode='instruction_only',attention_layers=(11,),attention_heads=(),destination_weight=0.0,lambda_scale=1.0,unnorm_key='libero_90_no_noops',position_mode='attention',selector_transform='identity',fixed_m=32,rank_bin=rank_bin,random_seed=random_seed(task_name,ep,step) if is_random else None)
     if int(meta['m_used'])!=32:raise RuntimeError(f'non-K32 intervention {task_name}:{ep}:{step}: {meta["m_used"]}')
     if len(meta['selected_token_ids'])!=32 or len(set(meta['selected_token_ids']))!=32:raise RuntimeError('selected token count/uniqueness failure')
     trace.append({'step':step,'selector_source':meta['selector_source'],'selected_token_ids':meta['selected_token_ids'],'attention_sha256':meta['attention_sha256'],'feature_perturbation_norm':meta['feature_perturbation_norm'],'guided_changed_dims':meta['guided_changed_dims'],'lambda':meta['lambda'],'non_target_bit_identical':meta['non_target_bit_identical'],'reconstruction_finite':meta['reconstruction_finite']})
    action=prepare_env_action(raw)
    if action.shape!=(7,) or not np.isfinite(action).all():raise FloatingPointError(f'invalid action {task_name}:{ep}:{step}')
    obs,_,done,_=env.step(action.tolist());actions.append(action.copy())
    if done:break
   success=bool(env.check_success());action_array=np.asarray(actions,dtype=np.float32)
   payload={'protocol_id':'L11_RANK_CAUSAL_EFFECT_CLOSEDLOOP_V1','benchmark':'LIBERO-90','task_id':a.task_id,'task':task_name,'instruction':task.language,'episode':ep,'init_state_id':ep,'env_seed':0,'case_seed':20_261_001+a.task_id*100_000+ep,'settle_steps':10,'max_policy_steps':a.max_steps,'arm':arm,'selector':'Vanilla' if arm=='vanilla' else ('Random32' if is_random else f'L11-B{int(arm[1:])}'),'rank_bin':int(arm[1:]) if arm.startswith('b') else None,'selected_k':0 if arm=='vanilla' else 32,'success':success,'steps':len(actions),'done':bool(done),'initial_state_sha256':state_sha,'initial_rgb_sha256_osmesa':rgb_sha,'renderer':'OSMesa CPU','inference_gpu':a.gpu,'checkpoint':str(CHECKPOINT),'lambda':0.5 if arm!='vanilla' else 0.0,'harmonic':'16x16 four-neighbor Dirichlet beta=0','trace':trace}
   arrpath.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(arrpath,executed_actions=action_array);atomic(out,payload);paired[arm]=payload
   env.close()
   print(json.dumps({'task_id':a.task_id,'episode':ep,'arm':arm,'success':success,'steps':len(actions)},sort_keys=True),flush=True)
  pair_hashes={(x['initial_state_sha256'],x['initial_rgb_sha256_osmesa']) for x in paired.values()}
  if set(paired)!=set(x[0] for x in ARMS) or len(pair_hashes)!=1:raise RuntimeError(f'paired-arm start-state mismatch {task_name}:{ep}')
if __name__=='__main__':main()
