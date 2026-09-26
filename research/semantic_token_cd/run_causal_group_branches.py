#!/usr/bin/env python3
"""Closed-loop validation for top-group removal and size-matched controls."""
from __future__ import annotations

import argparse, copy, hashlib, json, os, pickle, sys, time, traceback
from pathlib import Path
import numpy as np

REPO=Path('/home/leju-suzhou/zjt_ws/token-cd')
PCD_SOURCE=Path('/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source')
VAN=REPO/'artifacts/vanilla_recon_shr_canonical_0_299_v2'

def sha(x):
    a=np.ascontiguousarray(x); h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes()); return h.hexdigest()

def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f: json.dump(obj,f,indent=2,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x) if isinstance(x,np.integer) else float(x) if isinstance(x,np.floating) else str(x)); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def flat(action):
    return np.concatenate([np.asarray(action['world_vector']),np.asarray(action['rot_axangle']),np.asarray(action['gripper'])])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,choices=(1,2,3),required=True); ap.add_argument('--worker-id',required=True); ap.add_argument('--worker-index',type=int,required=True); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--shader-dir',choices=('ibl','rt'),default=None); args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu); os.environ['TOKENIZERS_PARALLELISM']='false'; os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
    for p in (REPO/'task1/shim_site',REPO,PCD_SOURCE):
        if str(p) not in sys.path: sys.path.insert(0,str(p))
    import torch
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from utils import convert_numpy_or_torch_to_python,stat_final,stat_first,summarize
    from research.semantic_token_cd.distractor_rollout import restore_snapshot
    from research.semantic_token_cd.rollout_pilot import wrapped_observation
    from research.semantic_token_cd.spatial_grid_rollout import make_environment
    from research.semantic_token_cd.l11_causal_group_preflight import build_historical_policy

    out=args.output.resolve(); manifest=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    jobs=[(i,r) for i,r in enumerate(manifest) if i%args.workers==args.worker_index and r['category'] in ('Rescue','Harm')]
    current=None; env=None; policy=None; failures=0
    for jid,row in jobs:
        task=row['task']; seed=int(row['seed']); fork_dir=out/'states'/task/f'seed_{seed:03d}'; patch_path=fork_dir/'patch.json'
        if not patch_path.exists(): continue
        patch=json.loads(patch_path.read_text())
        if patch.get('complete') is not True: continue
        bdir=out/'branches'/task/f'seed_{seed:03d}'
        names=('clean_at_fork','top_removed','control_removed')
        if all((bdir/f'{n}.json').exists() for n in names): continue
        try:
            if current!=task:
                if env is not None: env.close()
                env,_=make_environment(task,shader_dir=args.shader_dir); cfg=get_policy_config('openvla',str(PCD_SOURCE/'pretrained/openvla-7b'),task,{},False)
                policy=build_historical_policy(OpenVLAInference(**cfg),task); current=task
            snap_path=VAN/'snapshots'/task/f'seed_{seed:03d}.pkl'
            with snap_path.open('rb') as f: initial=pickle.load(f)
            is_rt_switched=args.shader_dir=='rt' and task in ('google_robot_pick_coke_can','google_robot_move_near')
            vanilla_root=out/'baselines' if is_rt_switched else VAN
            hist_v=np.load(vanilla_root/'episodes'/task/'vanilla'/f'episode_{seed:03d}_arrays.npz')['executed_actions']
            fork_step=int(patch['fork_step']); fork_sim_path=fork_dir/'fork_sim.pkl'
            with fork_sim_path.open('rb') as f: fork_sim=pickle.load(f)
            groups=patch['groups']; top=int(patch['top_group_ids'][0]); top_size=int(groups[top]['size'])
            eligible=[g for g in groups if abs(int(g['size'])-top_size)<=2 and int(g['group_id'])!=top]
            control=min(eligible,key=lambda g:(float(g['R']),int(g['group_id']))) if eligible else min((g for g in groups if int(g['group_id'])!=top),key=lambda g:(float(g['R']),int(g['group_id'])))
            control_id=int(control['group_id'])
            clean_gripper=int(patch['clean_action_tokens'][-1])
            top_action=list(patch['groups'][top]['patched_action_tokens'])+[clean_gripper]
            control_action=list(patch['groups'][control_id]['patched_action_tokens'])+[clean_gripper]
            forks={'clean_at_fork':patch['clean_action_tokens'], 'top_removed':top_action, 'control_removed':control_action}
            for name,token_ids in forks.items():
                dest=bdir/f'{name}.json'
                if dest.exists(): continue
                obs,_,_=restore_snapshot(env,seed,initial); instruction=env.unwrapped.get_language_instruction(); policy.reset(instruction,seed=seed); policy._episode_trace=[]; policy._episode_logits=[]
                for step in range(fork_step):
                    image=get_image_from_maniskill2_obs_dict(env,obs); _,acts,_=policy.step(image,None,instruction,proprio=obs['agent']['eef_pos']); acts=acts[0] if isinstance(acts,list) else acts
                    obs,_,_,truncated,_=env.step(flat(acts))
                    if truncated: raise RuntimeError('truncated before fork while restoring policy history')
                    instruction=env.unwrapped.get_language_instruction()
                # Recreate the exact frozen fork simulator state, retaining policy-side
                # gripper history accumulated from the canonical common prefix.
                inner=env.unwrapped; inner.set_state(fork_sim['sim_state'].copy()); inner.agent.set_state(copy.deepcopy(fork_sim['agent_state'])); inner._episode_rng.set_state(copy.deepcopy(fork_sim['rng_state'])); obs=wrapped_observation(env); instruction=fork_sim['instruction']
                image=get_image_from_maniskill2_obs_dict(env,obs)
                if sha(image)!=patch['fork_rgb_sha256']: raise RuntimeError(f'fork RGB mismatch: {sha(image)} != {patch["fork_rgb_sha256"]}')
                ids=torch.as_tensor(token_ids,device='cuda:0',dtype=torch.long)
                raw=policy._decode_actions(ids,policy.unnorm_key)
                _raw,acts=policy.postprocess_actions(raw[None]); acts=acts[0] if isinstance(acts,list) else acts
                executed=flat(acts); obs,_,_,truncated,info=env.step(executed); infos=[convert_numpy_or_torch_to_python(info)]; predicted_terminated=False; steps=1
                while not truncated and not predicted_terminated:
                    instruction=env.unwrapped.get_language_instruction(); image=get_image_from_maniskill2_obs_dict(env,obs)
                    _r,acts,_meta=policy.step(image,None,instruction,proprio=obs['agent']['eef_pos']); acts=acts[0] if isinstance(acts,list) else acts
                    executed=flat(acts); obs,_,_,truncated,info=env.step(executed); infos.append(convert_numpy_or_torch_to_python(info)); steps+=1
                    predicted_terminated=bool(acts.get('terminate_episode',np.array([0]))[0]>0)
                    if predicted_terminated and not env.unwrapped.is_final_subtask(): predicted_terminated=False; env.advance_to_next_subtask()
                result=summarize(infos); result.update(stat_first(infos)); result.update(stat_final(infos))
                payload={'protocol_id':'L11_MATCHED_CAUSAL_GROUP_PATCH_V1','complete':True,'task':task,'seed':seed,'category':row['category'],'branch':name,'fork_step':fork_step,'fork_rgb_sha256':patch['fork_rgb_sha256'],'first_action_token_ids':[int(x) for x in token_ids],'control_group_id':control_id,'control_group_size':int(control['size']),'top_group_id':top,'top_group_size':top_size,'success':bool(result['success']),'control_steps_after_fork':steps,'result':result,'failure_reason':None if result['success'] else ('environment_time_limit' if truncated else 'policy_terminated_without_success')}
                atomic(dest,payload); print(json.dumps({'worker':args.worker_id,'task':task,'seed':seed,'branch':name,'success':payload['success'],'steps':steps}),flush=True)
        except Exception as e:
            failures+=1; atomic(out/'logs'/f'branch_failure_{args.worker_id}_{task}_{seed}_{int(time.time())}.json',{'task':task,'seed':seed,'error':repr(e),'traceback':traceback.format_exc(),'worker':args.worker_id}); print(json.dumps({'worker':args.worker_id,'task':task,'seed':seed,'status':'FAILED','error':repr(e)}),flush=True)
    if env is not None: env.close()
    raise SystemExit(1 if failures else 0)

if __name__=='__main__': main()
