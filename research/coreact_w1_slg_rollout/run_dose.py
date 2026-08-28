from __future__ import annotations

import argparse, copy, hashlib, json, math, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import prepare
from research.coreact_self_guidance.reference_snapshot_gate import fingerprints
from research.coreact_trained_weak.run_libero10_quality_screen import env_config
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_w1_slg_rollout.sampler import sample_w1_slg_actions

ARMS = ("lambda_0", "lambda_005", "lambda_010", "lambda_025", "lambda_050")
DOSES = {"lambda_n010": -0.1, "lambda_n005": -0.05, "lambda_0": 0.0, "lambda_001": 0.01, "lambda_005": 0.05, "lambda_010": 0.1, "lambda_025": 0.25, "lambda_050": 0.5}
MAX_STEPS, EXECUTE = 520, 10

def digest(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous(); raw = value.numpy().tobytes(); meta = f"{value.dtype}{tuple(value.shape)}".encode()
    else:
        value = np.ascontiguousarray(value); raw = value.tobytes(); meta = f"{value.dtype}{value.shape}".encode()
    return hashlib.sha256(meta + raw).hexdigest()

def atomic(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n'); tmp.replace(path)

def manifest(path):
    groups = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line); groups[row['pair_id']][row['arm']] = row
    if len(groups) != 500 and len(groups) != 250: raise RuntimeError('dose manifest pair count invalid')
    if any('lambda_0' not in x for x in groups.values()): raise RuntimeError('dose manifest baseline missing')
    return dict(groups)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--shard-count',type=int,default=1); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--pair-id'); p.add_argument('--split',choices=('all','selection','confirmation'),default='all'); p.add_argument('--arms',nargs='+',choices=('lambda_n010','lambda_n005','lambda_0','lambda_001','lambda_005','lambda_010','lambda_025','lambda_050'),default=list(ARMS)); p.add_argument('--max-new-pairs',type=int); a=p.parse_args()
    w,out=a.workspace.resolve(),a.artifact.resolve(); os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1'); os.environ.setdefault('MUJOCO_GL','egl')
    protocol=json.loads((out/'protocol.lock.json').read_text()); config,policy,pre,post=load_policy(Path(protocol['strong_checkpoint'])); groups=manifest(out/'episode_manifest.jsonl'); selected=[x for i,x in enumerate(sorted(groups.items())) if i%a.shard_count==a.shard_index]; created=0
    for pair_id,specs in selected:
        if a.split != 'all' and specs['lambda_0']['split'] != a.split: continue
        if a.pair_id and pair_id != a.pair_id: continue
        active_arms=tuple(a.arms)
        outputs={arm:out/'episodes'/f"{specs[arm]['episode_id']}.json" for arm in active_arms}
        if all(x.exists() for x in outputs.values()): continue
        if any(x.exists() for x in outputs.values()): raise RuntimeError(f'partial pair {pair_id}')
        initial={}; records={}; canonical=None
        for arm in active_arms:
            spec=specs[arm]; cfg=env_config(spec['task_id']); env_pre,env_post=make_env_pre_post_processors(env_cfg=cfg,policy_cfg=config); env=cfg.create_envs(n_envs=1,use_async_envs=False)['libero_10'][spec['task_id']]
            queue=[]; actions=[]; physical_actions=[]; noises=[]; traces=[]; latencies=[]; replans=0; success=False; reason='horizon'
            try:
                inner=env.envs[0]; inner.init_state_id=spec['init_state_id']; real_obs,_=env.reset(seed=spec['reset_seed']); language=inner.task_description; real_batch=prepare(policy,pre,env_pre,real_obs,language); reset_fp=fingerprints(env,real_obs,real_batch,[]); canonical=copy.deepcopy(real_obs) if canonical is None else canonical; obs=copy.deepcopy(canonical); first_fp=None
                while len(actions)<MAX_STEPS:
                    if not queue:
                        batch=prepare(policy,pre,env_pre,obs,language); first_fp=first_fp or fingerprints(env,obs,batch,[]); gen=torch.Generator(device=batch['state'].device).manual_seed(spec['flow_noise_seed_base']+replans); noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype); noises.append(digest(noise)); t0=time.perf_counter(); chunk,trace=sample_w1_slg_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,arm='strong' if arm=='lambda_0' else 'low_w1',lambda_value=DOSES[arm]); torch.cuda.synchronize(); latencies.append(time.perf_counter()-t0); ref=trace['reference_action_chunk']; guided=chunk[...,:7].detach().cpu(); traces.append({'replan':replans,'active_steps':trace['active_steps'],'clip_scales':[x['clip_scale'] for x in trace['step_traces'] if x['active']],'velocity_correction_norms':[x['applied_correction_norm'] for x in trace['step_traces'] if x['active']],'normalized_action_delta_l2':float(torch.linalg.vector_norm(guided-ref)),'normalized_action_delta_xyz':float(torch.linalg.vector_norm(guided[...,:3]-ref[...,:3])),'normalized_action_delta_rotation':float(torch.linalg.vector_norm(guided[...,3:6]-ref[...,3:6])),'normalized_action_delta_gripper':float(torch.linalg.vector_norm(guided[...,6:7]-ref[...,6:7])),'normalized_action_sha256':digest(guided),'reference_action_sha256':digest(ref)}); queue=[x.detach().cpu() for x in chunk[:,:EXECUTE,:7].transpose(0,1)]; replans+=1
                    action=queue.pop(0); actions.append(action[0].float().cpu()); physical=post(action); physical_actions.append(physical[0].detach().float().cpu()); legal=env_post({'action':physical})['action']; obs,_,terminated,_,info=env.step(legal.detach().cpu().numpy()); success=bool(vector_info_value(info,'is_success'))
                    if success: reason='success'; break
                    if bool(terminated[0]): reason='terminated'; break
            finally: env.close()
            finite=bool(actions and torch.isfinite(torch.stack(actions)).all()); initial[arm]={'reset':reset_fp,'policy_input':first_fp,'instruction':language,'first_noise_sha256':noises[0],'first_strong_velocity_sha256':traces[0]['reference_action_sha256']}
            records[arm]={**spec,'status':'complete','success':success,'termination_reason':reason,'control_steps':len(actions),'replans':replans,'all_actions_finite':finite,'maximum_control_steps':MAX_STEPS,'normalized_executed_actions_sha256':digest(torch.stack(actions)),'physical_executed_actions_sha256':digest(torch.stack(physical_actions)),'noise_sha256_by_replan':noises,'replan_traces':traces,'initial_integrity':initial[arm],'median_replan_latency_seconds':float(np.median(latencies)),'paired_initial_integrity':'PENDING'}
            if not finite: raise RuntimeError(f'nonfinite {spec["episode_id"]}')
        failures=[]
        for sec in ('reset','policy_input'):
            for field in ('simulator_state','qpos','qvel','object_pose','robot_observation','camera1','camera2','full_observation','preprocessing'):
                if len({initial[x][sec][field] for x in active_arms}) != 1: failures.append(f'{sec}.{field}')
        for field in ('instruction','first_noise_sha256','first_strong_velocity_sha256'):
            if len({initial[x][field] for x in active_arms}) != 1: failures.append(field)
        if failures:
            atomic(out/'invalid_pairs'/f'{pair_id}.json',{'pair_id':pair_id,'mismatches':failures,'initial':initial}); raise RuntimeError(f'integrity {pair_id}: {failures}')
        for arm in active_arms: records[arm]['paired_initial_integrity']='PASS'; atomic(outputs[arm],records[arm])
        created+=1; print(json.dumps({'pair_id':pair_id,'new_pairs':created,'success':{x:records[x]['success'] for x in active_arms}}),flush=True)
        if a.max_new_pairs and created>=a.max_new_pairs: break

if __name__=='__main__': main()
