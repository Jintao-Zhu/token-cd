#!/usr/bin/env python3
"""Canonical paired LIBERO rollout with online L11-Matched action-value traces."""
from __future__ import annotations
import argparse, hashlib, json, os, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
import torch

from research.ar_token_counterfactual.intervention import decode_action_ids
from research.ar_token_counterfactual.libero_runtime import (
    load_policy, predict_action, prepare_agentview, prepare_env_action, set_determinism,
)
from research.semantic_token_cd import libero_matched_rollout as lmr
from research.semantic_token_cd.libero90_phase_worker import TASK_OBJECTS, body_pos, target_grasped
from research.semantic_token_cd.libero_policy import token_ids_from_mask
from research.semantic_token_cd.resolve_mujoco_egl_device import resolve as resolve_egl_device
from research.semantic_token_cd.libero90_formal_worker import atomic_json, code_commit, sha256_file

torch.set_num_threads(1)
PROTOCOL='LIBERO90_RAW_L11_VS_SPATIAL_NULL_V1'
ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
CODE_DIR=ROOT/'third_party/openvla/prismatic/extern/hf'
CHECKPOINT=Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
ACTION_BINS=256

def sha_bytes(x: np.ndarray)->str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def body_pose(inner,name):
    if name not in getattr(inner,'obj_body_id',{}): return None
    bid=int(inner.obj_body_id[name])
    return {'position':np.asarray(inner.sim.data.body_xpos[bid],dtype=float).tolist(),
            'quaternion_wxyz':np.asarray(inner.sim.data.body_xquat[bid],dtype=float).tolist()}

def tcp_pose(inner):
    try:
        robot=inner.robots[0]
        site=getattr(robot,'eef_site_id')
        if isinstance(site,dict): site=next(iter(site.values()))
        if isinstance(site,(list,tuple,np.ndarray)): site=site[0]
        sid=int(site)
        pos=np.asarray(inner.sim.data.site_xpos[sid],dtype=float)
        mat=np.asarray(inner.sim.data.site_xmat[sid],dtype=float).reshape(3,3)
        from robosuite.utils.transform_utils import mat2quat
        quat=np.asarray(mat2quat(mat),dtype=float)
        return {'position':pos.tolist(),'quaternion_wxyz':quat.tolist()}
    except Exception as exc:
        return {'unavailable':f'{type(exc).__name__}: {exc}'}

def robot_joint_state(inner):
    try:
        robot=inner.robots[0]
        idx=np.asarray(robot._ref_joint_pos_indexes,dtype=int)
        return {'joint_positions':np.asarray(inner.sim.data.qpos[idx],dtype=float).tolist(),
                'joint_velocities':np.asarray(inner.sim.data.qvel[idx],dtype=float).tolist()}
    except Exception as exc:
        return {'unavailable':f'{type(exc).__name__}: {exc}'}

def gripper_state(inner,target):
    out={'target_grasped':bool(target_grasped(inner,target))}
    try:
        robot=inner.robots[0]
        idx=np.asarray(robot._ref_gripper_joint_pos_indexes,dtype=int)
        out['joint_positions']=np.asarray(inner.sim.data.qpos[idx],dtype=float).tolist()
    except Exception as exc:
        out['joint_positions_unavailable']=f'{type(exc).__name__}: {exc}'
    try:
        out['current_action']=np.asarray(inner.robots[0].gripper.current_action,dtype=float).reshape(-1).tolist()
    except Exception:
        pass
    return out

_active_target=[None]

def phase_label(step, grasped, stable, dist, near_threshold=0.12):
    if stable:
        if not grasped: return 'placement_or_release'
        return 'placement' if dist<=near_threshold else 'transport'
    if grasped: return 'grasp'
    return 'alignment' if dist<=near_threshold else 'approach'

def install_capture(model):
    captured={}
    orig_clean=lmr.generate_clean_action
    orig_forward=lmr.forward_logits
    orig_attention=lmr.prompt_attention_and_features
    vocab_start=int(model.vocab_size)-ACTION_BINS
    def wrap_clean(*args,**kwargs):
        out=orig_clean(*args,**kwargs)
        captured['positive_logits']=out[1].detach().float().cpu().numpy()[:,vocab_start:vocab_start+ACTION_BINS].copy()
        captured['clean_token_ids']=out[0].detach().cpu().numpy()[0].astype(np.int64).copy()
        return out
    def wrap_forward(*args,**kwargs):
        out=orig_forward(*args,**kwargs)
        captured['negative_logits']=out.detach().float().cpu().numpy()[:,vocab_start:vocab_start+ACTION_BINS].copy()
        return out
    def wrap_attention(*args,**kwargs):
        out=orig_attention(*args,**kwargs)
        captured['attention_scores']=np.asarray(out[0],dtype=np.float32).copy()
        return out
    lmr.generate_clean_action=wrap_clean
    lmr.forward_logits=wrap_forward
    lmr.prompt_attention_and_features=wrap_attention
    return captured, (orig_clean,orig_forward,orig_attention)

def run_arm(arm,env,task,init_state,case,model,processor,args,verify_clean_equivalence=False):
    set_determinism(case['case_seed'])
    env.seed(case['env_seed']); env.reset(); obs=env.set_init_state(init_state)
    for _ in range(case['settle_steps']): obs,_,_,_=env.step([0,0,0,0,0,0,-1])
    initial_state=np.asarray(env.get_sim_state()).copy()
    initial_sha=sha_bytes(initial_state)
    inner=getattr(env,'env',env)
    source_name,target_name=TASK_OBJECTS[task.name]
    _active_target[0]=source_name
    source_initial=body_pose(inner,source_name)
    source_z0=(source_initial or {}).get('position',[0,0,0])[2]
    trace=[]; arrays=[]; done=False; stable=False; grasp_streak=0; first_stable=None; started=time.monotonic()
    captured,originals=install_capture(model)
    from libero.libero.envs import OffScreenRenderEnv
    try:
      for step in range(case['max_policy_steps']):
        sim_state=np.asarray(env.get_sim_state()).copy()
        rgb,image=prepare_agentview(obs)
        rgb_arr=np.asarray(image.convert('RGB'),dtype=np.uint8)
        rgb_hash=sha_bytes(rgb_arr)
        sim_hash=sha_bytes(sim_state)
        source_pose=body_pose(inner,source_name); target_pose=body_pose(inner,target_name)
        grasped=bool(target_grasped(inner,source_name))
        source_pos=np.asarray((source_pose or {}).get('position',[np.nan]*3),dtype=float)
        target_pos=np.asarray((target_pose or {}).get('position',[np.nan]*3),dtype=float)
        distance=float(np.linalg.norm(source_pos-target_pos)) if np.isfinite(source_pos).all() and np.isfinite(target_pos).all() else None
        lifted=(float(source_pos[2])-float(source_z0)) if np.isfinite(source_pos).all() else 0.0
        if grasped and lifted>0.01: grasp_streak+=1
        else: grasp_streak=0
        if not stable and grasp_streak>=3: stable=True; first_stable=step
        phase=phase_label(step,grasped,stable,distance if distance is not None else 1.0)
        captured.clear()
        selector_transform='rot180' if arm=='matched_rot180' else 'identity'
        fixed_m=int(case['fixed_m']) if arm=='matched_fixed_m' else None
        null_seed=None
        null_prior=None
        if arm=='matched_spatial_null':
            null_prior=np.asarray(args.position_prior[str(int(case['task_id']))],dtype=np.float64).reshape(-1)
            payload=f"{int(case['task_id'])}:{int(case['case_seed'])}:{step}:N1".encode()
            null_seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')
        guided_action,meta=lmr.predict_matched(model,processor,image,task.language,
            entity_mode='source_target_libero90',query_mode='instruction_only',attention_layers=(11,),
            attention_heads=(),destination_weight=0.0,lambda_scale=1.0,
            unnorm_key=args.unnorm_key,position_mode='attention',
            selector_transform=selector_transform,fixed_m=fixed_m,
            spatial_null_prior=null_prior,spatial_null_seed=null_seed)
        if not all(k in captured for k in ('positive_logits','negative_logits','attention_scores')):
            raise RuntimeError(f'missing online branch capture at {task.name}:{arm}:{step}: {list(captured)}')
        zplus=captured['positive_logits']; zminus=captured['negative_logits']; attention=captured['attention_scores']
        clean_token_ids=captured['clean_token_ids']
        clean_action=decode_action_ids(model,torch.as_tensor(clean_token_ids[None],dtype=torch.long),args.unnorm_key)
        clean_action=np.asarray(clean_action,dtype=float).reshape(7)
        guided_action=np.asarray(guided_action,dtype=float).reshape(7)
        if arm=='vanilla':
            executed_raw=np.asarray(predict_action(model,processor,image,task.language,unnorm_key=args.unnorm_key),dtype=float)
            official_diff=np.asarray(executed_raw-clean_action,dtype=float)
            if verify_clean_equivalence and step==0 and not np.allclose(official_diff,0.0,atol=1e-5,rtol=0.0):
                raise RuntimeError(f'official Vanilla action differs from captured clean-logit decode by {official_diff.tolist()}')
        else:
            executed_raw=guided_action.copy(); official_diff=None
        env_action=prepare_env_action(executed_raw)
        if env_action.shape!=(7,) or not np.isfinite(env_action).all(): raise FloatingPointError(f'invalid action {env_action}')
        final_logits=zplus.copy()
        final_logits[:6]=(1.5*zplus[:6]-.5*zminus[:6])
        top_order=np.lexsort((np.arange(ACTION_BINS),-attention))
        row={'step':step,'task_phase':phase,'stable_grasp_event':bool(stable),'gripper_object_contact':grasped,
             'sim_state_sha256':sim_hash,'rgb_sha256':rgb_hash,'instruction':task.language,
             'attention_top256_order':top_order.astype(int).tolist(),'attention_sha256':sha_bytes(attention),
             'matched_m':int(meta['m_matched']),'selected_token_ids':meta['selected_token_ids'],
             'selector_transform':selector_transform,
             'selector_source':meta.get('selector_source'),'spatial_null_id':'N1' if null_seed is not None else None,
             'spatial_null_seed':null_seed,
             'm_used':int(meta['m_used']),'fixed_m':meta['fixed_m'],
             'positive_token_ids':clean_token_ids.tolist(),'negative_token_ids':meta['negative_token_ids'],
             'guided_token_ids':meta['final_token_ids'],'clean_action_from_positive_logits':clean_action.tolist(),
             'guided_action':guided_action.tolist(),'executed_raw_action':executed_raw.tolist(),
             'executed_env_action':env_action.tolist(),'official_clean_minus_branch_action':official_diff.tolist() if official_diff is not None else None,
             'clean_vs_guided_changed_dims':int(np.count_nonzero(clean_token_ids[:6]!=np.asarray(meta['final_token_ids'])[:6])),
             'tcp_pose':tcp_pose(inner),'robot_joint_state':robot_joint_state(inner),
             'gripper_state':gripper_state(inner,source_name),
             'source_object_pose':source_pose,'target_object_pose':target_pose,
             'source_target_distance_m':distance,'source_lift_m':lifted}
        trace.append(row)
        arrays.append((zplus,zminus,final_logits,attention.astype(np.float32)))
        obs,_,done,_=env.step(env_action.tolist())
        if done: break
      success=bool(env.check_success())
    finally:
      lmr.generate_clean_action,lmr.forward_logits,lmr.prompt_attention_and_features=originals
    data={'positive_action_logits':np.stack([x[0] for x in arrays]).astype(np.float32),
          'negative_action_logits':np.stack([x[1] for x in arrays]).astype(np.float32),
          'guided_action_logits':np.stack([x[2] for x in arrays]).astype(np.float32),
          'l11_attention_scores':np.stack([x[3] for x in arrays]).astype(np.float32)}
    episode_dir=args.artifact/'episodes'/case['case_id']/arm
    episode_dir.mkdir(parents=True,exist_ok=True)
    npz=episode_dir/'step_arrays.npz'; tmp=npz.with_suffix('.npz.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**data)
    os.replace(tmp,npz)
    result={'protocol_id':PROTOCOL,'case_id':case['case_id'],'task_id':case['task_id'],'task_name':task.name,
      'instruction':task.language,'arm':arm,'init_state_id':case['init_state_id'],'case_seed':case['case_seed'],
      'env_seed':case['env_seed'],'settle_steps':case['settle_steps'],'max_policy_steps':case['max_policy_steps'],
      'initial_state_sha256':initial_sha,'success':success,'done':bool(done),'steps':len(trace),
      'first_stable_grasp_step':first_stable,'runtime_seconds':time.monotonic()-started,
      'checkpoint':str(CHECKPOINT),'checkpoint_revision':args.checkpoint_revision,'code_commit':code_commit(),
      'renderer_backend':args.renderer,
      'canonical_config':{'entity_mode':'source_target_libero90','query_mode':'instruction_only','attention_layers':[11],
                          'attention_heads':'all-mean','kmeans_k':8,'lambda':0.5,'negative':'canonical_harmonic_beta0',
                          'selector_transform':'rot180' if arm=='matched_rot180' else 'identity',
                          'budget':'fixed_task_m' if arm=='matched_fixed_m' else 'canonical_dynamic_matched',
                          'fixed_m':int(case['fixed_m']) if arm=='matched_fixed_m' else None},
      'arrays':str(npz.relative_to(args.artifact)),'trace':trace}
    atomic_json(episode_dir/'episode.json',result)
    return result

def main():
    global PROTOCOL
    p=argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True); p.add_argument('--gpu',type=int,choices=(1,2,3,4,5,6,7),required=True)
    p.add_argument('--render-gpu',type=int,choices=(1,2,3,4,5,6,7),default=None)
    p.add_argument('--renderer',choices=('egl','osmesa'),default='egl')
    p.add_argument('--position-prior',type=Path,default=None)
    p.add_argument('--cases-file',type=Path,default=None)
    p.add_argument('--protocol-id',default=PROTOCOL)
    p.add_argument('--worker-id',required=True); p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--selector-placebo',action='store_true',help='run only the L11 180-degree spatial placebo arm')
    p.add_argument('--fixed-budget',action='store_true',help='run only the task-specific fixed-m arm')
    p.add_argument('--checkpoint-revision',default='local-libero-vq-openvla-7b')
    p.add_argument('--unnorm-key',default='libero_90_no_noops'); p.add_argument('--verify-clean-equivalence',action='store_true')
    a=p.parse_args(); a.artifact=a.artifact.resolve(); a.protocol_id=str(a.protocol_id)
    PROTOCOL=a.protocol_id
    if a.position_prior is not None:
      prior_doc=json.loads(a.position_prior.read_text())
      if 'position_priors' in prior_doc:
        a.position_prior=prior_doc['position_priors']
      else:
        a.position_prior=prior_doc['results']['task_specific']['prior_16x16']
    else:
      a.position_prior={}
    render_gpu=int(a.render_gpu if a.render_gpu is not None else a.gpu)
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
    egl_selection=None
    if a.renderer=='osmesa':
        os.environ['MUJOCO_GL']='osmesa'
        os.environ['PYOPENGL_PLATFORM']='osmesa'
        os.environ.pop('MUJOCO_EGL_DEVICE_ID',None)
    else:
        os.environ['MUJOCO_GL']='egl'
        os.environ['PYOPENGL_PLATFORM']='egl'
        egl_selection=resolve_egl_device(render_gpu)
        resolved_egl_id=str(egl_selection['egl_ordinal'])
        # This robosuite version has an import-time assertion that incorrectly
        # expects MUJOCO_EGL_DEVICE_ID to be a CUDA_VISIBLE_DEVICES token.
        # Satisfy that guard during import, then restore MuJoCo's actual EGL
        # ordinal before any OffScreenRenderEnv constructs a GL context.
        os.environ['MUJOCO_EGL_DEVICE_ID']=str(a.gpu)
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    if a.renderer=='egl':
        os.environ['MUJOCO_EGL_DEVICE_ID']=resolved_egl_id
        print(json.dumps({'renderer':'egl','egl_physical_device_resolution':egl_selection},sort_keys=True),flush=True)
    else:
        print(json.dumps({'renderer':'osmesa','egl_gpu_context':'not used'},sort_keys=True),flush=True)
    visible=[int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    if a.gpu not in visible: raise RuntimeError(f'GPU {a.gpu} not visible: {visible}')
    device_index=visible.index(a.gpu)
    suite=benchmark.get_benchmark_dict()['libero_90']()
    model,processor=load_policy(CHECKPOINT,CODE_DIR,device=f'cuda:{device_index}',dataset_statistics_path=CHECKPOINT/'dataset_statistics.json',unnorm_key=a.unnorm_key)
    queue_path=a.cases_file if a.cases_file is not None else (a.artifact/'cases'/f'gpu{a.gpu}.jsonl')
    queue=queue_path.read_text().splitlines()
    log=a.artifact/'logs'/f'worker_{a.worker_id}.jsonl'; log.parent.mkdir(parents=True,exist_ok=True)
    for n,line in enumerate(queue):
      if a.max_cases and n>=a.max_cases: break
      case=json.loads(line); tid=int(case['task_id']); task=suite.get_task(tid)
      pair_path=a.artifact/'pairs'/f'{case["case_id"]}.json'
      if pair_path.is_file():
        print(json.dumps({'skip_completed_case':case['case_id']}),flush=True)
        continue
      if task.name!=case['task_name']: raise RuntimeError(f'task mismatch {tid}: {task.name}')
      init=suite.get_task_init_states(tid)[int(case['init_state_id'])]
      bddl=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
      pair_results={}
      if a.selector_placebo and a.fixed_budget:
        raise ValueError('selector placebo and fixed budget arms are mutually exclusive')
      arms=('matched_rot180',) if a.selector_placebo else (('matched_fixed_m',) if a.fixed_budget else ('vanilla','matched','matched_spatial_null'))
      if 'matched_spatial_null' in arms and len(a.position_prior)!=5:
        raise RuntimeError('spatial-null arm requires a frozen five-task position prior')
      for arm in arms:
        env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256)
        try:
          pair_results[arm]=run_arm(arm,env,task,init,case,model,processor,a,verify_clean_equivalence=(a.verify_clean_equivalence and n==0))
        finally: env.close()
      if a.selector_placebo:
        pair={'protocol_id':PROTOCOL+'_SELECTOR_ROT180','case_id':case['case_id'],'task_id':tid,'task_name':task.name,
          'init_state_id':case['init_state_id'],'case_seed':case['case_seed'],
          'initial_state_sha256':pair_results['matched_rot180']['initial_state_sha256'],
          'rot180_success':pair_results['matched_rot180']['success'],'rot180_steps':pair_results['matched_rot180']['steps'],
          'worker_id':a.worker_id,'gpu':a.gpu,'renderer_backend':a.renderer}
      elif a.fixed_budget:
        pair={'protocol_id':PROTOCOL+'_FIXED_TASK_M','case_id':case['case_id'],'task_id':tid,'task_name':task.name,
          'init_state_id':case['init_state_id'],'case_seed':case['case_seed'],
          'initial_state_sha256':pair_results['matched_fixed_m']['initial_state_sha256'],
          'fixed_m':int(case['fixed_m']),'fixed_m_success':pair_results['matched_fixed_m']['success'],
          'fixed_m_steps':pair_results['matched_fixed_m']['steps'],
          'worker_id':a.worker_id,'gpu':a.gpu,'renderer_backend':a.renderer}
      else:
        initial_hashes={pair_results[arm]['initial_state_sha256'] for arm in arms}
        same=len(initial_hashes)==1
        if not same: raise RuntimeError(f'paired initial state mismatch {case["case_id"]}')
        pair={'protocol_id':PROTOCOL,'case_id':case['case_id'],'task_id':tid,'task_name':task.name,
          'init_state_id':case['init_state_id'],'case_seed':case['case_seed'],'initial_state_sha256':pair_results['vanilla']['initial_state_sha256'],
          'vanilla_success':pair_results['vanilla']['success'],'matched_success':pair_results['matched']['success'],
          'spatial_null_success':pair_results['matched_spatial_null']['success'],
          'outcome_vs_vanilla':{
            'matched':'rescue' if pair_results['matched']['success'] and not pair_results['vanilla']['success'] else ('harm' if pair_results['vanilla']['success'] and not pair_results['matched']['success'] else 'concordant'),
            'spatial_null':'rescue' if pair_results['matched_spatial_null']['success'] and not pair_results['vanilla']['success'] else ('harm' if pair_results['vanilla']['success'] and not pair_results['matched_spatial_null']['success'] else 'concordant')},
          'vanilla_steps':pair_results['vanilla']['steps'],'matched_steps':pair_results['matched']['steps'],
          'spatial_null_steps':pair_results['matched_spatial_null']['steps'],'paired_initial_state_match':same,
          'worker_id':a.worker_id,'gpu':a.gpu,'renderer_backend':a.renderer}
      atomic_json(pair_path,pair)
      with log.open('a') as f: f.write(json.dumps({'status':'complete',**pair},sort_keys=True)+'\n')
      print(json.dumps(pair),flush=True)
if __name__=='__main__': main()
