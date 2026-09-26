#!/usr/bin/env python3
"""Paired phase-gating diagnostic worker.

Runs pre_grasp_only and post_grasp_only on the same case.  The stable-grasp
trigger is simulator-state based and is frozen before the run.  This is a
mechanism diagnostic; the eventual deployable method must estimate phase from
policy-available signals.
"""
from __future__ import annotations
import argparse, json, os, time, traceback
from pathlib import Path
import numpy as np
import torch

from research.ar_token_counterfactual.intervention import decode_action_ids
from research.ar_token_counterfactual.libero_runtime import (
    encode_video, load_policy, predict_action, prepare_agentview, prepare_env_action, set_determinism,
)
from research.semantic_token_cd.libero_matched_rollout import predict_matched
from research.semantic_token_cd.libero90_formal_worker import (
    atomic_json, claim_case, complete_case, fail_case, code_commit, sha256_file,
)

PROTOCOL='LIBERO90_PHASE_GATING_DIAGNOSTIC_V1'
TASK_OBJECTS={
 'KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it':('butter_2','wooden_cabinet_1'),
 'KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet':('akita_black_bowl_1','wooden_cabinet_1'),
 'LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket':('tomato_sauce_1','basket_1'),
 'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate':('porcelain_mug_1','plate_1'),
 'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy':('black_book_1','desk_caddy_1'),
 # Unseen-task positive-G gate validation set; object instance names verified
 # against each task's BDDL before adding these diagnostics.
 'KITCHEN_SCENE3_put_the_frying_pan_on_the_stove':('chefmate_8_frypan_1','flat_stove_1'),
 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet':('akita_black_bowl_1','white_cabinet_1'),
 'KITCHEN_SCENE7_put_the_white_bowl_on_the_plate':('white_bowl_1','plate_1'),
 'LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket':('alphabet_soup_1','basket_1'),
 'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy':('black_book_1','desk_caddy_1'),
}
STABLE_GRASP_CONSECUTIVE=3
STABLE_LIFT_THRESHOLD_M=0.01


def body_pos(inner,name):
    return np.asarray(inner.sim.data.body_xpos[inner.obj_body_id[name]],dtype=float)


def target_grasped(inner,target):
    try:
        return bool(inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=inner.objects_dict[target]))
    except Exception:
        return False


def action_delta(clean, guided):
    clean=np.asarray(clean,dtype=float); guided=np.asarray(guided,dtype=float)
    return {
        'translation_delta': np.asarray(guided[:3]-clean[:3],dtype=float).tolist(),
        'rotation_delta': np.asarray(guided[3:6]-clean[3:6],dtype=float).tolist(),
        'gripper_delta': float(guided[6]-clean[6]),
    }


def clean_from_matched_meta(model,meta,unnorm_key):
    ids=torch.tensor([meta['positive_token_ids']],dtype=torch.long)
    return decode_action_ids(model,ids,unnorm_key)


def run_arm(arm, env, task, init_state, model, processor, artifact, case, args, case_seed, config_hash, checkpoint_revision, commit):
    set_determinism(case_seed)
    env.seed(int(args.env_seed)); env.reset(); obs=env.set_init_state(init_state)
    for _ in range(int(args.settle_steps)): obs,_,_,_=env.step([0,0,0,0,0,0,-1])
    initial_state_sha=__import__('hashlib').sha256(np.asarray(env.get_sim_state()).tobytes()).hexdigest()
    inner=getattr(env,'env',env); target,_=TASK_OBJECTS[task.name]; t0=body_pos(inner,target); initial_z=float(t0[2])
    phase_locked=False; grasp_streak=0; trajectory=[]; trace=[]; frames=[]; done=False; started=time.monotonic()
    for step in range(int(args.max_steps)):
        grasped=target_grasped(inner,target); target_z=float(body_pos(inner,target)[2])
        if (not phase_locked) and grasped and (target_z-initial_z)>STABLE_LIFT_THRESHOLD_M:
            grasp_streak += 1
            if grasp_streak >= STABLE_GRASP_CONSECUTIVE:
                phase_locked=True
        else:
            grasp_streak=0
        phase='post_grasp' if phase_locked else 'pre_grasp'
        guidance_enabled = (arm=='pre_grasp_only' and not phase_locked) or (arm=='post_grasp_only' and phase_locked)
        if args.force_guidance:
            guidance_enabled=True
        _,image=prepare_agentview(obs)
        if args.save_video: frames.append(np.asarray(image.copy()))
        if guidance_enabled:
            guided,meta=predict_matched(model,processor,image,task.language,entity_mode=args.entity_mode,query_mode=args.query_mode,
                attention_layers=(11,),attention_heads=(),destination_weight=0.0,lambda_scale=1.0,unnorm_key=args.unnorm_key,position_mode='attention')
            clean=clean_from_matched_meta(model,meta,args.unnorm_key)
            info={
                'guidance_enabled':True,'m_matched':meta['m_matched'],'matched_cluster_ids':meta['kmeans_groups'],
                'selected_token_ids':meta['selected_token_ids'],'guided_changed_dims':meta['guided_changed_dims'],
                'feature_perturbation_relative':meta['feature_perturbation_relative'],
                'clean_action_source':'matched_positive_full_decode',
            }
        else:
            clean=predict_action(model,processor,image,task.language,unnorm_key=args.unnorm_key)
            guided=clean; info={'guidance_enabled':False,'m_matched':None,'matched_cluster_ids':[],'selected_token_ids':[],
                                'guided_changed_dims':0,'feature_perturbation_relative':0.0,'clean_action_source':'official_predict_action'}
        env_action=prepare_env_action(guided)
        if env_action.shape!=(7,) or not np.isfinite(env_action).all(): raise FloatingPointError(f'invalid action {env_action}')
        trace.append({'step':step,'phase':phase,'stable_grasp_trigger':bool(phase_locked),'guidance_enabled':bool(guidance_enabled),
            'target_contact':bool(grasped),'target_pose':body_pos(inner,target).tolist(),
            'clean_action':np.asarray(clean,dtype=float).tolist(),'guided_action':np.asarray(guided,dtype=float).tolist(),
            'executed_action':np.asarray(env_action,dtype=float).tolist(), **action_delta(clean,guided), **info})
        obs,_reward,done,_info=env.step(env_action.tolist()); trajectory.append(np.asarray(env_action,dtype=float).tolist())
        if done: break
    success=bool(env.check_success()); normal_end=bool(success or done or len(trajectory)>=int(args.max_steps))
    result={'protocol_id':PROTOCOL,'case_id':case['case_id'],'task_id':int(case['task_id']),'task_name':task.name,'instruction':task.language,
        'arm':arm,'init_state_id':int(case['init_state_id']),'init_state_sha256':initial_state_sha,'case_seed':case_seed,
        'env_seed':int(args.env_seed),'settle_steps':int(args.settle_steps),'max_policy_steps':int(args.max_steps),'success':success,
        'normal_end':normal_end,'done':bool(done),'failure_reason':'success' if success else ('environment_done' if done else 'environment_time_limit'),
        'steps':len(trajectory),'runtime_seconds':time.monotonic()-started,'checkpoint_revision':checkpoint_revision,'code_commit':commit,
        'config_sha256':config_hash,'worker_id':args.worker_id,'physical_gpu':int(args.physical_gpu),'render_gpu':int(args.render_gpu or args.physical_gpu),
        'stable_grasp_triggered':bool(phase_locked),'first_stable_grasp_step':next((x['step'] for x in trace if x['stable_grasp_trigger']),None),
        'trajectory':trajectory,'trace':trace}
    atomic_json(artifact/'episodes'/task.name/arm/f"init_{int(case['init_state_id']):03d}.json",result)
    if args.save_video and frames: encode_video(frames,artifact/'videos'/task.name/arm/f"init_{int(case['init_state_id']):03d}.mp4",fps=30)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--formal-root',type=Path,required=True); ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--checkpoint-revision',required=True); ap.add_argument('--unnorm-key',default='libero_90_no_noops')
    ap.add_argument('--entity-mode',default='source_target_libero90'); ap.add_argument('--query-mode',default='instruction_only'); ap.add_argument('--attention-layers',default='11')
    ap.add_argument('--env-seed',type=int,default=0); ap.add_argument('--settle-steps',type=int,default=10); ap.add_argument('--max-steps',type=int,default=400)
    ap.add_argument('--physical-gpu',type=int,choices=(1,2,3),required=True); ap.add_argument('--render-gpu',type=int,choices=(1,2,3),default=None); ap.add_argument('--worker-id',required=True)
    ap.add_argument('--max-cases',type=int,default=0); ap.add_argument('--save-video',action='store_true'); ap.add_argument('--force-guidance',action='store_true')
    a=ap.parse_args(); a.artifact=a.artifact.resolve(); a.formal_root=a.formal_root.resolve(); a.config=a.config.resolve(); render=int(a.render_gpu or a.physical_gpu)
    visible=[int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    if a.physical_gpu not in visible or render not in visible: raise RuntimeError(f'GPU mapping invalid visible={visible} model={a.physical_gpu} render={render}')
    if os.environ.get('MUJOCO_EGL_DEVICE_ID')!=str(render): raise RuntimeError('MUJOCO_EGL_DEVICE_ID mismatch')
    model_index=visible.index(a.physical_gpu)
    os.environ.setdefault('MUJOCO_GL','egl'); os.environ.setdefault('PYOPENGL_PLATFORM','egl'); os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    config_hash=sha256_file(a.config); commit=code_commit(); suite=benchmark.get_benchmark_dict()['libero_90']()
    model,processor=load_policy(a.checkpoint,Path('/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf'),device=f'cuda:{model_index}',dataset_statistics_path=a.checkpoint/'dataset_statistics.json',unnorm_key=a.unnorm_key)
    log=a.artifact/'logs'/f'worker_{a.worker_id}.jsonl'; log.parent.mkdir(parents=True,exist_ok=True); processed=0
    while a.max_cases<=0 or processed<a.max_cases:
        claimed=claim_case(a.artifact,a.worker_id)
        if claimed is None: break
        case,running=claimed; started=time.monotonic()
        try:
            tid=int(case['task_id']); task=suite.get_task(tid); init=suite.get_task_init_states(tid)[int(case['init_state_id'])]; bddl=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
            case_seed=20260922+tid*10000+int(case['init_state_id']); results={}
            for arm in ('pre_grasp_only','post_grasp_only'):
                env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256)
                try: results[arm]=run_arm(arm,env,task,init,model,processor,a.artifact,case,a,case_seed,config_hash,a.checkpoint_revision,commit)
                finally: env.close()
            if results['pre_grasp_only']['init_state_sha256']!=results['post_grasp_only']['init_state_sha256']: raise RuntimeError('paired init-state hash mismatch')
            pair={'protocol_id':PROTOCOL,'case_id':case['case_id'],'task_id':tid,'task_name':task.name,'instruction':task.language,'init_state_id':int(case['init_state_id']),
                  'case_seed':case_seed,'init_state_sha256':results['pre_grasp_only']['init_state_sha256'],
                  'pre_grasp_only_success':results['pre_grasp_only']['success'],'post_grasp_only_success':results['post_grasp_only']['success'],
                  'pre_grasp_only_steps':results['pre_grasp_only']['steps'],'post_grasp_only_steps':results['post_grasp_only']['steps'],
                  'pre_first_stable_grasp_step':results['pre_grasp_only']['first_stable_grasp_step'],
                  'post_first_stable_grasp_step':results['post_grasp_only']['first_stable_grasp_step'],
                  'worker_id':a.worker_id,'physical_gpu':a.physical_gpu}
            atomic_json(a.artifact/'pairs'/f"{case['case_id']}.json",pair); complete_case(running,a.artifact)
            with log.open('a') as h: h.write(json.dumps({'case_id':case['case_id'],'status':'complete','seconds':time.monotonic()-started,**pair},sort_keys=True)+'\n')
        except Exception as exc:
            fail_case(running,a.artifact,f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
            with log.open('a') as h: h.write(json.dumps({'case_id':case.get('case_id'),'status':'error','seconds':time.monotonic()-started,'error':f'{type(exc).__name__}: {exc}'},sort_keys=True)+'\n')
        processed+=1

if __name__=='__main__': main()
