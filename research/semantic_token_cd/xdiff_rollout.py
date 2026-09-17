"""Closed-loop workers for semantic-difference selector arms."""
from __future__ import annotations

import argparse, copy, hashlib, json, os, pickle, time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.xdiff_protocol import (
    ARTIFACT, ATTENTION_LAYERS, CANONICAL, EPSILON, LAMBDA, NEW_ARMS, PCD_SOURCE,
    TASKS, atomic_json, instruction_set, resolve_present_phrases, selector_config,
    swap_for_scene,
)
from research.semantic_token_cd.xswap_rollout import make_environment


def parse_seeds(spec):
    out=[]
    for part in spec.split(','):
        if '-' in part:
            a,b=map(int,part.split('-',1)); out.extend(range(a,b+1))
        elif part.strip(): out.append(int(part))
    return sorted(set(out))


def build_policy(base, task, arm, plan):
    p=copy.copy(base); p.__class__=PromptAttentionSHRInference; _init_common(p,LAMBDA)
    p.beta=0.0; p.selector_mode='prompt_attention'; p.task_index=TASK_INDEX[task]
    p.attention_layers=tuple(ATTENTION_LAYERS); p.selection_count=None; p.selection_top_p=None
    p.save_prompt_attention=True; p.selector_instruction=None
    cfg=selector_config(arm,plan)
    p.selector_contrast_instruction=cfg['contrast_instruction']; p.selector_difference_eta=cfg['eta']
    p.selector_difference_kind=cfg['contrast_kind']; p.selector_difference_epsilon=cfg['epsilon']
    return p


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda:fh.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def run_loop(env, policy, instruction, obs, video_path):
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import flatten_action
    from utils import convert_numpy_or_torch_to_python, summarize
    image=get_image_from_maniskill2_obs_dict(env,obs); infos=[]; actions=[]; frames=[]
    predicted=False; truncated=False; control=0
    while not (predicted or truncated) and control<140:
        frames.append(np.asarray(image,dtype=np.uint8))
        _raw,acts,_meta=policy.step(image,None,instruction,proprio=obs['agent']['eef_pos'])
        if not isinstance(acts,list): acts=[acts]
        for act in acts:
            a=flatten_action(act)
            if a.shape!=(7,) or not np.isfinite(a).all(): raise FloatingPointError('invalid action')
            actions.append(a.copy()); obs,_r,_s,truncated,info=env.step(a)
            image=get_image_from_maniskill2_obs_dict(env,obs); control+=1
            infos.append(convert_numpy_or_torch_to_python(info)); predicted=bool(act['terminate_episode'][0]>0)
            if predicted and not env.unwrapped.is_final_subtask():
                predicted=False; env.advance_to_next_subtask()
    frames.append(np.asarray(image,dtype=np.uint8))
    video_path.parent.mkdir(parents=True,exist_ok=True)
    imageio.mimwrite(video_path,frames,fps=10,codec='libx264',quality=7,macro_block_size=None)
    result=summarize(infos); result['failure_reason']=None if result.get('success') else ('time_limit' if truncated else 'policy_terminated')
    return result, np.asarray(actions,dtype=np.float32), infos


def write_arrays(path, records, actions):
    payload={'executed_actions':actions}
    keys=('positive','negative','selected_mask','reference_shr_mask','prompt_attention',
          'correct_attention_probability','contrast_attention_probability',
          'attention_log_difference','selector_score','per_token_perturbation_norm')
    for key in keys:
        if records and all(key in x for x in records): payload[key]=np.stack([x[key] for x in records])
    np.savez_compressed(path,**payload)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True,choices=TASKS); ap.add_argument('--seeds',required=True)
    ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--arm',required=True,choices=NEW_ARMS); ap.add_argument('--worker-id',default='manual')
    args=ap.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    env,env_id=make_environment(args.task,args.gpu); checkpoint=str(PCD_SOURCE/'pretrained/openvla-7b')
    base=OpenVLAInference(**get_policy_config('openvla',checkpoint,args.task,{},False))
    for seed in parse_seeds(args.seeds):
        out=ARTIFACT/'closed_loop/episodes'/args.task/args.arm
        sp=out/f'episode_{seed:03d}_summary.json'; apath=out/f'episode_{seed:03d}_arrays.npz'
        vp=ARTIFACT/'closed_loop/videos'/args.task/args.arm/f'episode_{seed:03d}.mp4'
        if sp.exists() and apath.exists() and vp.exists(): continue
        with (CANONICAL/'snapshots'/args.task/f'seed_{seed:03d}.pkl').open('rb') as fh: snapshot=pickle.load(fh)
        obs,state_sha,rgb_sha=restore_snapshot(env,seed,snapshot); instruction=env.unwrapped.get_language_instruction()
        present=resolve_present_phrases(env); plan=instruction_set(instruction)
        if plan['kind'] in ('pick_object','move_near'): plan=swap_for_scene(instruction,present)
        cfg=selector_config(args.arm,plan); policy=build_policy(base,args.task,args.arm,plan)
        policy._episode_trace=[]; policy._episode_logits=[]; policy._episode_seed=seed; policy._selector_step=0; policy.reset(instruction,seed=seed)
        t0=time.monotonic(); result,actions,infos=run_loop(env,policy,instruction,obs,vp); runtime=time.monotonic()-t0
        trace=policy._episode_trace
        checks={
          'all_feature_equal':all(x.get('feature_equal') is True for x in trace),
          'all_guided_prefix':all(x.get('guided_prefix') is True for x in trace),
          'all_coverage_exact':all(x.get('coverage_exact') is True for x in trace),
          'all_non_target_bit_identical':all(x.get('non_target_bit_identical') is True for x in trace),
          'all_layer_11':all(x.get('attention_layers')==list(ATTENTION_LAYERS) for x in trace),
          'all_real_instruction_decode':all(x.get('instruction')==instruction for x in trace),
          'all_contrast_instruction_locked':all(x.get('selector_contrast_instruction')==cfg['contrast_instruction'] for x in trace),
          'all_eta_locked':all(abs(float(x.get('selector_difference_eta'))-cfg['eta'])<1e-12 for x in trace),
          'all_epsilon_locked':all(abs(float(x.get('selector_difference_epsilon'))-EPSILON)<1e-15 for x in trace),
        }
        checks['technical_pass']=bool(trace) and all(checks.values())
        if not checks['technical_pass']: raise RuntimeError(f'audit failed {checks}')
        summary={'protocol':'PROMPT_ATTN_SEMANTIC_DIFFERENCE_V1','task':args.task,'seed':seed,'arm':args.arm,
          'instruction':instruction,'contrast_instruction':cfg['contrast_instruction'],'contrast_kind':cfg['contrast_kind'],
          'eta':cfg['eta'],'epsilon':EPSILON,'success':bool(result.get('success',False)),'result':jsonable(result),
          'control_steps':len(infos),'runtime_seconds':runtime,'worker_id':args.worker_id,'environment_id':env_id,
          'initial_state_sha256':state_sha,'initial_rgb_sha256':rgb_sha,'canonical_snapshot_sha256':snapshot_sha(snapshot),
          'mean_m_t':float(np.mean([x['num_tokens'] for x in trace])),
          'mean_correct_token_retention':float(np.mean([x['correct_prompt_retention'] for x in trace])),
          'mean_feature_perturbation_norm':float(np.mean([x['feature_perturbation_norm'] for x in trace])),
          'mean_centered_logit_residual_norm':float(np.mean([x['centered_logit_residual_norm'] for x in trace])),
          'selector_trace':jsonable(trace),'video_path':str(vp.relative_to(ARTIFACT)),'video_sha256':sha256(vp),**checks}
        out.mkdir(parents=True,exist_ok=True); write_arrays(apath,policy._episode_logits,actions); atomic_json(sp,summary)
        print(json.dumps({'task':args.task,'seed':seed,'arm':args.arm,'success':summary['success'],'seconds':round(runtime,1)}),flush=True)
    env.close()
if __name__=='__main__': main()
