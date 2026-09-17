"""Closed-loop worker for the single PA-Full-Joint-Top2K arm."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_action_full_joint_protocol import (
    ACTION_LAYERS, ARM, ARTIFACT, CANONICAL, LAMBDA, PCD_SOURCE, PROMPT_LAYERS,
    PROTOCOL, TASKS, atomic_json,
)
from research.semantic_token_cd.prompt_action_rerank_rollout import (
    file_sha, parse_seeds, run_loop, write_arrays,
)
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.xswap_rollout import make_environment


def build_policy(base, task):
    policy = copy.copy(base); policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0; policy.selector_mode = "prompt_attention"; policy.task_index = TASK_INDEX[task]
    policy.attention_layers = PROMPT_LAYERS; policy.action_attention_layers = ACTION_LAYERS
    policy.selection_count = None; policy.selection_top_p = None
    policy.selector_instruction = None; policy.selector_contrast_instruction = None
    policy.selector_difference_eta = None; policy.complement_arm = None
    policy.prompt_action_rerank_multiplier = None; policy.prompt_action_bounded_mode = None
    policy.prompt_action_full_joint = True; policy.save_prompt_attention = True
    return policy


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--task",required=True,choices=TASKS)
    parser.add_argument("--seeds",required=True); parser.add_argument("--gpu",required=True,type=int,choices=tuple(range(8)))
    parser.add_argument("--worker-id",default="manual"); parser.add_argument("--artifact",type=Path,default=ARTIFACT)
    args=parser.parse_args(); artifact=args.artifact.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    env,env_id=make_environment(args.task,args.gpu); checkpoint=str(PCD_SOURCE/"pretrained/openvla-7b")
    base=OpenVLAInference(**get_policy_config("openvla",checkpoint,args.task,{},False))
    policy_sha=file_sha(Path(__file__).with_name("prompt_attn_shr_policy.py"))
    for seed in parse_seeds(args.seeds):
        out=artifact/"closed_loop/episodes"/args.task/ARM; summary=out/f"episode_{seed:03d}_summary.json"; arrays=out/f"episode_{seed:03d}_arrays.npz"
        video=artifact/"closed_loop/videos"/args.task/ARM/f"episode_{seed:03d}.mp4"
        if summary.exists() and arrays.exists() and video.exists(): continue
        with (CANONICAL/"snapshots"/args.task/f"seed_{seed:03d}.pkl").open("rb") as handle: snapshot=pickle.load(handle)
        observation,state_sha,rgb_sha=restore_snapshot(env,seed,snapshot); instruction=env.unwrapped.get_language_instruction()
        policy=build_policy(base,args.task); policy._episode_trace=[]; policy._episode_logits=[]; policy._episode_seed=seed; policy._selector_step=0; policy.reset(instruction,seed=seed)
        start=time.monotonic(); result,actions,infos=run_loop(env,policy,instruction,observation,video); runtime=time.monotonic()-start; trace=policy._episode_trace
        checks={
            "all_prompt_l11":all(x.get("attention_layers")==[11] for x in trace),
            "all_action_layers_16_31":all(x.get("action_attention_layers")==list(range(16,32)) for x in trace),
            "all_exact_m":all(x.get("coverage_exact") is True and x.get("full_joint_exact_m") is True for x in trace),
            "all_inside_top2k":all(x.get("full_joint_subset_of_candidate_pool") is True for x in trace),
            "all_candidate_sizes":all(x.get("candidate_pool_size")==min(256,2*x.get("m_t")) for x in trace),
            "all_feature_equal":all(x.get("feature_equal") is True for x in trace),
            "all_non_target_equal":all(x.get("non_target_bit_identical") is True for x in trace),
            "all_guided_prefix":all(x.get("guided_prefix") is True for x in trace),
        }; checks["technical_pass"]=bool(trace) and all(checks.values())
        if not checks["technical_pass"]: raise RuntimeError(f"Full-Joint audit failure: {checks}")
        avg=lambda k:float(np.mean([x[k] for x in trace]))
        row={"protocol":PROTOCOL,"task":args.task,"seed":seed,"arm":ARM,"instruction":instruction,
             "success":bool(result.get("success",False)),"result":jsonable(result),"control_steps":len(infos),
             "runtime_seconds":runtime,"worker_id":args.worker_id,"environment_id":env_id,
             "initial_state_sha256":state_sha,"initial_rgb_sha256":rgb_sha,"canonical_snapshot_sha256":snapshot_sha(snapshot),
             "policy_code_sha256":policy_sha,"mean_m":avg("m_t"),"mean_candidate_pool_size":avg("candidate_pool_size"),
             "mean_overlap_ratio":avg("full_joint_overlap_ratio"),"mean_jaccard":avg("full_joint_jaccard"),
             "mean_feature_perturbation_norm":avg("feature_perturbation_norm"),"mean_centered_residual_norm":avg("centered_logit_residual_norm"),
             "mean_guided_changed_dims":avg("guided_changed_dims"),"candidate_saturation_rate":avg("candidate_pool_saturated"),
             "selector_trace":jsonable(trace),"video_path":str(video.relative_to(artifact)),"video_sha256":file_sha(video),**checks}
        write_arrays(arrays,policy._episode_logits,actions); atomic_json(summary,row)
        print(json.dumps({"task":args.task,"seed":seed,"success":row["success"],"seconds":round(runtime,1)}),flush=True)
    env.close()


if __name__=="__main__": main()
