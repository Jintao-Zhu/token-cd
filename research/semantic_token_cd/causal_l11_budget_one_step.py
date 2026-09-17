"""One-step budget intervention followed by a shared Matched continuation."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, clone, flatten_action, get_image_from_maniskill2_obs_dict,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import build_policies as build_count, make_environment
from research.semantic_token_cd.prompt_attn_l11_top_p_rollout import build_policies as build_top_p
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json
from research.semantic_token_cd.rollout_pilot import wrapped_observation


TASK="google_robot_open_drawer"


def restore_mid(env,seed:int,snapshot:dict):
    env.reset(seed=seed);inner=env.unwrapped
    inner.set_state(snapshot["sim_state"].copy());inner.agent.set_state(clone(snapshot["agent_state"]));inner._episode_rng.set_state(clone(snapshot["rng_state"]))
    elapsed=int(snapshot["elapsed_steps"]);inner._elapsed_steps=elapsed
    current=env
    while hasattr(current,"env"):
        if hasattr(current,"_elapsed_steps"): current._elapsed_steps=elapsed
        current=current.env
    return wrapped_observation(env)


def run_branch(env,policy_first,policy_matched,instruction,seed,step,obs):
    policy_first.reset(instruction,seed=seed);policy_first._selector_step=step;policy_first._episode_trace=[];policy_first._episode_logits=[]
    policy_matched.reset(instruction,seed=seed);policy_matched._selector_step=step+1;policy_matched._episode_trace=[];policy_matched._episode_logits=[]
    predicted=truncated=False;infos=[];control=0;first_meta=None
    while not (predicted or truncated):
        image=np.asarray(get_image_from_maniskill2_obs_dict(env,obs),dtype=np.uint8)
        policy=policy_first if control==0 else policy_matched
        _raw,actions,meta=policy.step(image,None,instruction,proprio=obs["agent"]["eef_pos"])
        if control==0:first_meta=meta
        if not isinstance(actions,list):actions=[actions]
        for action in actions:
            obs,_reward,_success,truncated,info=env.step(flatten_action(action));infos.append(info)
            predicted=bool(action["terminate_episode"][0]>0)
            if predicted and not env.unwrapped.is_final_subtask():predicted=False;env.advance_to_next_subtask()
            instruction=env.unwrapped.get_language_instruction()
        control+=1
    success=bool(any(bool(info.get("success",False)) for info in infos))
    return {"success":success,"continuation_control_steps":control,"first_action_meta":first_meta}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--artifact",type=Path,required=True);parser.add_argument("--seeds",required=True);parser.add_argument("--gpu",type=int,choices=(2,3),required=True);args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu);os.environ["TOKENIZERS_PARALLELISM"]="false";os.environ.setdefault("HF_HUB_OFFLINE","1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    root=args.artifact.resolve();manifest=json.loads((root/"statistics/causal_state_manifest.json").read_text());env,_=make_environment(TASK)
    checkpoint=str(PCD_SOURCE/"pretrained/openvla-7b");config=get_policy_config("openvla",checkpoint,TASK,{},False);base=OpenVLAInference(**config)
    matched_first=build_count(base,TASK,("l11_matched",))["l11_matched"]
    matched_cont=build_count(base,TASK,("l11_matched",))["l11_matched"]
    p85_first=build_top_p(base,TASK,("l11_top_p85",))["l11_top_p85"]
    for seed in sorted(set(int(x) for x in args.seeds.split(",") if x.strip())):
        selected=manifest[str(seed)];step=int(selected["state_id"].rsplit("step",1)[1]);state_dir=root/"outcome_states"/TASK/f"seed_{seed:03d}"
        with (state_dir/f"step_{step:03d}_snapshot.pkl").open("rb") as handle:snapshot=pickle.load(handle)
        source=json.loads((state_dir/f"step_{step:03d}.json").read_text());results={}
        for name,first in (("matched_action_then_matched",matched_first),("top_p85_action_then_matched",p85_first)):
            obs=restore_mid(env,seed,snapshot)
            results[name]=run_branch(env,first,matched_cont,snapshot["instruction"],seed,step,obs)
        original=bool(source["closed_loop_outcomes"]["l11_matched"])
        technical=results["matched_action_then_matched"]["success"]==original
        out=root/"causal_one_step"/TASK;out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/f"seed_{seed:03d}_step_{step:03d}.json",{
            "protocol_id":"L11_BUDGET_ONE_STEP_CAUSAL_V1","task":TASK,"seed":seed,"step":step,
            "phase":selected["phase"],"category":selected["category"],"selection_reason":selected,
            "original_matched_success":original,"results":results,
            "matched_continuation_reproduced_original_outcome":technical,
            "interpretation_valid":technical,
        })
        print(json.dumps({"seed":seed,"step":step,"original":original,
                          "matched_branch":results["matched_action_then_matched"]["success"],
                          "p85_branch":results["top_p85_action_then_matched"]["success"],"valid":technical}),flush=True)


if __name__=="__main__":main()
