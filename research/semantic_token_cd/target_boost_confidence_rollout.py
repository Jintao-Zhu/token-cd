"""Held-out closed-loop rollout for confidence-gated Positive-Boost."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.target_boost_confidence_common import build_policy
from research.semantic_token_cd.target_boost_confidence_protocol import (
    ARMS, ARTIFACT, CANONICAL, PCD_SOURCE, PROTOCOL, SEEDS, TASKS, arm_config,
)
from research.semantic_token_cd.target_specific_rollout import file_sha, run_loop
from research.semantic_token_cd.xswap_protocol import atomic_json
from research.semantic_token_cd.xswap_rollout import make_environment


def parse_seeds(spec: str) -> list[int]:
    out=[]
    for part in spec.split(","):
        if "-" in part:
            lo,hi=map(int,part.split("-",1));out.extend(range(lo,hi+1))
        elif part.strip():out.append(int(part))
    result=sorted(set(out))
    if not result or any(x not in SEEDS for x in result):raise ValueError("seeds must be 100..199")
    return result


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--task",choices=TASKS,required=True)
    p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--seeds",required=True)
    p.add_argument("--gpu",type=int,choices=(2,3),required=True);p.add_argument("--worker-id",required=True);a=p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu);os.environ["HF_HUB_OFFLINE"]="1";os.environ["TOKENIZERS_PARALLELISM"]="false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    env,env_id=make_environment(a.task,a.gpu)
    base=OpenVLAInference(**get_policy_config("openvla",str(PCD_SOURCE/"pretrained/openvla-7b"),a.task,{},False))
    cfg=arm_config(a.task,a.arm)
    try:
      for seed in parse_seeds(a.seeds):
        out=ARTIFACT/"episodes"/a.task/a.arm;summary_path=out/f"episode_{seed:03d}_summary.json"
        arrays_path=out/f"episode_{seed:03d}_arrays.npz";video_path=ARTIFACT/"videos"/a.task/a.arm/f"episode_{seed:03d}.mp4"
        if summary_path.exists() and arrays_path.exists() and video_path.exists():continue
        with (CANONICAL/"snapshots"/a.task/f"seed_{seed:03d}.pkl").open("rb") as h:snapshot=pickle.load(h)
        obs,state_sha,rgb_sha=restore_snapshot(env,seed,snapshot);instruction=env.unwrapped.get_language_instruction()
        policy=build_policy(base,a.task,a.arm);policy.reset(instruction,seed=seed);policy._episode_seed=seed;policy._selector_step=0
        policy._episode_trace=[];policy._episode_logits=[];started=time.monotonic()
        result,actions,infos=run_loop(env,policy,instruction,obs,video_path);trace=policy._episode_trace
        checks={
          "all_feature_equal":all(x.get("feature_equal") is True for x in trace),
          "all_guided_prefix":all(x.get("guided_prefix") is True for x in trace),
          "all_coverage_exact":all(x.get("coverage_exact") is True for x in trace),
          "all_non_target_equal":all(x.get("non_target_bit_identical") is True for x in trace),
          "all_layer_11":all(x.get("attention_layers")==[11] for x in trace),
          "all_formula_locked":all(x.get("selector_difference_formula")==cfg["formula"] for x in trace),
          "all_threshold_locked":all(x.get("selector_difference_confidence_threshold")==cfg["confidence_threshold"] for x in trace),
          "all_lambda_0p5":all(abs(float(x.get("lambda"))-.5)<1e-12 for x in trace),
          "all_gripper_clean":all(x["positive_token_ids"][6]==x["final_token_ids"][6] for x in trace),
        };checks["technical_pass"]=bool(trace) and all(checks.values())
        if not checks["technical_pass"]:raise RuntimeError(checks)
        payload={"executed_actions":actions}
        for key in ("positive","negative","selected_mask","correct_prompt_mask","reference_shr_mask",
                    "correct_attention_probability","contrast_attention_probability","attention_difference",
                    "selector_score","relative_difference_confidence","confidence_gate_mask"):
          if trace and all(key in x for x in policy._episode_logits):payload[key]=np.stack([x[key] for x in policy._episode_logits])
        arrays_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(arrays_path,**payload)
        summary={"protocol_id":PROTOCOL,"task":a.task,"seed":seed,"arm":a.arm,"instruction":instruction,
          "generic_instruction":cfg["contrast_instruction"],"formula":cfg["formula"],"eta":cfg["eta"],
          "confidence_threshold":cfg["confidence_threshold"],"success":bool(result.get("success",False)),
          "result":jsonable(result),"control_steps":len(infos),"runtime_seconds":time.monotonic()-started,
          "worker_id":a.worker_id,"environment_id":env_id,"initial_state_sha256":state_sha,
          "initial_rgb_sha256":rgb_sha,"canonical_snapshot_sha256":snapshot_sha(snapshot),
          "mean_m":float(np.mean([x["m_t"] for x in trace])),
          "mean_correct_retention":float(np.mean([x["correct_prompt_retention"] for x in trace])),
          "mean_feature_perturbation_norm":float(np.mean([x["feature_perturbation_norm"] for x in trace])),
          "mean_residual_norm":float(np.mean([x["centered_logit_residual_norm"] for x in trace])),
          "mean_gate_eligible_fraction":float(np.mean([x.get("confidence_gate_eligible_fraction",1.0) for x in trace])),
          "video_sha256":file_sha(video_path),**checks}
        atomic_json(summary_path,summary);print(json.dumps({"task":a.task,"seed":seed,"arm":a.arm,
          "success":summary["success"],"seconds":round(summary["runtime_seconds"],1)}),flush=True)
    finally:env.close()


if __name__=="__main__":main()
