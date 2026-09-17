"""Equivalence and isolation gates for Target-Boost confidence gating."""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

from research.semantic_token_cd.distractor_rollout import get_image_from_maniskill2_obs_dict, restore_snapshot
from research.semantic_token_cd.prompt_attn_l11_count_rollout import build_policies
from research.semantic_token_cd.target_boost_confidence_common import build_policy
from research.semantic_token_cd.target_boost_confidence_protocol import ARTIFACT, BASELINE, CANONICAL, PCD_SOURCE, TASKS, write_lock
from research.semantic_token_cd.xswap_rollout import make_environment
from research.semantic_token_cd.xswap_protocol import atomic_json


def call(policy, image, instruction, proprio, seed=100):
    policy.reset(instruction, seed=seed); policy._episode_seed=seed; policy._selector_step=0
    policy._episode_trace=[]; policy._episode_logits=[]
    _raw,_actions,meta=policy.step(image,None,instruction,proprio=proprio)
    return meta,policy._episode_logits[-1]


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ["TOKENIZERS_PARALLELISM"]="false"
    write_lock(); task=TASKS[0]; seed=100; env,_=make_environment(task,2)
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    base=OpenVLAInference(**get_policy_config("openvla",str(PCD_SOURCE/"pretrained/openvla-7b"),task,{},False))
    with (CANONICAL/"snapshots"/task/f"seed_{seed:03d}.pkl").open("rb") as h:snapshot=pickle.load(h)
    obs,_,_=restore_snapshot(env,seed,snapshot);instruction=env.unwrapped.get_language_instruction()
    image=np.asarray(get_image_from_maniskill2_obs_dict(env,obs),dtype=np.uint8);proprio=obs["agent"]["eef_pos"]
    correct=build_policies(base,task,("l11_matched",))["l11_matched"]
    ungated=build_policy(base,task,"positive_boost_0p5")
    gate0=build_policy(base,task,"confidence_gate_0p4");gate0.selector_difference_confidence_threshold=0.0
    disabled=build_policy(base,task,"confidence_gate_0p4");disabled.selector_difference_confidence_threshold=1.000001
    cm,cr=call(correct,image,instruction,proprio);um,ur=call(ungated,image,instruction,proprio)
    gm,gr=call(gate0,image,instruction,proprio);dm,dr=call(disabled,image,instruction,proprio)
    historical=np.load(BASELINE/"episodes"/task/"l11_matched"/f"episode_{seed:03d}_arrays.npz",allow_pickle=False)
    checks={
      "gate_zero_mask_equals_ungated":gm["selected_token_ids"]==um["selected_token_ids"],
      "gate_zero_negative_equals_ungated":np.array_equal(gr["negative"],ur["negative"]),
      "disabled_mask_equals_correct":dm["selected_token_ids"]==cm["selected_token_ids"],
      "disabled_negative_equals_correct":np.array_equal(dr["negative"],cr["negative"]),
      "clean_positive_all_equal":all(np.array_equal(cr["positive"],x["positive"]) for x in (ur,gr,dr)),
      "matched_m_all_equal":len({cm["m_t"],um["m_t"],gm["m_t"],dm["m_t"]})==1,
      "lambda_all_0p5":all(abs(x["lambda"]-.5)<1e-12 for x in (cm,um,gm,dm)),
      "gripper_all_clean":all(x["positive_token_ids"][6]==x["final_token_ids"][6] for x in (cm,um,gm,dm)),
      "historical_matched_mask_exact":np.array_equal(cr["selected_mask"],historical["selected_mask"][0]),
      "historical_matched_positive_exact":np.array_equal(cr["positive"],historical["positive"][0]),
      "historical_matched_negative_exact":np.array_equal(cr["negative"],historical["negative"][0]),
    }
    checks["pass"]=all(checks.values());atomic_json(ARTIFACT/"PREFLIGHT.json",checks)
    (ARTIFACT/("PREFLIGHT_PASS" if checks["pass"] else "PREFLIGHT_FAIL")).write_text(json.dumps(checks)+"\n")
    env.close();print(json.dumps(checks));
    if not checks["pass"]:raise RuntimeError(checks)


if __name__=="__main__":main()
