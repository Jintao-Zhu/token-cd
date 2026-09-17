"""Replay selected Matched trajectories and evaluate Top-p on identical states."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, clone, flatten_action, get_image_from_maniskill2_obs_dict,
    jsonable, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import build_policies as build_count, make_environment
from research.semantic_token_cd.prompt_attn_l11_top_p_rollout import build_policies as build_top_p
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json, load_reference


TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
ARMS = ("l11_matched", "l11_top_p80", "l11_top_p85")


def parse_seeds(specification: str) -> list[int]:
    values = sorted(set(int(value) for value in specification.split(",") if value.strip()))
    if not values or any(value < 100 or value > 199 for value in values):
        raise ValueError("seeds must be within 100..199")
    return values


def target_steps(length: int) -> dict[int, str]:
    return {int(round((length - 1) * fraction)): phase
            for fraction, phase in zip((.1, .5, .9), ("early", "middle", "late"))}


def current_snapshot(env) -> dict:
    inner = env.unwrapped
    return {
        "sim_state": np.asarray(inner.get_state()).copy(),
        "agent_state": clone(inner.agent.get_state()),
        "rng_state": clone(inner._episode_rng.get_state()),
        "elapsed_steps": int(inner._elapsed_steps),
        "instruction": inner.get_language_instruction(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--top-p-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact=args.artifact.resolve(); canonical=args.canonical.resolve()
    matched_artifact=args.matched_artifact.resolve(); top_p_artifact=args.top_p_artifact.resolve()
    env,_=make_environment(args.task)
    checkpoint=str(PCD_SOURCE/"pretrained/openvla-7b")
    config=get_policy_config("openvla",checkpoint,args.task,{},False)
    base=OpenVLAInference(**config)
    policies={**build_count(base,args.task,("l11_matched",)),
              **build_top_p(base,args.task,("l11_top_p80","l11_top_p85"))}
    for policy in policies.values(): policy.save_prompt_attention=True

    for seed in parse_seeds(args.seeds):
        matched_summary_path=matched_artifact/"episodes"/args.task/"l11_matched"/f"episode_{seed:03d}_summary.json"
        matched_summary=json.loads(matched_summary_path.read_text())
        matched_arrays=np.load(matched_summary_path.with_name(f"episode_{seed:03d}_arrays.npz"))
        outcomes={"l11_matched":bool(matched_summary["success"])}
        for arm in ("l11_top_p80","l11_top_p85"):
            outcomes[arm]=bool(json.loads((top_p_artifact/"episodes"/args.task/arm/f"episode_{seed:03d}_summary.json").read_text())["success"])
        with (canonical/"snapshots"/args.task/f"seed_{seed:03d}.pkl").open("rb") as handle: snapshot=pickle.load(handle)
        reference=load_reference(canonical,args.task,seed)
        if snapshot_sha(snapshot)!=reference["canonical_snapshot_sha256"]: raise RuntimeError("canonical snapshot mismatch")
        obs,state_sha,rgb_sha=restore_snapshot(env,seed,snapshot)
        if (state_sha,rgb_sha)!=(reference["initial_state_sha256"],reference["initial_rgb_sha256"]): raise RuntimeError("restore mismatch")
        instruction=env.unwrapped.get_language_instruction(); driver=policies["l11_matched"]
        driver.reset(instruction,seed=seed); driver._episode_trace=[]; driver._episode_logits=[]
        targets=target_steps(len(matched_summary["selector_trace"])); predicted_terminated=False; truncated=False
        action_index=0; control_step=0
        while not (predicted_terminated or truncated):
            current_instruction=env.unwrapped.get_language_instruction()
            image=np.asarray(get_image_from_maniskill2_obs_dict(env,obs),dtype=np.uint8)
            _raw,actions,_meta=driver.step(image,None,current_instruction,proprio=obs["agent"]["eef_pos"])
            driver_record=driver._episode_logits[-1]; driver_meta=driver._episode_trace[-1]
            expected_positive=matched_arrays["positive"][control_step]
            clean_logits_max_abs_diff=float(np.max(np.abs(
                driver_record["positive"].astype(np.float32)-expected_positive.astype(np.float32)
            )))
            if not np.array_equal(driver_record["positive"].argmax(axis=1),expected_positive.argmax(axis=1)):
                raise RuntimeError(
                    f"Matched clean action-token mismatch: {args.task}/{seed}/{control_step}, "
                    f"max_logit_diff={clean_logits_max_abs_diff}"
                )
            if not np.array_equal(driver_record["selected_mask"],matched_arrays["selected_mask"][control_step]):
                raise RuntimeError(f"Matched mask mismatch: {args.task}/{seed}/{control_step}")

            if control_step in targets:
                phase=targets[control_step]; out=artifact/"outcome_states"/args.task/f"seed_{seed:03d}"
                out_json=out/f"step_{control_step:03d}.json"; out_npz=out_json.with_suffix(".npz")
                metrics={"l11_matched":driver_meta}; arrays={"image":image}
                for key,value in driver_record.items(): arrays[f"l11_matched__{key}"]=value
                for arm in ("l11_top_p80","l11_top_p85"):
                    policy=policies[arm]; policy.reset(current_instruction,seed=seed); policy._selector_step=control_step
                    policy._episode_trace=[];policy._episode_logits=[]
                    policy.step(image,None,current_instruction,proprio=obs["agent"]["eef_pos"])
                    metrics[arm]=policy._episode_trace[-1]
                    for key,value in policy._episode_logits[-1].items(): arrays[f"{arm}__{key}"]=value
                if not all(np.array_equal(arrays["l11_matched__positive"],arrays[f"{arm}__positive"]) for arm in ARMS[1:]):
                    raise RuntimeError("same-state clean logits differ")
                sets={arm:set(metrics[arm]["selected_token_ids"]) for arm in ARMS}
                if not all(sets["l11_matched"]<=sets[arm] or sets[arm]<=sets["l11_matched"] for arm in ARMS[1:]):
                    raise RuntimeError("same-ranking masks are not nested")
                out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out_npz,**arrays)
                with (out/f"step_{control_step:03d}_snapshot.pkl").open("wb") as handle: pickle.dump(current_snapshot(env),handle)
                atomic_json(out_json,{"protocol_id":"L11_BUDGET_OUTCOME_ALIGNED_SAME_STATE_V1",
                    "task":args.task,"seed":seed,"control_step":control_step,"phase":phase,
                    "instruction":current_instruction,"closed_loop_outcomes":outcomes,
                    "metrics":jsonable(metrics),"arrays_file":out_npz.name,
                    "snapshot_file":f"step_{control_step:03d}_snapshot.pkl",
                    "matched_trajectory_replay_source":str(matched_summary_path),
                    "same_state_clean_logits_bit_equal":True,"masks_nested":True,
                    "matched_replay_clean_logits_max_abs_diff":clean_logits_max_abs_diff})

            if not isinstance(actions,list): actions=[actions]
            for action in actions:
                executed=flatten_action(action)
                expected_action=matched_arrays["executed_actions"][action_index]
                max_action_diff=float(np.max(np.abs(np.asarray(executed)-np.asarray(expected_action))))
                if not np.allclose(executed,expected_action,rtol=0.0,atol=1e-7):
                    raise RuntimeError(
                        f"executed action mismatch: {args.task}/{seed}/{action_index}, max_diff={max_action_diff}"
                    )
                obs,_reward,_success,truncated,_info=env.step(executed);action_index+=1
                predicted_terminated=bool(action["terminate_episode"][0]>0)
                if predicted_terminated and not env.unwrapped.is_final_subtask():
                    predicted_terminated=False;env.advance_to_next_subtask()
            control_step+=1
        if control_step!=len(matched_summary["selector_trace"]): raise RuntimeError("control-step count mismatch")
        print(json.dumps({"task":args.task,"seed":seed,"outcomes":outcomes,"states":len(targets)}),flush=True)


if __name__=="__main__": main()
