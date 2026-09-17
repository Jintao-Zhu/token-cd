"""Closed-loop L11 visual Top-p adaptive-count rollout."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_attn_l11_count_rollout import TASKS, make_environment, parse_count_sweep_seeds
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import LAMBDA, atomic_json, finite_mean, load_reference, write_arrays
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_L11_VISUAL_TOP_P_V1"
ARM_THRESHOLDS = {"l11_top_p75": .75, "l11_top_p80": .80, "l11_top_p85": .85}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_config(artifact: Path, canonical: Path, matched: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    policy_file = repo / "research/semantic_token_cd/prompt_attn_shr_policy.py"
    rollout_file = Path(__file__).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    code = {"git_commit": commit, "policy_sha256": file_sha(policy_file), "rollout_sha256": file_sha(rollout_file)}
    payload = {
        "protocol_id": PROTOCOL, "purpose": "replace KMeans-derived L11 coverage with clipped visual Top-p",
        "tasks": list(TASKS), "seeds_by_task": {task: list(range(100, 200)) for task in TASKS},
        "new_arms": ARM_THRESHOLDS, "new_episode_count": 1200,
        "reference_arm": {"arm": "l11_matched", "artifact": str(matched)},
        "canonical_snapshot_artifact": str(canonical), "count_bounds": [16, 64],
        "visual_attention_normalization": "R_i / sum over exactly 256 visual-token scores",
        "code_version": code,
        "locked_downstream": {
            "attention_layer": 11, "query": "full instruction excluding special/template/padding tokens",
            "head_query_aggregation": "equal arithmetic mean", "tie_break": "ascending visual-token index",
            "spatial_postprocessing": False, "harmonic": "16x16 four-neighbor Dirichlet beta=0/gamma=1",
            "prefix": "shared clean greedy prefix", "lambda": .5,
            "guided_dimensions": [0, 1, 2, 3, 4, 5], "gripper": "clean positive dimension 6",
            "sampling": False,
        },
    }
    artifact.mkdir(parents=True, exist_ok=True)
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("Top-p config lock differs")
    atomic_json(path, payload)
    return code


def build_policies(base, task: str, arms: tuple[str, ...]) -> dict:
    result = {}
    for arm in arms:
        policy = copy.copy(base); policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, LAMBDA)
        policy.beta = 0.0; policy.selector_mode = "prompt_attention"; policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,); policy.selection_count = None
        policy.selection_top_p = ARM_THRESHOLDS[arm]
        policy.selection_min_count = 16; policy.selection_max_count = 64
        policy.save_prompt_attention = False
        result[arm] = policy
    return result


def audit(trace: list[dict], threshold: float) -> dict:
    checks = {
        "technical_nonempty": bool(trace),
        "all_top_p_locked": bool(trace) and all(abs(x.get("top_p_threshold", -1) - threshold) < 1e-12 for x in trace),
        "all_raw_counts_valid": bool(trace) and all(1 <= x.get("m_raw", 0) <= 256 for x in trace),
        "all_clipped_counts_exact": bool(trace) and all(x.get("actual_selected_count") == min(64, max(16, x["m_raw"])) for x in trace),
        "all_attention_mass_valid": bool(trace) and all(0 < x.get("selected_attention_mass", 0) <= 1 for x in trace),
        "all_bounds_consistent": bool(trace) and all(
            x.get("lower_bound_triggered") == (x["m_raw"] < 16)
            and x.get("upper_bound_triggered") == (x["m_raw"] > 64) for x in trace
        ),
        "all_kmeans_bypassed": bool(trace) and all(not x.get("reference_shr_token_ids") for x in trace),
        "all_feature_equal": bool(trace) and all(x.get("feature_equal") is True for x in trace),
        "all_non_target_equal": bool(trace) and all(x.get("non_target_bit_identical") is True for x in trace),
        "all_reconstruction_finite": bool(trace) and all(x.get("reconstruction_finite") is True for x in trace),
        "all_l11": bool(trace) and all(x.get("attention_layers") == [11] for x in trace),
        "all_shared_prefix": bool(trace) and all(x.get("guided_prefix") is True for x in trace),
        "all_lambda": bool(trace) and all(abs(x.get("lambda", -1) - .5) < 1e-12 for x in trace),
        "all_seven_dims": bool(trace) and all(len(x.get("positive_token_ids", [])) == 7 and len(x.get("final_token_ids", [])) == 7 for x in trace),
        "all_gripper_clean": bool(trace) and all(x["positive_token_ids"][6] == x["final_token_ids"][6] for x in trace),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"Top-p rollout audit failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--arms", default=",".join(ARM_THRESHOLDS))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact=args.artifact.resolve(); canonical=args.canonical.resolve(); matched=args.matched_artifact.resolve()
    code=ensure_config(artifact, canonical, matched)
    arms=tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not arms or any(arm not in ARM_THRESHOLDS for arm in arms): raise ValueError(f"invalid arms: {arms}")
    seeds=parse_count_sweep_seeds(args.seeds)
    env,environment_id=make_environment(args.task)
    checkpoint=str(PCD_SOURCE/"pretrained/openvla-7b")
    config=get_policy_config("openvla",checkpoint,args.task,{},False)
    policies=build_policies(OpenVLAInference(**config),args.task,arms)

    for seed in seeds:
        with (canonical/"snapshots"/args.task/f"seed_{seed:03d}.pkl").open("rb") as handle: snapshot=pickle.load(handle)
        reference=load_reference(canonical,args.task,seed)
        expected=(reference["canonical_snapshot_sha256"],reference["initial_state_sha256"],reference["initial_rgb_sha256"])
        if snapshot_sha(snapshot)!=expected[0]: raise RuntimeError("canonical snapshot mismatch")
        for arm,policy in policies.items():
            out=artifact/"episodes"/args.task/arm
            summary_path=out/f"episode_{seed:03d}_summary.json"; arrays_path=out/f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip":True,"task":args.task,"seed":seed,"arm":arm}),flush=True);continue
            obs,state_sha,rgb_sha=restore_snapshot(env,seed,snapshot)
            if (state_sha,rgb_sha)!=expected[1:]: raise RuntimeError("restored snapshot mismatch")
            instruction=env.unwrapped.get_language_instruction()
            if instruction!=reference["instruction"]: raise RuntimeError("instruction mismatch")
            policy.reset(instruction,seed=seed);policy._episode_trace=[];policy._episode_logits=[]
            started=time.monotonic();result,steps,reason,actions,jerk=run_episode(env,policy,instruction,obs);runtime=time.monotonic()-started
            trace=jsonable(policy._episode_trace);checks=audit(trace,ARM_THRESHOLDS[arm])
            out.mkdir(parents=True,exist_ok=True);write_arrays(arrays_path,policy._episode_logits,actions)
            summary={
                "protocol_id":PROTOCOL,"task":args.task,"environment_id":environment_id,"seed":seed,
                "evaluation_seed":seed,"episode_id":seed,"arm":arm,"top_p_threshold":ARM_THRESHOLDS[arm],
                "count_bounds":[16,64],"attention_layers":[11],"instruction":instruction,
                "success":bool(result["success"]),"result":jsonable(result),"failure_reason":reason,
                "control_steps":steps,"runtime_seconds":runtime,"gpu_id":args.gpu,"worker_id":args.worker_id,
                "canonical_snapshot_sha256":expected[0],"initial_state_sha256":state_sha,"initial_rgb_sha256":rgb_sha,
                "lambda":LAMBDA,"beta":0.0,"kmeans_used":False,"code_version":code,"arrays_file":arrays_path.name,
                "action_jitter_index":jerk,"mean_m_raw":finite_mean(x.get("m_raw") for x in trace),
                "mean_m_t":finite_mean(x.get("actual_selected_count") for x in trace),
                "lower_bound_trigger_rate":finite_mean(float(x.get("lower_bound_triggered")) for x in trace),
                "upper_bound_trigger_rate":finite_mean(float(x.get("upper_bound_triggered")) for x in trace),
                "mean_selected_attention_mass":finite_mean(x.get("selected_attention_mass") for x in trace),
                "mean_centered_logit_residual_norm":finite_mean(x.get("centered_logit_residual_norm") for x in trace),
                "mean_feature_perturbation_relative":finite_mean(x.get("feature_perturbation_relative") for x in trace),
                "mean_guided_change_ratio":finite_mean(x.get("guided_change_ratio") for x in trace),
                "selector_trace":trace,**checks,
            }
            atomic_json(summary_path,summary)
            print(json.dumps({"task":args.task,"seed":seed,"arm":arm,"success":summary["success"],"runtime_seconds":round(runtime,2)}),flush=True)


if __name__=="__main__": main()
