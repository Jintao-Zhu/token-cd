"""Held-out fixed-arm rollouts for L11 episode-start lambda routing."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.l11_lambda_router_protocol import (
    PROTOCOL,
    TASKS,
    arm_arrays_path,
    arm_summary_path,
    atomic_json,
    parse_seeds,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import finite_mean, load_reference, write_arrays
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


ARM_LAMBDAS = {"l11_positive_only": 0.0, "l11_fixed_025": 0.25}


def build_policies(base, task: str, arms: tuple[str, ...]) -> dict:
    policies = {}
    for arm in arms:
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, ARM_LAMBDAS[arm])
        policy.beta = 0.0
        policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,)
        policy.selection_count = None
        policy.save_prompt_attention = False
        policies[arm] = policy
    return policies


def audit(trace: list[dict], expected: float) -> dict:
    checks = {
        "technical_nonempty": bool(trace),
        "all_lambda_exact": bool(trace) and all(abs(float(row.get("lambda", -1)) - expected) < 1e-12 for row in trace),
        "all_l11": bool(trace) and all(row.get("attention_layers") == [11] for row in trace),
        "all_matched_coverage": bool(trace) and all(row.get("actual_selected_count") == row.get("m_t") for row in trace),
        "all_feature_equal": bool(trace) and all(row.get("feature_equal") is True for row in trace),
        "all_non_target_equal": bool(trace) and all(row.get("non_target_bit_identical") is True for row in trace),
        "all_reconstruction_finite": bool(trace) and all(row.get("reconstruction_finite") is True for row in trace),
        "all_seven_dims": bool(trace) and all(len(row.get("positive_token_ids", [])) == 7 and len(row.get("final_token_ids", [])) == 7 for row in trace),
        "all_gripper_equal_positive": bool(trace) and all(row["final_token_ids"][6] == row["positive_token_ids"][6] for row in trace),
    }
    if expected == 0.0:
        checks.update({
            "all_final_tokens_equal_positive": all(row["final_token_ids"] == row["positive_token_ids"] for row in trace),
            "all_continuous_actions_equal_positive": all(row.get("guided_clean_action_l2") == 0.0 for row in trace),
        })
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"held-out fixed-arm audit failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--arms", default="l11_positive_only,l11_fixed_025")
    args = parser.parse_args()

    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not arms or any(arm not in ARM_LAMBDAS for arm in arms):
        raise ValueError(arms)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    canonical = args.canonical.resolve()
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task, arms)
    for seed in parse_seeds(args.seeds):
        with (canonical / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(canonical, args.task, seed)
        if snapshot_sha(snapshot) != reference["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        for arm, policy in policies.items():
            summary_path = arm_summary_path(artifact, args.task, arm, seed)
            arrays_path = arm_arrays_path(artifact, args.task, arm, seed)
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if (state_sha, rgb_sha) != (reference["initial_state_sha256"], reference["initial_rgb_sha256"]):
                raise RuntimeError("restored snapshot mismatch")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError("instruction mismatch")
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs)
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, ARM_LAMBDAS[arm])
            arrays_path.parent.mkdir(parents=True, exist_ok=True)
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "arm": arm,
                "lambda": ARM_LAMBDAS[arm],
                "attention_layers": [11],
                "instruction": instruction,
                "success": bool(result["success"]),
                "result": jsonable(result),
                "failure_reason": reason,
                "control_steps": steps,
                "runtime_seconds": time.monotonic() - started,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": reference["canonical_snapshot_sha256"],
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "beta": 0.0,
                "arrays_file": arrays_path.name,
                "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(row.get("actual_selected_count") for row in trace),
                "selector_trace": trace,
                **checks,
            }
            atomic_json(summary_path, summary)
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": summary["success"]}), flush=True)


if __name__ == "__main__":
    main()
