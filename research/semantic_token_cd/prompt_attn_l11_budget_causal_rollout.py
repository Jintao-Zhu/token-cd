"""Closed-loop pilot for causal manipulations of the L11 matched budget."""
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

import numpy as np

from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import (
    KMEANS_K, KMEANS_SEED, LAMBDA, atomic_json, finite_mean, load_reference, write_arrays,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_L11_MATCHED_BUDGET_CAUSAL_PILOT_V1"
SCHEDULE_ARMS = ("global_shuffle", "within_task_shuffle", "episode_fixed")
SCALE_ARMS = {
    "matched_scale_050": 0.50,
    "matched_scale_075": 0.75,
    "matched_scale_125": 1.25,
    "matched_scale_150": 1.50,
}
ARMS = SCHEDULE_ARMS + tuple(SCALE_ARMS)


def parse_seeds(specification: str) -> list[int]:
    seeds = []
    for part in specification.split(","):
        part = part.strip()
        if "-" in part:
            low, high = map(int, part.split("-", 1))
            seeds.extend(range(low, high + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(seed < 100 or seed > 199 for seed in result):
        raise ValueError("budget causal seeds must be within 100..199")
    return result


def make_environment(task: str):
    import gymnasium as gym
    import simpler_env

    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd", prepackaged_config=True, distractor_config="less",
            renderer_kwargs=renderer_kwargs,
        ), "GraspSingleOpenedCokeCanDistractorInScene-v0"
    environment_id, base_kwargs = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base_kwargs)
    kwargs.update({"prepackaged_config": True, "renderer_kwargs": renderer_kwargs})
    return gym.make(environment_id, obs_mode="rgbd", **kwargs), environment_id


def build_policies(base, task: str, arms: tuple[str, ...]) -> dict:
    policies = {}
    for arm in arms:
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, LAMBDA)
        policy.beta = 0.0
        policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,)
        policy.selection_count = None
        policy.selection_top_p = None
        policy.selection_budget_schedule = None
        policy.selection_budget_scale = SCALE_ARMS.get(arm, 1.0)
        policy.selection_budget_label = arm if arm in SCHEDULE_ARMS else None
        policy.save_prompt_attention = False
        policies[arm] = policy
    return policies


def audit(trace: list[dict], arm: str, expected_schedule: list[int] | None) -> dict:
    scale = SCALE_ARMS.get(arm)
    checks = {
        "technical_nonempty": bool(trace),
        "all_feature_equal": bool(trace) and all(item.get("feature_equal") is True for item in trace),
        "all_guided_prefix": bool(trace) and all(item.get("guided_prefix") is True for item in trace),
        "all_reconstruction_finite": bool(trace) and all(item.get("reconstruction_finite") is True for item in trace),
        "all_lambda_locked": bool(trace) and all(abs(float(item.get("lambda", -1)) - .5) < 1e-12 for item in trace),
        "all_beta_zero": bool(trace) and all(abs(float(item.get("beta", -1))) < 1e-12 for item in trace),
        "all_coverage_exact": bool(trace) and all(item.get("coverage_exact") is True for item in trace),
        "all_non_target_bit_identical": bool(trace) and all(item.get("non_target_bit_identical") is True for item in trace),
        "all_l11_locked": bool(trace) and all(item.get("attention_layers") == [11] for item in trace),
        "all_post_softmax": bool(trace) and all(item.get("attention_post_softmax") is True for item in trace),
        "all_visual_keys_locked": bool(trace) and all(item.get("visual_key_indices") == [1, 256] for item in trace),
        "all_gripper_clean": bool(trace) and all(
            item["positive_token_ids"][6] == item["final_token_ids"][6] for item in trace
        ),
    }
    if expected_schedule is not None:
        checks.update({
            "schedule_counts_exact": all(
                item["actual_selected_count"] == expected_schedule[index]
                for index, item in enumerate(trace)
            ),
            "schedule_kmeans_bypassed": all(not item.get("reference_shr_token_ids") for item in trace),
            "schedule_label_exact": all(item.get("budget_schedule_label") == arm for item in trace),
        })
    else:
        checks.update({
            "scaled_counts_exact": all(
                item["actual_selected_count"]
                == int(np.clip(round(float(scale) * item["unscaled_matched_m_t"]), 1, 256))
                for item in trace
            ),
            "scale_locked": all(abs(float(item.get("budget_scale", -1)) - float(scale)) < 1e-12 for item in trace),
            "scale_kmeans_used": all(bool(item.get("reference_shr_token_ids")) for item in trace),
        })
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    return checks


def code_version(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    try:
        digest.update(subprocess.check_output(["git", "diff", "--", *map(str, paths)]))
    except Exception:
        pass
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--schedule-file", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()

    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"invalid arms: {arms}")
    seeds = parse_seeds(args.seeds)
    schedules = json.loads(args.schedule_file.read_text())["schedules"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    environment, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task, arms)
    version = code_version([Path(__file__), Path(__file__).with_name("prompt_attn_shr_policy.py")])

    for seed in seeds:
        with (source / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(source, args.task, seed)
        expected = (
            reference["canonical_snapshot_sha256"], reference["initial_state_sha256"],
            reference["initial_rgb_sha256"],
        )
        if snapshot_sha(snapshot) != expected[0]:
            raise RuntimeError("canonical snapshot mismatch")
        for arm, policy in policies.items():
            output = artifact / "episodes" / args.task / arm
            summary_path = output / f"episode_{seed:03d}_summary.json"
            arrays_path = output / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            observation, state_sha, rgb_sha = restore_snapshot(environment, seed, snapshot)
            if (state_sha, rgb_sha) != expected[1:]:
                raise RuntimeError("restored snapshot mismatch")
            instruction = environment.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError("instruction mismatch")
            schedule = schedules[arm][args.task][str(seed)] if arm in SCHEDULE_ARMS else None
            policy.selection_budget_schedule = tuple(schedule) if schedule is not None else None
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(environment, policy, instruction, observation)
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, arm, schedule)
            output.mkdir(parents=True, exist_ok=True)
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task, "environment_id": environment_id,
                "seed": seed, "evaluation_seed": seed, "episode_id": seed, "arm": arm,
                "attention_layers": [11], "instruction": instruction,
                "success": bool(result["success"]), "result": jsonable(result),
                "failure_reason": reason, "control_steps": steps,
                "runtime_seconds": runtime, "gpu_id": args.gpu, "worker_id": args.worker_id,
                "canonical_snapshot_sha256": expected[0], "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha, "lambda": LAMBDA, "beta": 0.0,
                "budget_schedule_arm": arm if arm in SCHEDULE_ARMS else None,
                "budget_scale": SCALE_ARMS.get(arm),
                "kmeans_K": None if arm in SCHEDULE_ARMS else KMEANS_K,
                "kmeans_seed": None if arm in SCHEDULE_ARMS else KMEANS_SEED,
                "code_version": version, "arrays_file": arrays_path.name,
                "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(item.get("actual_selected_count") for item in trace),
                "mean_feature_perturbation_relative": finite_mean(item.get("feature_perturbation_relative") for item in trace),
                "mean_centered_logit_residual_norm": finite_mean(item.get("centered_logit_residual_norm") for item in trace),
                "mean_guided_change_ratio": finite_mean(item.get("guided_change_ratio") for item in trace),
                "selector_trace": trace,
                **checks,
            }
            atomic_json(summary_path, summary)
            print(json.dumps({
                "task": args.task, "seed": seed, "arm": arm,
                "success": summary["success"], "runtime_seconds": round(runtime, 2),
            }), flush=True)


if __name__ == "__main__":
    main()
