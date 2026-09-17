"""Closed-loop L11 token-count sweep on canonical SIMPLER snapshots."""
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
    KMEANS_K, KMEANS_SEED, LAMBDA, atomic_json, finite_mean, load_reference,
    write_arrays,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_L11_TOKEN_COUNT_V1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ARM_COUNTS = {
    "l11_matched": None,
    "l11_k16": 16,
    "l11_k24": 24,
    "l11_k32": 32,
    "l11_k48": 48,
    "l11_k64": 64,
}


def parse_count_sweep_seeds(specification: str) -> list[int]:
    seeds = []
    for part in specification.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(seed < 100 or seed > 199 for seed in result):
        raise ValueError("L11 token-count sweep seeds must be within 100..199")
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
        policy.selection_count = ARM_COUNTS[arm]
        policy.save_prompt_attention = False
        policies[arm] = policy
    return policies


def audit(trace: list[dict], expected_count: int | None) -> dict:
    fixed = expected_count is not None
    checks = {
        "technical_nonempty": bool(trace),
        "all_feature_equal": bool(trace) and all(x.get("feature_equal") is True for x in trace),
        "all_guided_prefix": bool(trace) and all(x.get("guided_prefix") is True for x in trace),
        "all_reconstruction_finite": bool(trace) and all(x.get("reconstruction_finite") is True for x in trace),
        "all_lambda_locked": bool(trace) and all(abs(float(x.get("lambda", -1)) - .5) < 1e-12 for x in trace),
        "all_beta_zero": bool(trace) and all(abs(float(x.get("beta", -1))) < 1e-12 for x in trace),
        "all_coverage_exact": bool(trace) and all(x.get("coverage_exact") is True for x in trace),
        "all_non_target_bit_identical": bool(trace) and all(x.get("non_target_bit_identical") is True for x in trace),
        "all_l11_locked": bool(trace) and all(x.get("attention_layers") == [11] for x in trace),
        "all_post_softmax": bool(trace) and all(x.get("attention_post_softmax") is True for x in trace),
        "all_visual_keys_locked": bool(trace) and all(x.get("visual_key_indices") == [1, 256] for x in trace),
        "all_seven_action_dims": bool(trace) and all(
            len(x.get("positive_token_ids", [])) == 7 and len(x.get("final_token_ids", [])) == 7
            for x in trace
        ),
        "all_gripper_clean": bool(trace) and all(
            x["positive_token_ids"][6] == x["final_token_ids"][6] for x in trace
        ),
        "fixed_count_exact_or_matched": bool(trace) and all(
            x.get("actual_selected_count") == (expected_count if fixed else x.get("m_t"))
            for x in trace
        ),
        "fixed_arms_bypass_kmeans": bool(trace) and all(
            (not fixed) or (
                x.get("coverage_mode") == f"fixed_top_{expected_count}"
                and not x.get("reference_shr_token_ids")
            ) for x in trace
        ),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"L11 count rollout audit failed: {checks}")
    return checks


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_config(artifact: Path, source: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    policy_file = repo / "research/semantic_token_cd/prompt_attn_shr_policy.py"
    rollout_file = Path(__file__).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    payload = {
        "protocol_id": PROTOCOL,
        "purpose": "isolate L11 selected-token count while locking all downstream SHR operations",
        "tasks": list(TASKS),
        "seeds_by_task": {task: list(range(100, 200)) for task in TASKS},
        "arms": ARM_COUNTS,
        "episode_count": 2400,
        "snapshot_artifact": str(source),
        "code_version": {
            "git_commit": commit,
            "policy_sha256": file_sha(policy_file),
            "rollout_sha256": file_sha(rollout_file),
        },
        "locked_downstream": {
            "attention_layers": [11],
            "query": "full instruction excluding special/template/padding tokens",
            "head_query_aggregation": "equal arithmetic mean",
            "spatial_postprocessing": False,
            "harmonic": "16x16 four-neighbor Dirichlet beta=0/gamma=1",
            "prefix": "shared clean greedy prefix",
            "lambda": .5,
            "guided_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper": "clean positive dimension 6",
            "sampling": False,
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("L11 token-count config lock differs")
    artifact.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
    return payload["code_version"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--arms", default=",".join(ARM_COUNTS))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    code_version = ensure_config(artifact, source)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not arms or any(arm not in ARM_COUNTS for arm in arms):
        raise ValueError(f"invalid arms: {arms}")
    seeds = parse_count_sweep_seeds(args.seeds)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task, arms)

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
            out = artifact / "episodes" / args.task / arm
            summary_path = out / f"episode_{seed:03d}_summary.json"
            arrays_path = out / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if (state_sha, rgb_sha) != expected[1:]:
                raise RuntimeError("restored snapshot mismatch")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError("instruction mismatch")
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs)
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, ARM_COUNTS[arm])
            out.mkdir(parents=True, exist_ok=True)
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task, "environment_id": environment_id,
                "seed": seed, "evaluation_seed": seed, "episode_id": seed, "arm": arm,
                "selected_token_count_config": ARM_COUNTS[arm],
                "attention_layers": [11], "instruction": instruction,
                "success": bool(result["success"]), "result": jsonable(result),
                "failure_reason": reason, "control_steps": steps,
                "runtime_seconds": runtime, "gpu_id": args.gpu, "worker_id": args.worker_id,
                "canonical_snapshot_sha256": expected[0], "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha, "lambda": LAMBDA, "beta": 0.0,
                "kmeans_K": KMEANS_K if ARM_COUNTS[arm] is None else None,
                "kmeans_seed": KMEANS_SEED if ARM_COUNTS[arm] is None else None,
                "code_version": code_version, "arrays_file": arrays_path.name,
                "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(x.get("actual_selected_count") for x in trace),
                "mean_mask_component_count": finite_mean(x.get("mask_component_count") for x in trace),
                "mean_isolated_token_ratio": finite_mean(x.get("isolated_token_ratio") for x in trace),
                "mean_feature_perturbation_norm": finite_mean(x.get("feature_perturbation_norm") for x in trace),
                "mean_centered_logit_residual_norm": finite_mean(x.get("centered_logit_residual_norm") for x in trace),
                "mean_guided_change_ratio": finite_mean(x.get("guided_change_ratio") for x in trace),
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
