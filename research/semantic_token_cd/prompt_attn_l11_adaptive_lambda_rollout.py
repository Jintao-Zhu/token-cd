"""Canonical four-task rollout for L11-Matched Adaptive-lambda controllers."""
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
from research.semantic_token_cd.prompt_attn_l11_adaptive_lambda_policy import (
    CONTROLLER_MODES,
    CORRECTION_EPS,
    EMA_RHO,
    FIXED_LOW_LAMBDA,
    MIN_LAMBDA,
    PROBE_LAMBDA,
    AdaptiveLambdaPromptAttentionSHRInference,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_rollout import (
    KMEANS_K,
    KMEANS_SEED,
    atomic_json,
    finite_mean,
    load_reference,
    write_arrays,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_L11_ADAPTIVE_LAMBDA_4TASK_V1"
TASKS = (
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "widowx_carrot_on_plate",
)
ARMS = {
    "l11_fixed_025": "fixed_025",
    "l11_adaptive_raw": "adaptive_raw",
    "l11_adaptive_ema": "adaptive_ema",
}


def parse_seeds(specification: str) -> list[int]:
    seeds: list[int] = []
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(seed < 0 or seed > 299 for seed in result):
        raise ValueError("Adaptive-lambda seeds must be within 0..299")
    return result


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_config(artifact: Path, source: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    policy_file = repo / "research/semantic_token_cd/prompt_attn_l11_adaptive_lambda_policy.py"
    rollout_file = Path(__file__).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    payload = {
        "protocol_id": PROTOCOL,
        "created_date": "2026-09-13",
        "purpose": "test whether temporal consistency can preserve L11 rescues while reducing harms",
        "tasks": list(TASKS),
        "development_seeds": [0, 99],
        "confirmation_seeds": [100, 299],
        "development_arms": list(ARMS),
        "confirmation_arms": ["l11_fixed_025", "development-selected adaptive arm"],
        "snapshot_artifact": str(source),
        "code_version": {
            "git_commit": commit,
            "policy_sha256": file_sha(policy_file),
            "rollout_sha256": file_sha(rollout_file),
        },
        "selector": {
            "mode": "prompt_attention",
            "attention_layers": [11],
            "coverage": "own-state Standard-SHR matched m_t",
            "harmonic": "16x16 four-neighbor Dirichlet beta=0",
        },
        "controller": {
            "probe_lambda": PROBE_LAMBDA,
            "lambda_min": MIN_LAMBDA,
            "fixed_low_lambda": FIXED_LOW_LAMBDA,
            "raw_mapping": "0.1 + 0.4 * clip(cosine, 0, 1)",
            "ema_rho": EMA_RHO,
            "correction": "six-dimensional soft expected normalized action; gripper excluded",
            "correction_epsilon": CORRECTION_EPS,
            "small_norm_rule": "hold previous lambda",
            "first_step_lambda": PROBE_LAMBDA,
            "forward_overhead": 0,
        },
    }
    artifact.mkdir(parents=True, exist_ok=True)
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError(f"Adaptive-lambda config lock differs: {path}")
    atomic_json(path, payload)
    return payload["code_version"]


def build_policies(base, task: str, arms: tuple[str, ...]) -> dict:
    policies = {}
    for arm in arms:
        policy = copy.copy(base)
        policy.__class__ = AdaptiveLambdaPromptAttentionSHRInference
        _init_common(policy, PROBE_LAMBDA)
        policy.beta = 0.0
        policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,)
        policy.selection_count = None
        policy.save_prompt_attention = False
        policy.controller_mode = ARMS[arm]
        policies[arm] = policy
    return policies


def audit(trace: list[dict], arm: str) -> dict:
    expected_mode = ARMS[arm]
    checks = {
        "technical_nonempty": bool(trace),
        "all_feature_equal": bool(trace) and all(x.get("feature_equal") is True for x in trace),
        "all_guided_prefix": bool(trace) and all(x.get("guided_prefix") is True for x in trace),
        "all_reconstruction_finite": bool(trace) and all(x.get("reconstruction_finite") is True for x in trace),
        "all_probe_lambda_locked": bool(trace) and all(abs(float(x.get("probe_lambda", -1)) - PROBE_LAMBDA) < 1e-12 for x in trace),
        "all_beta_zero": bool(trace) and all(abs(float(x.get("beta", -1))) < 1e-12 for x in trace),
        "all_coverage_exact": bool(trace) and all(x.get("coverage_exact") is True for x in trace),
        "all_non_target_bit_identical": bool(trace) and all(x.get("non_target_bit_identical") is True for x in trace),
        "all_l11_locked": bool(trace) and all(x.get("attention_layers") == [11] for x in trace),
        "all_post_softmax": bool(trace) and all(x.get("attention_post_softmax") is True for x in trace),
        "all_visual_keys_locked": bool(trace) and all(x.get("visual_key_indices") == [1, 256] for x in trace),
        "all_matched_count": bool(trace) and all(x.get("actual_selected_count") == x.get("m_t") for x in trace),
        "all_controller_mode_locked": bool(trace) and all(x.get("controller_mode") == expected_mode for x in trace),
        "all_lambda_bounded": bool(trace) and all(MIN_LAMBDA <= float(x.get("final_lambda", -1)) <= PROBE_LAMBDA for x in trace),
        "all_correction_six_dimensional": bool(trace) and all(len(x.get("probe_correction", [])) == 6 for x in trace),
        "all_gripper_unchanged": bool(trace) and all(x.get("gripper_token_unchanged") is True for x in trace),
    }
    if expected_mode == "fixed_025":
        checks["all_fixed_low_exact"] = all(
            abs(float(x.get("final_lambda", -1)) - FIXED_LOW_LAMBDA) < 1e-12 for x in trace
        )
    else:
        checks["first_step_probe_lambda"] = abs(float(trace[0].get("final_lambda", -1)) - PROBE_LAMBDA) < 1e-12
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"Adaptive-lambda audit failed for {arm}: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--arms", required=True)
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
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"invalid Adaptive-lambda arms: {arms}")
    seeds = parse_seeds(args.seeds)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task, arms)

    for seed in seeds:
        with (source / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(source, args.task, seed)
        expected = (
            reference["canonical_snapshot_sha256"],
            reference["initial_state_sha256"],
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
            checks = audit(trace, arm)
            out.mkdir(parents=True, exist_ok=True)
            write_arrays(arrays_path, policy._episode_logits, actions)
            lambdas = [float(x["final_lambda"]) for x in trace]
            similarities = [
                float(x["correction_cosine"])
                for x in trace if x.get("correction_cosine") is not None
            ]
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "evaluation_seed": seed,
                "episode_id": seed,
                "arm": arm,
                "controller_mode": ARMS[arm],
                "attention_layers": [11],
                "instruction": instruction,
                "success": bool(result["success"]),
                "result": jsonable(result),
                "failure_reason": reason,
                "control_steps": steps,
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": expected[0],
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "probe_lambda": PROBE_LAMBDA,
                "lambda_min": MIN_LAMBDA,
                "beta": 0.0,
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "code_version": code_version,
                "arrays_file": arrays_path.name,
                "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(x.get("actual_selected_count") for x in trace),
                "mean_feature_perturbation_norm": finite_mean(x.get("feature_perturbation_norm") for x in trace),
                "mean_centered_logit_residual_norm": finite_mean(x.get("centered_logit_residual_norm") for x in trace),
                "mean_guided_change_ratio": finite_mean(x.get("guided_change_ratio") for x in trace),
                "mean_final_lambda": float(np.mean(lambdas)),
                "min_final_lambda": float(np.min(lambdas)),
                "max_final_lambda": float(np.max(lambdas)),
                "reduced_lambda_steps": int(sum(value < PROBE_LAMBDA - 1e-12 for value in lambdas)),
                "mean_correction_cosine": float(np.mean(similarities)) if similarities else None,
                "selector_trace": trace,
                **checks,
            }
            atomic_json(summary_path, summary)
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": summary["success"],
                "mean_lambda": round(summary["mean_final_lambda"], 4),
                "runtime_seconds": round(runtime, 2),
            }), flush=True)


if __name__ == "__main__":
    main()
