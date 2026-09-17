"""Closed-loop rollout for the pure L11-Matched dense lambda sweep."""
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

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.l11_dense_lambda_protocol import (
    ARMS,
    ARM_LAMBDAS,
    ARTIFACT,
    CANONICAL,
    PROTOCOL,
    TASKS,
    atomic_json,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import finite_mean, load_reference, write_arrays
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


def parse_seeds(specification: str) -> list[int]:
    seeds: list[int] = []
    for part in specification.split(","):
        part = part.strip()
        if "-" in part:
            low, high = map(int, part.split("-", 1))
            seeds.extend(range(low, high + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or min(result) < 0 or max(result) > 99:
        raise ValueError("dense lambda sweep seeds must be within 0..99")
    return result


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_config(artifact: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    payload = {
        "protocol_id": PROTOCOL,
        "created_date": "2026-09-14",
        "purpose": "complete the pure L11-Matched fixed-lambda response curve near the global optimum",
        "tasks": list(TASKS),
        "seeds": [0, 99],
        "lambdas": list(ARM_LAMBDAS.values()),
        "arms": ARM_LAMBDAS,
        "episodes": len(TASKS) * 100 * len(ARMS),
        "canonical_snapshot_artifact": str(CANONICAL),
        "locked_selector": {
            "mode": "prompt_attention",
            "layers": [11],
            "coverage": "own-state Standard-SHR matched m_t",
            "beta": 0.0,
        },
        "comparison_set": "same tasks and canonical seeds 0-99 as the existing lambda={0,.25,.5,.6,.75} sweep",
        "code_version": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "policy_sha256": file_sha(repo / "research/semantic_token_cd/prompt_attn_shr_policy.py"),
            "rollout_sha256": file_sha(Path(__file__).resolve()),
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("dense lambda config lock differs")
    atomic_json(path, payload)
    return payload["code_version"]


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
        policy.selection_top_p = None
        policy.save_prompt_attention = False
        policies[arm] = policy
    return policies


def audit(trace: list[dict], arm: str) -> dict:
    expected = ARM_LAMBDAS[arm]
    checks = {
        "technical_nonempty": bool(trace),
        "all_lambda_exact": bool(trace) and all(abs(float(row.get("lambda", -1)) - expected) < 1e-12 for row in trace),
        "all_l11": bool(trace) and all(row.get("attention_layers") == [11] for row in trace),
        "all_matched_coverage": bool(trace) and all(row.get("actual_selected_count") == row.get("m_t") for row in trace),
        "all_feature_equal": bool(trace) and all(row.get("feature_equal") is True for row in trace),
        "all_non_target_equal": bool(trace) and all(row.get("non_target_bit_identical") is True for row in trace),
        "all_reconstruction_finite": bool(trace) and all(row.get("reconstruction_finite") is True for row in trace),
        "all_seven_dims": bool(trace) and all(
            len(row.get("positive_token_ids", [])) == 7 and len(row.get("final_token_ids", [])) == 7
            for row in trace
        ),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"dense lambda audit failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    code = ensure_config(artifact)
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not arms or any(arm not in ARM_LAMBDAS for arm in arms):
        raise ValueError(f"invalid arms: {arms}")
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task, arms)
    for seed in parse_seeds(args.seeds):
        with (CANONICAL / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(CANONICAL, args.task, seed)
        expected = (
            reference["canonical_snapshot_sha256"],
            reference["initial_state_sha256"],
            reference["initial_rgb_sha256"],
        )
        if snapshot_sha(snapshot) != expected[0]:
            raise RuntimeError("canonical snapshot mismatch")
        hashes = []
        for arm, policy in policies.items():
            out = artifact / "episodes" / args.task / arm
            summary_path = out / f"episode_{seed:03d}_summary.json"
            arrays_path = out / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            observation, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if (state_sha, rgb_sha) != expected[1:]:
                raise RuntimeError("restored snapshot mismatch")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError("instruction mismatch")
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(env, policy, instruction, observation)
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, arm)
            out.mkdir(parents=True, exist_ok=True)
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
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": expected[0],
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "beta": 0.0,
                "code_version": code,
                "arrays_file": arrays_path.name,
                "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(row.get("actual_selected_count") for row in trace),
                "selector_trace": trace,
                **checks,
            }
            atomic_json(summary_path, summary)
            hashes.append(expected)
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "lambda": ARM_LAMBDAS[arm],
                "success": summary["success"],
                "runtime_seconds": round(runtime, 2),
            }), flush=True)
        if len(set(hashes)) not in (0, 1):
            raise RuntimeError(f"within-seed canonical mismatch: {args.task} seed={seed}")
    env.close()


if __name__ == "__main__":
    main()

