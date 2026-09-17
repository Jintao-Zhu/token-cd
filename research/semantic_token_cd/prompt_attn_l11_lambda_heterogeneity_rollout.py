"""Discovery rollout for fixed L11 guidance strengths lambda=0 and 0.25."""
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
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json, finite_mean, load_reference, write_arrays
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_L11_LAMBDA_HETEROGENEITY_DISCOVERY_V1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ARM_LAMBDAS = {"l11_positive_only": 0.0, "l11_fixed_025": 0.25}


def parse_seeds(specification: str) -> list[int]:
    seeds = []
    for part in specification.split(","):
        part = part.strip()
        if "-" in part:
            low, high = map(int, part.split("-", 1)); seeds.extend(range(low, high + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or min(result) < 0 or max(result) > 99:
        raise ValueError("discovery seeds must be within 0..99")
    return result


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_config(artifact: Path, canonical: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    payload = {
        "protocol_id": PROTOCOL,
        "created_date": "2026-09-13",
        "purpose": "measure episode-level heterogeneity across fixed L11 guidance strengths",
        "tasks": list(TASKS),
        "seeds": [0, 99],
        "arms": ARM_LAMBDAS,
        "new_episodes": 500,
        "new_arm_plan": {
            "l11_positive_only": {task: 100 for task in TASKS},
            "l11_fixed_025": {"google_robot_open_drawer": 100},
        },
        "reused": {
            "lambda_0_5": "prompt_attn_layer_selection_v1 paired L11 Prompt-Single",
            "lambda_0_25": "adaptive-lambda artifact for close/pick/move seeds 0-99",
        },
        "canonical_snapshot_artifact": str(canonical),
        "locked_selector": {
            "mode": "prompt_attention", "layers": [11],
            "coverage": "own-state Standard-SHR matched m_t", "beta": 0.0,
        },
        "positive_only_definition": "compute the unchanged L11 negative branch but execute z*=z+ for all seven action dimensions",
        "code_version": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "policy_sha256": file_sha(repo / "research/semantic_token_cd/prompt_attn_shr_policy.py"),
            "rollout_sha256": file_sha(Path(__file__).resolve()),
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("lambda heterogeneity config lock differs")
    atomic_json(path, payload)
    return payload["code_version"]


def build_policies(base, task: str, arms: tuple[str, ...]) -> dict:
    result = {}
    for arm in arms:
        policy = copy.copy(base); policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, ARM_LAMBDAS[arm])
        policy.beta = 0.0; policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]; policy.attention_layers = (11,)
        policy.selection_count = None; policy.save_prompt_attention = False
        result[arm] = policy
    return result


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
        "all_seven_dims": bool(trace) and all(len(row.get("positive_token_ids", [])) == 7 and len(row.get("final_token_ids", [])) == 7 for row in trace),
    }
    if expected == 0.0:
        checks.update({
            "all_final_tokens_equal_positive": all(row["final_token_ids"] == row["positive_token_ids"] for row in trace),
            "all_continuous_actions_equal_positive": all(np.array_equal(np.asarray(row["guided_action"]), np.asarray(row["clean_action"])) for row in trace),
            "all_guided_change_zero": all(row.get("guided_changed_dims") == 0 and row.get("guided_clean_action_l2") == 0.0 for row in trace),
            "all_gripper_equal_positive": all(row["final_token_ids"][6] == row["positive_token_ids"][6] for row in trace),
        })
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"lambda heterogeneity audit failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--arms", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve(); canonical = args.canonical.resolve()
    code = ensure_config(artifact, canonical)
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not arms or any(arm not in ARM_LAMBDAS for arm in arms): raise ValueError(arms)
    if args.task != "google_robot_open_drawer" and "l11_fixed_025" in arms:
        raise ValueError("new lambda=.25 rollout is locked to open_drawer")
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task, arms)
    for seed in parse_seeds(args.seeds):
        with (canonical / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle: snapshot = pickle.load(handle)
        reference = load_reference(canonical, args.task, seed)
        expected = (reference["canonical_snapshot_sha256"], reference["initial_state_sha256"], reference["initial_rgb_sha256"])
        if snapshot_sha(snapshot) != expected[0]: raise RuntimeError("canonical snapshot mismatch")
        for arm, policy in policies.items():
            out = artifact / "episodes" / args.task / arm
            summary_path = out / f"episode_{seed:03d}_summary.json"; arrays_path = out / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True); continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if (state_sha, rgb_sha) != expected[1:]: raise RuntimeError("restored snapshot mismatch")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]: raise RuntimeError("instruction mismatch")
            policy.reset(instruction, seed=seed); policy._episode_trace = []; policy._episode_logits = []
            started = time.monotonic(); result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs); runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace); checks = audit(trace, arm)
            out.mkdir(parents=True, exist_ok=True); write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL, "task": args.task, "environment_id": environment_id,
                "seed": seed, "arm": arm, "lambda": ARM_LAMBDAS[arm], "attention_layers": [11],
                "instruction": instruction, "success": bool(result["success"]), "result": jsonable(result),
                "failure_reason": reason, "control_steps": steps, "runtime_seconds": runtime,
                "gpu_id": args.gpu, "worker_id": args.worker_id,
                "canonical_snapshot_sha256": expected[0], "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                "beta": 0.0, "code_version": code, "arrays_file": arrays_path.name,
                "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(row.get("actual_selected_count") for row in trace),
                "selector_trace": trace, **checks,
            }
            atomic_json(summary_path, summary)
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": summary["success"], "runtime_seconds": round(runtime, 2)}), flush=True)


if __name__ == "__main__": main()
