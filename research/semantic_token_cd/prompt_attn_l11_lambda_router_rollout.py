"""True episode-start closed-loop rollout for the frozen L11 lambda router."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.l11_lambda_router_protocol import (
    ARM_NAMES,
    ROUTER_PROTOCOL,
    TASKS,
    arm_arrays_path,
    arm_summary_path,
    atomic_json,
    choose_lambda,
    feature_metadata_path,
    feature_path,
    load_frozen_model,
    parse_seeds,
    sha256,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.prompt_attn_l11_lambda_confirmation_rollout import audit
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import finite_mean, load_reference, write_arrays
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


LAMBDAS = (0.0, 0.25, 0.5)
HALF_ROOT = Path("artifacts/prompt_attn_l11_matched_full_9task_0_299_v1/run/episodes")


def build_policies(base, task: str) -> dict[float, PromptAttentionSHRInference]:
    result = {}
    for value in LAMBDAS:
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, value)
        policy.beta = 0.0
        policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,)
        policy.selection_count = None
        policy.save_prompt_attention = False
        result[value] = policy
    return result


def selected_reference(confirmation: Path, task: str, seed: int, value: float) -> tuple[Path, Path]:
    if value == 0.5:
        root = HALF_ROOT / task / "prompt_single"
        return root / f"episode_{seed:03d}_summary.json", root / f"episode_{seed:03d}_arrays.npz"
    arm = ARM_NAMES[value]
    return arm_summary_path(confirmation, task, arm, seed), arm_arrays_path(confirmation, task, arm, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()

    confirmation = args.confirmation.resolve()
    confirm_results = json.loads((confirmation / "analysis" / "CONFIRMATION_RESULTS.json").read_text())
    if not confirm_results["go_no_go"]["passed"]:
        raise RuntimeError("held-out confirmation did not pass")
    model, model_path = load_frozen_model(args.discovery.resolve())
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
    policies = build_policies(OpenVLAInference(**config), args.task)
    for seed in parse_seeds(args.seeds):
        output = artifact / "episodes" / args.task / "router"
        summary_path = output / f"episode_{seed:03d}_summary.json"
        arrays_path = output / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            print(json.dumps({"skip": True, "task": args.task, "seed": seed}), flush=True)
            continue
        metadata = json.loads(feature_metadata_path(confirmation, args.task, seed).read_text())
        chosen_lambda, scores = choose_lambda(model, feature_path(confirmation, args.task, seed), args.task)
        reference_summary_path, reference_arrays_path = selected_reference(confirmation, args.task, seed, chosen_lambda)
        selected_summary = json.loads(reference_summary_path.read_text())
        with (canonical / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(canonical, args.task, seed)
        if snapshot_sha(snapshot) != reference["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        hashes = (reference["canonical_snapshot_sha256"], state_sha, rgb_sha)
        if hashes != (
            metadata["canonical_snapshot_sha256"], metadata["initial_state_sha256"], metadata["initial_rgb_sha256"]
        ):
            raise RuntimeError("router feature/state pairing mismatch")
        instruction = env.unwrapped.get_language_instruction()
        policy = policies[chosen_lambda]
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        started = time.monotonic()
        result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs)
        trace = jsonable(policy._episode_trace)
        checks = audit(trace, chosen_lambda)
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        write_arrays(arrays_path, policy._episode_logits, actions)
        with np.load(reference_arrays_path) as expected_arrays:
            replay_actions_equal = np.array_equal(actions, expected_arrays["executed_actions"])
        replay_success_equal = bool(result["success"]) == bool(selected_summary["success"])
        if not replay_actions_equal or not replay_success_equal:
            raise RuntimeError(f"routed replay differs from selected fixed arm: {args.task} seed={seed}")
        summary = {
            "protocol_id": ROUTER_PROTOCOL,
            "task": args.task,
            "environment_id": environment_id,
            "seed": seed,
            "arm": "episode_start_lambda_router",
            "chosen_lambda": chosen_lambda,
            "q_scores": scores,
            "success": bool(result["success"]),
            "result": jsonable(result),
            "failure_reason": reason,
            "control_steps": steps,
            "runtime_seconds": time.monotonic() - started,
            "gpu_id": args.gpu,
            "worker_id": args.worker_id,
            "canonical_snapshot_sha256": hashes[0],
            "initial_state_sha256": hashes[1],
            "initial_rgb_sha256": hashes[2],
            "frozen_model_path": str(model_path),
            "frozen_model_sha256": sha256(model_path),
            "feature_file": str(feature_path(confirmation, args.task, seed)),
            "selected_fixed_summary": str(reference_summary_path),
            "selected_fixed_arrays": str(reference_arrays_path),
            "replay_actions_equal": replay_actions_equal,
            "replay_success_equal": replay_success_equal,
            "action_jitter_index": jerk,
            "mean_selected_token_count": finite_mean(row.get("actual_selected_count") for row in trace),
            "selector_trace": trace,
            **checks,
        }
        atomic_json(summary_path, summary)
        print(json.dumps({"task": args.task, "seed": seed, "lambda": chosen_lambda, "success": summary["success"]}), flush=True)


if __name__ == "__main__":
    main()
