"""Paired closed-loop rollout for frozen single/sparse Prompt-Attn layers."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import (
    KMEANS_K, KMEANS_SEED, LAMBDA, TASKS as BASE_TASKS, atomic_json, finite_mean,
    load_reference, parse_seeds, write_arrays,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_LAYER_SELECTION_V1"
DEFAULT_ARMS = ("prompt_single", "prompt_sparse")
AVAILABLE_ARMS = ("prompt_v1", "prompt_single", "prompt_sparse")
SUPPORTED_TASKS = tuple(TASK_INDEX)


def make_environment(task: str, gpu: int):
    """Create any canonical SIMPLER task with Vulkan pinned to this worker GPU."""
    import gymnasium as gym
    import simpler_env

    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task is not preregistered: {task}")
    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd",
            prepackaged_config=True,
            distractor_config="less",
            renderer_kwargs=renderer_kwargs,
        ), "GraspSingleOpenedCokeCanDistractorInScene-v0"
    environment_id, base_kwargs = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base_kwargs)
    kwargs.update({"prepackaged_config": True, "renderer_kwargs": renderer_kwargs})
    return gym.make(environment_id, obs_mode="rgbd", **kwargs), environment_id


def build_candidate_policies(base, task: str, lock: dict, arms: tuple[str, ...]) -> dict:
    result = {}
    for arm in arms:
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, LAMBDA)
        policy.beta = 0.0
        policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = tuple(lock[arm])
        result[arm] = policy
    return result


def audit(trace: list[dict], layers: list[int]) -> dict:
    checks = {
        "technical_nonempty": bool(trace),
        "all_feature_equal": bool(trace) and all(x.get("feature_equal") is True for x in trace),
        "all_guided_prefix": bool(trace) and all(x.get("guided_prefix") is True for x in trace),
        "all_reconstruction_finite": bool(trace) and all(x.get("reconstruction_finite") is True for x in trace),
        "all_lambda_locked": bool(trace) and all(abs(float(x.get("lambda", -1)) - .5) < 1e-12 for x in trace),
        "all_beta_zero": bool(trace) and all(abs(float(x.get("beta", -1))) < 1e-12 for x in trace),
        "all_coverage_exact": bool(trace) and all(x.get("coverage_exact") is True for x in trace),
        "all_non_target_bit_identical": bool(trace) and all(x.get("non_target_bit_identical") is True for x in trace),
        "all_layers_locked": bool(trace) and all(x.get("attention_layers") == layers for x in trace),
        "all_post_softmax": bool(trace) and all(x.get("attention_post_softmax") is True for x in trace),
        "all_visual_keys_locked": bool(trace) and all(x.get("visual_key_indices") == [1, 256] for x in trace),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"candidate rollout audit failed: {checks}")
    return checks


def ensure_config(
    artifact: Path,
    source: Path,
    closed_loop: Path,
    lock: dict,
    seeds_by_task: dict,
    arms: tuple[str, ...],
) -> None:
    payload = {
        "protocol_id": PROTOCOL,
        "purpose": "test whether sparse layer readout fixes Prompt-v1 layer averaging",
        "tasks": list(seeds_by_task), "seeds_by_task": seeds_by_task,
        "new_arms": list(arms),
        "reference_arms": {"standard_shr": str(closed_loop), "prompt_v1": str(closed_loop)},
        "layers": {arm: lock[arm] for arm in arms},
        "candidate_lock_sha256": hashlib.sha256((artifact.parent / "CANDIDATES_LOCK.json").read_bytes()).hexdigest(),
        "snapshot_artifact": str(source),
        "locked_downstream": {
            "query": "full instruction excluding special/template tokens",
            "head_aggregation": "all heads arithmetic mean",
            "coverage": "own-state Standard SHR m_t",
            "spatial_postprocessing": False,
            "harmonic": "16x16 four-neighbor Dirichlet beta=0/gamma=1",
            "prefix": "shared clean greedy prefix",
            "lambda": .5, "guided_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper": "clean positive dimension 6", "sampling": False,
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("closed-loop config lock differs")
    atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--layer-artifact", type=Path, required=True)
    parser.add_argument("--closed-loop-v1", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=SUPPORTED_TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument(
        "--config-tasks",
        default=",".join(BASE_TASKS),
        help="comma-separated locked task universe; defaults to the original three-task screen",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve(); layer_artifact = args.layer_artifact.resolve()
    closed_loop = args.closed_loop_v1.resolve(); source = args.snapshot_artifact.resolve()
    lock = json.loads((layer_artifact / "CANDIDATES_LOCK.json").read_text())
    selected_arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not selected_arms or any(arm not in AVAILABLE_ARMS for arm in selected_arms):
        raise ValueError(f"invalid --arms: {selected_arms}")
    candidate_lock = {arm: lock[arm] for arm in selected_arms}
    seed_lock = json.loads((layer_artifact / "CLOSED_LOOP_SEEDS.json").read_text())
    config_tasks = tuple(x.strip() for x in args.config_tasks.split(",") if x.strip())
    if not config_tasks or any(task not in SUPPORTED_TASKS for task in config_tasks):
        raise ValueError(f"invalid --config-tasks: {config_tasks}")
    if args.task not in config_tasks:
        raise ValueError(f"task {args.task} is outside --config-tasks")
    if "seed_range" in seed_lock:
        lo, hi = seed_lock["seed_range"]
        all_eval_seeds = {task: list(range(int(lo), int(hi) + 1)) for task in config_tasks}
    else:
        all_eval_seeds = {task: seed_lock["seeds_by_task"][task] for task in config_tasks}
    seeds = parse_seeds(args.seeds)
    if any(seed not in all_eval_seeds[args.task] for seed in seeds):
        raise ValueError("requested seed is outside locked unseen closed-loop set")
    artifact.mkdir(parents=True, exist_ok=True)
    ensure_config(artifact, source, closed_loop, lock | candidate_lock, all_eval_seeds, selected_arms)

    env, environment_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_candidate_policies(OpenVLAInference(**config), args.task, candidate_lock, selected_arms)

    for seed in seeds:
        with (source / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(source, args.task, seed)
        expected = (reference["canonical_snapshot_sha256"], reference["initial_state_sha256"], reference["initial_rgb_sha256"])
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
            if (expected[1], expected[2]) != (state_sha, rgb_sha):
                raise RuntimeError("restored snapshot mismatch")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError("instruction mismatch")
            policy.reset(instruction, seed=seed); policy._episode_trace=[]; policy._episode_logits=[]
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs)
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, list(policy.attention_layers))
            out.mkdir(parents=True, exist_ok=True)
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL, "task": args.task, "environment_id": environment_id,
                "seed": seed, "evaluation_seed": seed, "episode_id": seed, "arm": arm,
                "attention_layers": list(policy.attention_layers), "instruction": instruction,
                "success": bool(result["success"]), "result": jsonable(result), "failure_reason": reason,
                "control_steps": steps, "runtime_seconds": runtime, "gpu_id": args.gpu,
                "worker_id": args.worker_id, "canonical_snapshot_sha256": expected[0],
                "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                "lambda": LAMBDA, "beta": 0.0, "kmeans_K": KMEANS_K, "kmeans_seed": KMEANS_SEED,
                "arrays_file": arrays_path.name, "action_jitter_index": jerk,
                "mean_m_t": finite_mean(x.get("m_t") for x in trace),
                "mean_prompt_shr_overlap_ratio": finite_mean(x.get("prompt_shr_overlap_ratio") for x in trace),
                "mean_feature_perturbation_norm": finite_mean(x.get("feature_perturbation_norm") for x in trace),
                "mean_centered_logit_residual_norm": finite_mean(x.get("centered_logit_residual_norm") for x in trace),
                "mean_guided_change_ratio": finite_mean(x.get("guided_change_ratio") for x in trace),
                "selector_trace": trace, **checks,
            }
            atomic_json(summary_path, summary)
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm,
                              "success": summary["success"], "runtime_seconds": round(runtime, 2)}), flush=True)


if __name__ == "__main__":
    main()
