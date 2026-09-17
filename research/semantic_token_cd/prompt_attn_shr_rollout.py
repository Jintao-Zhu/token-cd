"""Canonical three-arm rollout for Prompt-Attn-SHR v1."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_shr_policy import (
    LAYER_END,
    LAYER_START,
    PromptAttentionSHRInference,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_SHR_V1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ARMS = ("standard_shr", "prompt_attn_shr", "random_shr")
MODES = {
    "standard_shr": "standard_shr",
    "prompt_attn_shr": "prompt_attention",
    "random_shr": "random_matched",
}
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0


def make_environment(task: str, gpu: int):
    """Pin SAPIEN's Vulkan renderer to the same physical GPU as the worker."""
    import gymnasium as gym
    import simpler_env

    # CUDA_VISIBLE_DEVICES exposes the assigned physical GPU as logical cuda:0.
    # Passing it explicitly prevents all Vulkan renderers from selecting the
    # same default physical device during concurrent environment creation.
    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd",
            prepackaged_config=True,
            distractor_config="less",
            renderer_kwargs=renderer_kwargs,
        ), "GraspSingleOpenedCokeCanDistractorInScene-v0"
    if task in TASKS:
        environment_id, kwargs = simpler_env.ENVIRONMENT_MAP[task]
        kwargs = dict(kwargs)
        kwargs.update({"prepackaged_config": True, "renderer_kwargs": renderer_kwargs})
        return gym.make(environment_id, obs_mode="rgbd", **kwargs), environment_id
    raise ValueError(f"task is not preregistered: {task}")


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(seed < 0 or seed > 99 for seed in result):
        raise ValueError("Prompt-Attn-SHR v1 seeds must be within 0..99")
    return result


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def ensure_config(artifact: Path, snapshot_artifact: Path) -> None:
    config = {
        "protocol_id": PROTOCOL,
        "purpose": "selector-only screen: contextual prompt attention vs KMeans entity cosine vs matched random",
        "tasks": list(TASKS),
        "seeds": list(range(100)),
        "arms": list(ARMS),
        "snapshot_artifact": str(snapshot_artifact),
        "snapshot_policy": "read-only canonical snapshot reuse; hash mismatch is fatal",
        "coverage": "each arm computes standard SHR m_t on its own current observation; alternative selectors choose exactly m_t unique tokens",
        "prompt_query": "all non-special tokens from the exact instruction input_ids; audited after 256-token multimodal insertion",
        "visual_keys": "multimodal positions 1..256",
        "attention": {
            "layers_zero_based": list(range(LAYER_START, LAYER_END)),
            "weights": "post-softmax",
            "aggregation": "equal mean over layers, heads, and instruction queries",
            "visual_renormalization": False,
            "spatial_postprocessing": False,
        },
        "random": "without replacement; SeedSequence([task_index, episode_seed, control_step, 0x50A77E11])",
        "shared_downstream": {
            "visual_tokens": 256,
            "harmonic": "16x16 four-neighbor Dirichlet; beta=0/gamma=1",
            "lambda": LAMBDA,
            "guided_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper": "clean positive dimension 6",
            "prefix": "clean greedy teacher-forced prefix",
            "sampling": False,
        },
        "gpu_policy": "use only GPUs found idle at launch; conservative workers per GPU after smoke test",
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"CONFIG_LOCK differs from Prompt-Attn-SHR v1: {path}")
    atomic_json(path, config)


def build_policies(base, task: str) -> dict[str, PromptAttentionSHRInference]:
    policies = {}
    for arm in ARMS:
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, LAMBDA)
        policy.beta = 0.0
        policy.selector_mode = MODES[arm]
        policy.task_index = TASK_INDEX[task]
        policies[arm] = policy
    return policies


def load_reference(root: Path, task: str, seed: int) -> dict:
    path = root / "episodes" / task / "shr_harmonic" / f"episode_{seed:03d}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing canonical SHR reference: {path}")
    return json.loads(path.read_text())


def audit_trace(trace: list[dict], arm: str) -> dict:
    checks = {
        "all_feature_equal": bool(trace) and all(step.get("feature_equal") is True for step in trace),
        "all_guided_prefix": bool(trace) and all(step.get("guided_prefix") is True for step in trace),
        "all_reconstruction_finite": bool(trace) and all(step.get("reconstruction_finite") is True for step in trace),
        "all_lambda_locked": bool(trace) and all(abs(float(step.get("lambda", -1)) - LAMBDA) < 1e-12 for step in trace),
        "all_beta_zero": bool(trace) and all(abs(float(step.get("beta", -1))) < 1e-12 for step in trace),
        "all_coverage_exact": bool(trace) and all(
            len(step.get("selected_token_ids", [])) == int(step.get("num_tokens", -1)) for step in trace
        ),
    }
    if arm != "standard_shr":
        checks.update({
            "all_m_exact": all(step.get("coverage_exact") is True for step in trace),
            "all_non_target_bit_identical": all(step.get("non_target_bit_identical") is True for step in trace),
            "all_centered_residual_six_dims": all(
                len(step.get("centered_logit_residual_norm_per_dim", [])) == 6 for step in trace
            ),
        })
    if arm == "prompt_attn_shr":
        checks.update({
            "all_attention_post_softmax": all(step.get("attention_post_softmax") is True for step in trace),
            "all_attention_layers_locked": all(
                step.get("attention_layers") == list(range(LAYER_START, LAYER_END)) for step in trace
            ),
            "all_query_indices_nonempty": all(step.get("prompt_multimodal_query_indices") for step in trace),
            "all_visual_keys_locked": all(step.get("visual_key_indices") == [1, 256] for step in trace),
        })
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"Prompt-Attn-SHR audit failed for {arm}: {checks}")
    return checks


def write_arrays(path: Path, records: list[dict], actions: np.ndarray) -> None:
    payload = {"executed_actions": actions}
    for key in ("positive", "negative", "selected_mask", "reference_shr_mask", "prompt_attention"):
        if records and all(key in record for record in records):
            payload[key] = np.stack([record[key] for record in records])
    np.savez_compressed(path, **payload)


def finite_mean(values) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", default="manual")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    ensure_config(artifact, source)
    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = source / "snapshots" / args.task
    env, environment_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**policy_config), args.task)

    for seed in parse_seeds(args.seeds):
        snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"missing canonical snapshot: {snapshot_path}")
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        reference = load_reference(source, args.task, seed)
        reference_hashes = (
            reference["canonical_snapshot_sha256"],
            reference["initial_state_sha256"],
            reference["initial_rgb_sha256"],
        )
        if canonical != reference_hashes[0]:
            raise RuntimeError(f"canonical snapshot mismatch: {args.task} seed={seed}")

        hashes = {}
        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                existing = json.loads(summary_path.read_text())
                hashes[arm] = tuple(existing[key] for key in (
                    "canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"
                ))
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True)
                continue

            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if (canonical, state_sha, rgb_sha) != reference_hashes:
                raise RuntimeError(f"restored snapshot mismatch: {args.task} seed={seed}")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference.get("instruction"):
                raise RuntimeError(f"instruction mismatch: {args.task} seed={seed}")
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, action_jerk = run_episode(env, policy, instruction, obs)
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            audit = audit_trace(trace, arm)
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "evaluation_seed": seed,
                "episode_id": seed,
                "arm": arm,
                "selector_mode": MODES[arm],
                "instruction": instruction,
                "success": bool(result["success"]),
                "result": jsonable(result),
                "failure_reason": reason,
                "control_steps": steps,
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "reference_arm": "shr_harmonic",
                "reference_success": bool(reference["success"]),
                "lambda": LAMBDA,
                "beta": 0.0,
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "arrays_file": arrays_path.name,
                "action_jitter_index": action_jerk,
                "mean_m_t": finite_mean(step.get("m_t", step.get("num_tokens")) for step in trace),
                "mean_prompt_shr_overlap_ratio": finite_mean(
                    step.get("prompt_shr_overlap_ratio") for step in trace
                ),
                "mean_feature_perturbation_norm": finite_mean(
                    step.get("feature_perturbation_norm") for step in trace
                ),
                "mean_centered_logit_residual_norm": finite_mean(
                    step.get("centered_logit_residual_norm") for step in trace
                ),
                "mean_guided_change_ratio": finite_mean(
                    step.get("guided_change_ratio") for step in trace
                ),
                "selector_trace": trace,
                **audit,
            }
            atomic_json(summary_path, summary)
            hashes[arm] = (canonical, state_sha, rgb_sha)
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": summary["success"],
                "m": summary["mean_m_t"],
                "overlap": summary["mean_prompt_shr_overlap_ratio"],
                "runtime_seconds": round(runtime, 2),
                "technical_pass": True,
            }), flush=True)
        if set(hashes) != set(ARMS) or len(set(hashes.values())) != 1:
            raise RuntimeError(f"three-arm pairing mismatch: {args.task} seed={seed}")


if __name__ == "__main__":
    main()
