"""Canonical-snapshot rollout driver for Adaptive-SHR v1."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.adaptive_shr_policy import (
    AdaptiveSHRCDInference,
    INITIAL_LAMBDA,
    N_SHIFT_TOKENS,
    REDUCED_LAMBDA,
    SHIFT_THRESHOLD,
)
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.semantic_recon_rollout import TASKS, _init_common
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)


PROTOCOL = "ADAPTIVE_SHR_ACTION_SHIFT_V1"
ARM = "adaptive_shr"
REFERENCE_ARMS = ("vanilla", "semantic_recon_k8_m10", "shr_harmonic")
KMEANS_K = 8
KMEANS_SEED = 0


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    seeds = sorted(set(seeds))
    if not seeds or any(seed < 0 or seed > 299 for seed in seeds):
        raise ValueError("seeds must be a non-empty subset of 0..299")
    return seeds


def build_policy(base):
    policy = copy.copy(base)
    policy.__class__ = AdaptiveSHRCDInference
    _init_common(policy, INITIAL_LAMBDA)
    policy.beta = 0.0
    return policy


def load_and_validate_references(task_root: Path, seed: int) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for arm in REFERENCE_ARMS:
        path = task_root / arm / f"episode_{seed:03d}_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing canonical reference: {path}")
        summaries[arm] = json.loads(path.read_text())
    for field in (
        "canonical_snapshot_sha256",
        "initial_state_sha256",
        "initial_rgb_sha256",
    ):
        values = {summary.get(field) for summary in summaries.values()}
        if len(values) != 1 or None in values:
            raise RuntimeError(f"canonical references disagree on {field}: task={task_root.name} seed={seed}")
    return summaries


def audit_trace(trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError("Adaptive-SHR produced no selector trace")
    checks = {
        "all_feature_equal": all(step.get("feature_equal", False) for step in trace),
        "all_guided_prefix": all(step.get("guided_prefix", False) for step in trace),
        "all_reconstruction_finite": all(step.get("reconstruction_finite", False) for step in trace),
        "all_beta_zero": all(abs(float(step.get("beta", -1.0))) < 1e-12 for step in trace),
        "all_initial_lambda_locked": all(
            abs(float(step.get("initial_lambda", -1.0)) - INITIAL_LAMBDA) < 1e-12
            for step in trace
        ),
        "all_final_lambda_valid": all(
            float(step.get("final_lambda", -1.0)) in (INITIAL_LAMBDA, REDUCED_LAMBDA)
            for step in trace
        ),
        "all_shift_denominator_six": all(
            step.get("shift_denominator") == N_SHIFT_TOKENS for step in trace
        ),
        "all_threshold_locked": all(
            abs(float(step.get("shift_threshold", -1.0)) - SHIFT_THRESHOLD) < 1e-12
            for step in trace
        ),
        "all_gripper_unchanged": all(step.get("gripper_token_unchanged", False) for step in trace),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"Adaptive-SHR audit failed: {checks}")
    return checks


def write_config_lock(artifact: Path, snapshot_artifact: Path) -> None:
    lock = {
        "protocol_id": PROTOCOL,
        "arm": ARM,
        "tasks": list(TASKS),
        "seeds": list(range(300)),
        "reference_arms": list(REFERENCE_ARMS),
        "snapshot_artifact": str(snapshot_artifact),
        "pairing": "strict reuse; no reset capture; all snapshot/state/RGB hashes must match",
        "negative_branch": "unchanged beta=0 KMeans-K8 semantic four-neighbor harmonic SHR",
        "kmeans": {"K": KMEANS_K, "seed": KMEANS_SEED},
        "initial_lambda": INITIAL_LAMBDA,
        "reduced_lambda": REDUCED_LAMBDA,
        "shift_threshold": SHIFT_THRESHOLD,
        "shift_rule": "lambda=0.5 iff S<=0.33; lambda=0.25 iff S>0.33",
        "shift_tokens": "OpenVLA action dimensions 0..5; gripper dimension 6 excluded and clean",
        "episode_trigger_aggregation": "any control-step trigger",
        "episode_shift_ratio_aggregation": "maximum control-step shift ratio",
        "gpu_allowlist": [1, 2, 3, 4, 5],
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError(f"CONFIG_LOCK differs from requested protocol: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", default="0-299")
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3, 4, 5), required=True)
    parser.add_argument("--worker-id", default="manual")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    snapshot_artifact = args.snapshot_artifact.resolve()
    for directory in ("rollout_logs", "episode_json", "paired_results", "statistics"):
        (artifact / directory).mkdir(parents=True, exist_ok=True)
    write_config_lock(artifact, snapshot_artifact)

    source_task_root = snapshot_artifact / "episodes" / args.task
    snapshot_dir = snapshot_artifact / "snapshots" / args.task
    arm_dir = artifact / "episode_json" / args.task / ARM
    arm_dir.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config))

    for seed in parse_seeds(args.seeds):
        summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
        arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            print(json.dumps({"skip": True, "task": args.task, "seed": seed}), flush=True)
            continue

        references = load_and_validate_references(source_task_root, seed)
        snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"missing canonical snapshot (capture is forbidden): {snapshot_path}")
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        reference = references["shr_harmonic"]
        if canonical != reference["canonical_snapshot_sha256"]:
            raise RuntimeError(f"snapshot SHA mismatch: {args.task} seed={seed}")

        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if state_sha != reference["initial_state_sha256"] or rgb_sha != reference["initial_rgb_sha256"]:
            raise RuntimeError(f"restored initial state/RGB mismatch: {args.task} seed={seed}")
        instruction = env.unwrapped.get_language_instruction()
        if instruction != reference.get("instruction"):
            raise RuntimeError(f"instruction mismatch: {args.task} seed={seed}")

        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        started = time.monotonic()
        result, steps, reason, actions, action_jerk = run_episode(env, policy, instruction, obs)
        runtime = time.monotonic() - started
        trace = jsonable(policy._episode_trace)
        audit = audit_trace(trace)
        write_arrays(arrays_path, policy._episode_logits, actions)

        triggers = [bool(step["adapt_trigger"]) for step in trace]
        shift_ratios = [float(step["shift_ratio"]) for step in trace]
        episode_trigger = any(triggers)
        summary = {
            "protocol_id": PROTOCOL,
            "task": args.task,
            "environment_id": environment_id,
            "seed": seed,
            "evaluation_seed": seed,
            "episode_id": seed,
            "arm": ARM,
            "instruction": instruction,
            "success": bool(result["success"]),
            "result": jsonable(result),
            "failure_reason": reason,
            "trajectory_length": steps,
            "control_steps": steps,
            "runtime_seconds": runtime,
            "gpu_id": args.gpu,
            "worker_id": args.worker_id,
            "canonical_snapshot_sha256": canonical,
            "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha,
            "snapshot_artifact": str(snapshot_artifact),
            "reference_arm": "shr_harmonic",
            "reference_success": bool(reference["success"]),
            "arrays_file": arrays_path.name,
            "kmeans_K": KMEANS_K,
            "kmeans_seed": KMEANS_SEED,
            "beta": 0.0,
            "initial_lambda": INITIAL_LAMBDA,
            "final_lambda": REDUCED_LAMBDA if episode_trigger else INITIAL_LAMBDA,
            "adapt_trigger": episode_trigger,
            "shift_ratio": max(shift_ratios),
            "triggered_control_steps": sum(triggers),
            "trigger_fraction": sum(triggers) / len(triggers),
            "step_final_lambdas": [float(step["final_lambda"]) for step in trace],
            "step_adapt_triggers": triggers,
            "step_shift_ratios": shift_ratios,
            "vanilla_action": [step["vanilla_action"] for step in trace],
            "shr_action": [step["shr_action"] for step in trace],
            "adaptive_action": [step["adaptive_action"] for step in trace],
            "action_jerk": action_jerk,
            "action_jitter_index": action_jerk,
            "selector_trace": trace,
            **audit,
        }
        tmp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, summary_path)
        print(json.dumps({
            "task": args.task,
            "seed": seed,
            "success": summary["success"],
            "shr_success": summary["reference_success"],
            "adapt_trigger": episode_trigger,
            "max_shift_ratio": summary["shift_ratio"],
            "steps": steps,
            "runtime_seconds": round(runtime, 2),
            "technical_pass": True,
        }), flush=True)


if __name__ == "__main__":
    main()
