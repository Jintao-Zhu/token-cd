"""Canonical paired Projected-SHR rollout for five tasks and seeds 0..299."""
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
from research.semantic_token_cd.projected_shr_policy import ProjectedSHRCDInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode


PROTOCOL = "PROJECTED_SHR_5TASK_0_299_V1"
TASKS = (
    "google_robot_close_drawer",
    "google_robot_open_drawer",
    "google_robot_move_near",
    "google_robot_pick_coke_can",
    "widowx_carrot_on_plate",
)
ARMS = ("proj_shr_eta000", "proj_shr_eta025")
ETAS = {"proj_shr_eta000": 0.0, "proj_shr_eta025": 0.25}
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0
EPS = 1e-8


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    seeds = sorted(set(seeds))
    if not seeds or any(seed < 0 or seed > 299 for seed in seeds):
        raise ValueError("seeds must be a non-empty subset of 0..299")
    return seeds


def build_policies(base) -> dict[str, ProjectedSHRCDInference]:
    policies = {}
    for arm, eta in ETAS.items():
        policy = copy.copy(base)
        policy.__class__ = ProjectedSHRCDInference
        _init_common(policy, LAMBDA)
        policy.beta = 0.0
        policy.projection_eta = eta
        policy.projection_eps = EPS
        policies[arm] = policy
    return policies


def ensure_config(artifact: Path, snapshot_artifact: Path) -> None:
    config = {
        "protocol_id": PROTOCOL,
        "tasks": list(TASKS),
        "seeds": "0-299",
        "arms": list(ARMS),
        "etas": ETAS,
        "snapshot_artifact": str(snapshot_artifact),
        "snapshot_policy": "read-only; missing snapshot/reference is fatal",
        "pairing": "canonical snapshot, initial state, and initial RGB hashes match existing SHR",
        "negative_branch": "unchanged beta=0 KMeans-K8 entity SHR with 4-neighbor harmonic reconstruction",
        "action_projection": "independent centered projection over each dimension's 256 executable action logits",
        "autoregressive_prefix": "clean greedy prefix, unchanged from canonical SHR implementation",
        "gripper": "dimension 6 remains positive",
        "common": {
            "lambda": LAMBDA,
            "kmeans_K": KMEANS_K,
            "kmeans_seed": KMEANS_SEED,
            "projection_eps": EPS,
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"CONFIG_LOCK differs: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def audit_trace(trace: list[dict], eta: float) -> dict:
    checks = {
        "all_feature_equal": bool(trace) and all(x.get("feature_equal") for x in trace),
        "all_guided_prefix": bool(trace) and all(x.get("guided_prefix") for x in trace),
        "all_reconstruction_finite": bool(trace) and all(x.get("reconstruction_finite") for x in trace),
        "all_projection_finite": bool(trace) and all(x.get("projection_finite") for x in trace),
        "all_eta_locked": bool(trace) and all(x.get("projection_eta") == eta for x in trace),
        "all_six_dimensions": bool(trace) and all(len(x.get("projection_alpha", [])) == 6 for x in trace),
        "all_gripper_positive": bool(trace) and all(x.get("gripper_positive_unchanged") for x in trace),
        "all_projection_orthogonal": bool(trace) and all(
            x.get("orthogonal_positive_relative_dot_max", 1.0) < 2e-4 for x in trace
        ),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"Projected-SHR audit failed: {checks}")
    return checks


def write_arrays(path: Path, logits: list[dict], actions: np.ndarray) -> None:
    payload = {"executed_actions": actions}
    for key in ("positive", "negative", "original_shr", "projected"):
        payload[f"{key}_logits"] = np.stack([record[key] for record in logits])
    np.savez_compressed(path, **payload)


def finite_mean(values) -> float | None:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
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
    snapshot_artifact = args.snapshot_artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    ensure_config(artifact, snapshot_artifact)
    task_root = artifact / "episodes" / args.task
    source_task_root = snapshot_artifact / "episodes" / args.task
    snapshot_dir = snapshot_artifact / "snapshots" / args.task
    task_root.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config))

    for seed in parse_seeds(args.seeds):
        snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
        reference_path = source_task_root / "shr_harmonic" / f"episode_{seed:03d}_summary.json"
        if not snapshot_path.exists() or not reference_path.exists():
            raise FileNotFoundError(f"missing canonical SHR source for {args.task} seed={seed}")
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        reference = json.loads(reference_path.read_text())
        reference_hashes = (
            reference["canonical_snapshot_sha256"],
            reference["initial_state_sha256"],
            reference["initial_rgb_sha256"],
        )
        if reference_hashes[0] != canonical:
            raise RuntimeError(f"snapshot hash mismatch for {args.task} seed={seed}")

        episode_hashes = {}
        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                summary = json.loads(summary_path.read_text())
                episode_hashes[arm] = (
                    summary["canonical_snapshot_sha256"],
                    summary["initial_state_sha256"],
                    summary["initial_rgb_sha256"],
                )
                continue

            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if (canonical, state_sha, rgb_sha) != reference_hashes:
                raise RuntimeError(f"canonical pairing mismatch for {args.task} seed={seed}")
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, action_jerk = run_episode(
                env, policy, instruction, obs
            )
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            audit = audit_trace(trace, ETAS[arm])
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "evaluation_seed": seed,
                "episode_id": seed,
                "arm": arm,
                "eta": ETAS[arm],
                "lambda": LAMBDA,
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
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
                "reference_summary": str(reference_path),
                "arrays_file": arrays_path.name,
                "action_jitter_index": action_jerk,
                "mean_residual_positive_cosine": finite_mean(
                    value for step in trace for value in step["residual_positive_cosine"]
                ),
                "mean_parallel_ratio": finite_mean(
                    value for step in trace for value in step["parallel_ratio"]
                ),
                "mean_projection_alpha": finite_mean(
                    value for step in trace for value in step["projection_alpha"]
                ),
                "mean_orthogonal_norm": finite_mean(
                    value for step in trace for value in step["orthogonal_norm"]
                ),
                "mean_lambda_eff": finite_mean(
                    value for step in trace for value in step["lambda_eff"]
                ),
                "projected_changes_original_shr_count": int(sum(
                    step["projected_changes_original_shr_count"] for step in trace
                )),
                "selector_trace": trace,
                **audit,
            }
            tmp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, summary_path)
            episode_hashes[arm] = (canonical, state_sha, rgb_sha)
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": summary["success"],
                "steps": steps,
                "runtime_seconds": round(runtime, 2),
                "technical_pass": True,
            }), flush=True)

        if any(value != reference_hashes for value in episode_hashes.values()):
            raise RuntimeError(f"cross-arm pairing mismatch for {args.task} seed={seed}")


if __name__ == "__main__":
    main()
