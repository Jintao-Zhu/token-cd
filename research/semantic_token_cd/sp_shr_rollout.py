"""Paired Boundary-SHR and Partial-SHR rollout on existing canonical snapshots."""
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
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode, write_arrays
from research.semantic_token_cd.sp_shr_policy import (
    BOUNDARY,
    PARTIAL50,
    StructurePreservingSHRCDInference,
)


PROTOCOL = "SP_SHR_BOUNDARY_PARTIAL50_3TASK_0_99_V1"
SOURCE_ARMS = ("vanilla", "shr_harmonic")
ARMS = ("boundary_shr", "partial_shr50")
VARIANTS = {"boundary_shr": BOUNDARY, "partial_shr50": PARTIAL50}
TASKS = (
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_pick_coke_can",
)
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    seeds = sorted(set(seeds))
    if not seeds or any(seed < 0 or seed > 99 for seed in seeds):
        raise ValueError("seeds must be a non-empty subset of 0..99")
    return seeds


def build_policies(base, task: str) -> dict[str, StructurePreservingSHRCDInference]:
    policies = {}
    for arm, variant in VARIANTS.items():
        policy = copy.copy(base)
        policy.__class__ = StructurePreservingSHRCDInference
        _init_common(policy, LAMBDA)
        policy.beta = 0.0
        policy.sp_variant = variant
        policy.partial_fraction = 0.5
        policy._task_id = TASK_INDEX[task]
        policies[arm] = policy
    return policies


def audit_trace(trace: list[dict], variant: str) -> dict:
    if not trace:
        raise RuntimeError("SP-SHR produced no trace")
    checks = {
        "all_feature_equal": all(step.get("feature_equal", False) for step in trace),
        "all_guided_prefix": all(step.get("guided_prefix", False) for step in trace),
        "all_reconstruction_finite": all(step.get("reconstruction_finite", False) for step in trace),
        "all_non_target_bit_identical": all(step.get("non_target_bit_identical", False) for step in trace),
        "all_variant_locked": all(step.get("sp_variant") == variant for step in trace),
        "all_targets_subset_semantic": all(
            set(step.get("selected_token_ids", [])) <= set(step.get("reference_semantic_token_ids", []))
            for step in trace
        ),
    }
    if variant == BOUNDARY:
        checks["all_boundary_partition_exact"] = all(
            all(
                set(entity.get("reconstructed_token_ids", []))
                | set(entity.get("boundary_token_ids", []))
                == set(entity.get("semantic_token_ids", []))
                and not (
                    set(entity.get("reconstructed_token_ids", []))
                    & set(entity.get("boundary_token_ids", []))
                )
                for entity in step.get("entity_regions", [])
                if entity.get("num_semantic_tokens", 0) > 0
            )
            for step in trace
        )
    elif variant == PARTIAL50:
        checks["all_partial_counts_exact"] = all(
            all(
                entity.get("num_reconstructed_tokens")
                == int(np.ceil(0.5 * entity.get("num_semantic_tokens")))
                for entity in step.get("entity_regions", [])
                if entity.get("num_semantic_tokens", 0) > 0
            )
            for step in trace
        )
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"SP-SHR audit failed: {checks}")
    return checks


def ensure_config(artifact: Path, snapshot_artifact: Path) -> None:
    config = {
        "protocol_id": PROTOCOL,
        "tasks": list(TASKS),
        "seeds": "0-99",
        "arms": list(ARMS),
        "snapshot_artifact": str(snapshot_artifact),
        "snapshot_policy": "read-only; missing snapshot/reference is fatal",
        "pairing": "canonical snapshot, initial state, and initial RGB hashes must match SHR",
        "common": {"lambda": LAMBDA, "kmeans_K": KMEANS_K, "kmeans_seed": KMEANS_SEED},
        "boundary_shr": "reconstruct mask minus one-token 4-neighbor morphological boundary",
        "partial_shr50": (
            "reconstruct ceil(0.5 * entity-region tokens), uniformly sampled without replacement; "
            "SeedSequence(task_id, episode_seed, replan_step, entity_index, 0x50534852)"
        ),
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"CONFIG_LOCK differs: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


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
    policies = build_policies(OpenVLAInference(**config), args.task)

    for seed in parse_seeds(args.seeds):
        snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
        reference_path = source_task_root / "shr_harmonic" / f"episode_{seed:03d}_summary.json"
        if not snapshot_path.exists() or not reference_path.exists():
            raise FileNotFoundError(f"missing canonical source for {args.task} seed={seed}")
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        reference = json.loads(reference_path.read_text())
        if reference.get("canonical_snapshot_sha256") != canonical:
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
            if state_sha != reference.get("initial_state_sha256"):
                raise RuntimeError(f"initial state hash mismatch for {args.task} seed={seed}")
            if rgb_sha != reference.get("initial_rgb_sha256"):
                raise RuntimeError(f"initial RGB hash mismatch for {args.task} seed={seed}")
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
            audit = audit_trace(trace, VARIANTS[arm])
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "evaluation_seed": seed,
                "episode_id": seed,
                "arm": arm,
                "method": "Boundary-SHR" if arm == "boundary_shr" else "Partial-SHR-50",
                "success": bool(result["success"]),
                "result": jsonable(result),
                "failure_reason": reason,
                "control_steps": steps,
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "lambda": LAMBDA,
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "reference_arm": "shr_harmonic",
                "reference_summary": str(reference_path),
                "source_snapshot": str(snapshot_path),
                "arrays_file": arrays_path.name,
                "action_jitter_index": action_jerk,
                "mean_semantic_tokens": float(np.mean([x["num_semantic_tokens"] for x in trace])),
                "mean_reconstructed_tokens": float(np.mean([x["num_tokens"] for x in trace])),
                "mean_preserved_semantic_tokens": float(np.mean([
                    x["num_preserved_semantic_tokens"] for x in trace
                ])),
                "mean_residual_norm": float(np.mean([x["residual_norm"] for x in trace])),
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

        reference_hashes = (
            reference["canonical_snapshot_sha256"],
            reference["initial_state_sha256"],
            reference["initial_rgb_sha256"],
        )
        if any(value != reference_hashes for value in episode_hashes.values()):
            raise RuntimeError(f"cross-arm pairing mismatch for {args.task} seed={seed}")


if __name__ == "__main__":
    main()
