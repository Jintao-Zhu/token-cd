"""Canonical-snapshot rollout for Instruction-aware Component SHR v1."""
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
from research.semantic_token_cd.instruction_component_shr_policy import (
    CENTER_SIGMA,
    CENTER_WEIGHT,
    IMAGE_CENTER,
    LAMBDA,
    SEMANTIC_WEIGHT,
    SIZE_WEIGHT,
    InstructionComponentSHRInference,
)
from research.semantic_token_cd.semantic_recon_rollout import _init_common
from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode, write_arrays


PROTOCOL = "INSTRUCTION_COMPONENT_SHR_V1"
ARM = "ic_shr"
TASKS = (
    "google_robot_close_drawer",
    "google_robot_open_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
REFERENCE_ARMS = ("vanilla", "semantic_recon_k8_m10", "shr_harmonic")
KMEANS_K = 8
KMEANS_SEED = 0


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    seeds = sorted(set(seeds))
    if not seeds or any(seed < 0 or seed > 299 for seed in seeds):
        raise ValueError("seeds must be a non-empty subset of 0..299")
    return seeds


def atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_config_lock(artifact: Path, snapshot_artifact: Path) -> None:
    lock = {
        "protocol_id": PROTOCOL,
        "arm": ARM,
        "tasks": list(TASKS),
        "seeds": list(range(300)),
        "reference_arms": list(REFERENCE_ARMS),
        "snapshot_artifact": str(snapshot_artifact),
        "pairing": "strict canonical snapshot reuse; no random reset/capture",
        "unchanged": "OpenVLA, KMeans K=8 semantic selection, beta=0 harmonic reconstruction, guided CD lambda=0.5",
        "only_change": "atomic instruction-aware 4-neighbor component filter",
        "component_score": "0.5*S_sem + 0.3*S_center + 0.2*S_size",
        "semantic_score": "mean token cosine with full instruction embedding",
        "image_center": list(IMAGE_CENTER),
        "center_sigma": CENTER_SIGMA,
        "target_rule": "Top1 for one entity; Top2 for two entities; never split a component",
        "kmeans": {"K": KMEANS_K, "seed": KMEANS_SEED},
        "lambda": LAMBDA,
        "gpu_allowlist": [0, 1, 2, 3, 4, 5],
        "max_workers_per_gpu": 6,
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError(f"CONFIG_LOCK differs from IC-SHR v1 protocol: {path}")
    atomic_write_json(path, lock)


def build_policy(base):
    policy = copy.copy(base)
    policy.__class__ = InstructionComponentSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0
    return policy


def load_references(task_root: Path, seed: int) -> dict[str, dict]:
    refs = {}
    for arm in REFERENCE_ARMS:
        path = task_root / arm / f"episode_{seed:03d}_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing canonical reference: {path}")
        refs[arm] = json.loads(path.read_text())
    for field in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
        values = {ref.get(field) for ref in refs.values()}
        if len(values) != 1 or None in values:
            raise RuntimeError(f"reference mismatch for {field}: task={task_root.name} seed={seed}")
    return refs


def audit_trace(trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError("IC-SHR produced no trace")
    checks = {
        "all_feature_equal": all(s.get("feature_equal") is True for s in trace),
        "all_guided_prefix": all(s.get("guided_prefix") is True for s in trace),
        "all_reconstruction_finite": all(s.get("reconstruction_finite") is True for s in trace),
        "all_non_target_bit_identical": all(s.get("non_target_bit_identical") is True for s in trace),
        "all_components_atomic": all(s.get("component_atomic") is True for s in trace),
        "all_lambda_locked": all(abs(float(s.get("lambda", -1)) - LAMBDA) < 1e-12 for s in trace),
        "all_beta_zero": all(abs(float(s.get("beta", -1))) < 1e-12 for s in trace),
        "all_kmeans_locked": all(s.get("kmeans_K") == 8 and s.get("kmeans_seed") == 0 for s in trace),
        "all_weights_locked": all(s.get("component_weights") == {
            "semantic": SEMANTIC_WEIGHT, "center": CENTER_WEIGHT, "size": SIZE_WEIGHT
        } for s in trace),
        "all_mask_reduced_or_equal": all(s["mask_tokens_after"] <= s["mask_tokens_before"] for s in trace),
        "all_target_rule_valid": all(s["selected_num_components"] <= s["target_component_count"] for s in trace),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"IC-SHR audit failed: {checks}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", default="0-299")
    ap.add_argument("--gpu", type=int, choices=(0, 1, 2, 3, 4, 5), required=True)
    ap.add_argument("--worker-id", default="manual")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    snapshot_artifact = args.snapshot_artifact.resolve()
    for directory in ("rollout", "episode_summary", "masks", "component_stats", "paired_results", "rollout_logs"):
        (artifact / directory).mkdir(parents=True, exist_ok=True)
    write_config_lock(artifact, snapshot_artifact)
    summary_dir = artifact / "episode_summary" / args.task / ARM
    array_dir = artifact / "rollout" / args.task / ARM
    mask_dir = artifact / "masks" / args.task
    for directory in (summary_dir, array_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_task_root = snapshot_artifact / "episodes" / args.task
    snapshot_dir = snapshot_artifact / "snapshots" / args.task
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config))

    for seed in parse_seeds(args.seeds):
        summary_path = summary_dir / f"episode_{seed:03d}_summary.json"
        arrays_path = array_dir / f"episode_{seed:03d}_arrays.npz"
        mask_path = mask_dir / f"episode_{seed:03d}_components.json"
        if summary_path.exists() and arrays_path.exists() and mask_path.exists():
            print(json.dumps({"skip": True, "task": args.task, "seed": seed}), flush=True)
            continue
        refs = load_references(source_task_root, seed)
        snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"missing canonical snapshot: {snapshot_path}")
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        reference = refs["shr_harmonic"]
        if canonical != reference["canonical_snapshot_sha256"]:
            raise RuntimeError(f"snapshot SHA mismatch: {args.task} seed={seed}")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if state_sha != reference["initial_state_sha256"] or rgb_sha != reference["initial_rgb_sha256"]:
            raise RuntimeError(f"restored snapshot mismatch: {args.task} seed={seed}")
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
        mask_payload = {
            "task": args.task, "seed": seed, "instruction": instruction,
            "steps": [{k: step[k] for k in (
                "original_group_tokens", "selected_components", "all_components",
                "num_components", "selected_num_components", "mask_tokens_before",
                "mask_tokens_after", "removed_token_ids", "removed_token_fraction",
            )} for step in trace],
        }
        atomic_write_json(mask_path, mask_payload)
        before = [int(s["mask_tokens_before"]) for s in trace]
        after = [int(s["mask_tokens_after"]) for s in trace]
        ncomp = [int(s["num_components"]) for s in trace]
        selected_comp = [int(s["selected_num_components"]) for s in trace]
        summary = {
            "protocol_id": PROTOCOL, "arm": ARM, "task": args.task,
            "environment_id": environment_id, "seed": seed, "evaluation_seed": seed,
            "episode_id": seed, "instruction": instruction,
            "success": bool(result["success"]), "result": jsonable(result),
            "failure_reason": reason, "trajectory_length": steps, "control_steps": steps,
            "runtime_seconds": runtime, "gpu_id": args.gpu, "worker_id": args.worker_id,
            "canonical_snapshot_sha256": canonical, "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha, "snapshot_artifact": str(snapshot_artifact),
            "reference_arm": "shr_harmonic", "reference_success": bool(reference["success"]),
            "reference_successes": {arm: bool(ref["success"]) for arm, ref in refs.items()},
            "arrays_file": str(arrays_path.relative_to(artifact)),
            "masks_file": str(mask_path.relative_to(artifact)),
            "kmeans_K": KMEANS_K, "kmeans_seed": KMEANS_SEED, "beta": 0.0, "lambda": LAMBDA,
            "first_action": trace[0]["first_action"], "guided_action": trace[0]["guided_action"],
            "mean_num_components": float(np.mean(ncomp)),
            "mean_selected_num_components": float(np.mean(selected_comp)),
            "mean_filtered_num_components": float(np.mean(np.asarray(ncomp) - np.asarray(selected_comp))),
            "mean_mask_tokens_before": float(np.mean(before)),
            "mean_mask_tokens_after": float(np.mean(after)),
            "deleted_token_fraction": float(1.0 - sum(after) / sum(before)),
            "action_jerk": action_jerk, "action_jitter_index": action_jerk,
            "selector_trace": trace, **audit,
        }
        atomic_write_json(summary_path, summary)
        print(json.dumps({
            "task": args.task, "seed": seed, "success": summary["success"],
            "shr_success": summary["reference_success"], "components": summary["mean_num_components"],
            "mask_before": summary["mean_mask_tokens_before"], "mask_after": summary["mean_mask_tokens_after"],
            "runtime_seconds": round(runtime, 2), "technical_pass": True,
        }), flush=True)


if __name__ == "__main__":
    main()
