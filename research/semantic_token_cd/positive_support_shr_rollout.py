"""Canonical paired rollout for PSC-SHR K10/K20 on drawer seeds 0..99."""
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
from research.semantic_token_cd.positive_support_shr_policy import PositiveSupportSHRInference
from research.semantic_token_cd.semantic_recon_rollout import _init_common
from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode


TASKS = ("google_robot_close_drawer", "google_robot_open_drawer")
REFERENCE_ARMS = ("vanilla", "shr_harmonic")


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_arrays(path: Path, records, actions) -> None:
    payload = {key: np.stack([row[key] for row in records]) for key in (
        "positive_logits", "negative_logits", "fused_logits", "filtered_logits"
    )}
    payload["executed_actions"] = np.asarray(actions, dtype=np.float32)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(tmp, path)


def build_policy(base, k: int):
    policy = copy.copy(base)
    policy.__class__ = PositiveSupportSHRInference
    _init_common(policy, 0.5)
    policy.beta = 0.0
    policy.support_top_k = k
    policy._episode_psc_logits = []
    return policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--k", type=int, choices=(10, 20), required=True)
    ap.add_argument("--seeds", required=True, help="inclusive lo-hi within 0..99")
    ap.add_argument("--gpu", type=int, choices=(0, 1, 2, 3, 4), required=True)
    ap.add_argument("--worker-id", required=True)
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split("-", 1))
    if not (0 <= lo <= hi <= 99):
        raise ValueError("PSC pilot seeds must lie in 0..99")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact, source = args.artifact.resolve(), args.snapshot_artifact.resolve()
    arm = f"psc_shr_k{args.k}"
    summary_dir = artifact / "episodes" / args.task / arm
    array_dir = artifact / "episode_logits" / args.task / arm
    summary_dir.mkdir(parents=True, exist_ok=True); array_dir.mkdir(parents=True, exist_ok=True)
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config), args.k)
    for seed in range(lo, hi + 1):
        summary_path = summary_dir / f"episode_{seed:03d}_summary.json"
        arrays_path = array_dir / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            print(json.dumps({"skip": True, "task": args.task, "k": args.k, "seed": seed}), flush=True)
            continue
        refs = {arm0: json.loads((source / "episodes" / args.task / arm0 /
                                  f"episode_{seed:03d}_summary.json").read_text())
                for arm0 in REFERENCE_ARMS}
        for field in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
            if refs["vanilla"][field] != refs["shr_harmonic"][field]:
                raise RuntimeError(f"reference mismatch {field}: {args.task} seed={seed}")
        snapshot_path = source / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        if canonical != refs["shr_harmonic"]["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot hash mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        if state_sha != refs["shr_harmonic"]["initial_state_sha256"] or rgb_sha != refs["shr_harmonic"]["initial_rgb_sha256"]:
            raise RuntimeError("restored snapshot mismatch")
        if instruction != refs["shr_harmonic"]["instruction"]:
            raise RuntimeError("instruction mismatch")
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []; policy._episode_logits = []; policy._episode_psc_logits = []
        started = time.monotonic()
        result, steps, reason, actions, action_jerk = run_episode(env, policy, instruction, obs)
        runtime = time.monotonic() - started
        trace = jsonable(policy._episode_trace)
        checks = {
            "trace_nonempty": bool(trace),
            "all_top_k_locked": all(x.get("psc_top_k") == args.k for x in trace),
            "all_lambda_locked": all(abs(x.get("lambda", -1) - 0.5) < 1e-12 for x in trace),
            "all_beta_zero": all(abs(x.get("beta", -1)) < 1e-12 for x in trace),
            "all_guided_prefix": all(x.get("guided_prefix") is True for x in trace),
            "all_support_valid": all(x.get("positive_support_valid") is True for x in trace),
            "all_gripper_passthrough": all(x.get("gripper_passthrough") is True for x in trace),
            "all_reconstruction_finite": all(x.get("reconstruction_finite") is True for x in trace),
        }
        checks["technical_pass"] = all(checks.values()) and len(policy._episode_psc_logits) == steps
        if not checks["technical_pass"]:
            raise RuntimeError(f"PSC technical audit failed: {checks}")
        write_arrays(arrays_path, policy._episode_psc_logits, actions)
        changed = np.asarray([x["changed_by_filter"] for x in trace], dtype=bool)
        summary = {
            "protocol_id": "PSC_SHR_DRAWER_0_99_V1", "arm": arm, "task": args.task,
            "seed": seed, "evaluation_seed": seed, "environment_id": environment_id,
            "instruction": instruction, "success": bool(result["success"]),
            "result": jsonable(result), "failure_reason": reason,
            "trajectory_length": steps, "runtime_seconds": runtime,
            "gpu_id": args.gpu, "worker_id": args.worker_id,
            "canonical_snapshot_sha256": canonical, "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha, "lambda": 0.5, "beta": 0.0,
            "support_top_k": args.k, "shr_success": bool(refs["shr_harmonic"]["success"]),
            "vanilla_success": bool(refs["vanilla"]["success"]),
            "changed_dimensions": int(changed.sum()),
            "changed_by_dimension": changed.sum(axis=0).tolist(),
            "changed_steps": int(changed.any(axis=1).sum()),
            "arrays_file": str(arrays_path.relative_to(artifact)),
            "action_jitter_index": action_jerk, "selector_trace": trace, **checks,
        }
        atomic_json(summary_path, summary)
        print(json.dumps({"task": args.task, "k": args.k, "seed": seed,
                          "success": summary["success"], "shr_success": summary["shr_success"],
                          "changed_dimensions": summary["changed_dimensions"],
                          "runtime_seconds": round(runtime, 2), "technical_pass": True}), flush=True)


if __name__ == "__main__":
    main()
