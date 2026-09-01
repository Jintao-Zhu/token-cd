"""Single-arm rollout driver for ST-SHR beta selection on seeds 0-99."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, capture_snapshot, jsonable, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.semantic_recon_rollout import TASKS, _init_common
from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode, write_arrays
from research.semantic_token_cd.st_shr_policy import STSHRCDInference, STReconCDInference


PROTOCOL = "ST_SHR_CD_V1"
BETAS = (0.25, 1.0, 4.0)
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    if not out or any(seed < 0 or seed > 299 for seed in out):
        raise ValueError("seeds must be a non-empty subset of 0..299")
    return sorted(set(out))


def arm_name(beta: float) -> str:
    return "st_shr_beta" + {0.25: "025", 1.0: "100", 4.0: "400"}[beta]


def build_policy(base, beta: float):
    policy = copy.copy(base)
    policy.__class__ = STSHRCDInference
    _init_common(policy, LAMBDA)
    policy.beta = beta
    return policy


def audit_trace(trace: list[dict], beta: float) -> dict:
    if not trace:
        raise RuntimeError("ST-SHR produced no trace")
    checks = {
        "all_feature_equal": all(x.get("feature_equal", False) for x in trace),
        "all_guided_prefix": all(x.get("guided_prefix", False) for x in trace),
        "all_reconstruction_finite": all(x.get("reconstruction_finite", False) for x in trace),
        "all_beta_locked": all(abs(x.get("beta", -1) - beta) < 1e-12 for x in trace),
        "first_step_fallback": trace[0].get("temporal_prior_used_fraction") == 0.0,
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"ST-SHR audit failed: {checks}")
    return checks


def finite_mean(values) -> float | None:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--beta", type=float, choices=BETAS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument("--method", choices=("st_shr", "st_recon"), default="st_shr")
    args = parser.parse_args()
    if args.gpu == 0:
        raise ValueError("GPU 0 is excluded due to hardware faults")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    seeds = parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    arm = "st_recon_beta100" if args.method == "st_recon" else arm_name(args.beta)
    arm_dir = artifact / "episodes" / args.task / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config), args.beta)
    if args.method == "st_recon":
        policy.__class__ = STReconCDInference

    for seed in seeds:
        summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
        arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            print(json.dumps({"skip": True, "task": args.task, "arm": arm, "seed": seed}), flush=True)
            continue
        started = time.monotonic()
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        result, steps, reason, actions, action_jerk = run_episode(env, policy, instruction, obs)
        runtime = time.monotonic() - started
        trace = jsonable(policy._episode_trace)
        audit = audit_trace(trace, args.beta)
        write_arrays(arrays_path, policy._episode_logits, actions)
        displacements = [e.get("centroid_displacement") for x in trace
                         for e in x.get("entity_alignment", [])]
        summary = {
            "protocol_id": PROTOCOL, "task": args.task, "environment_id": environment_id,
            "seed": seed, "evaluation_seed": seed, "episode_id": seed, "arm": arm,
            "beta": args.beta, "lambda": LAMBDA, "kmeans_K": KMEANS_K,
            "kmeans_seed": KMEANS_SEED, "instruction": instruction,
            "success": bool(result["success"]), "result": jsonable(result),
            "failure_reason": reason, "trajectory_length": steps, "control_steps": steps,
            "runtime_seconds": runtime, "gpu_id": args.gpu, "worker_id": args.worker_id,
            "canonical_snapshot_sha256": canonical, "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha, "arrays_file": arrays_path.name,
            "temporal_prior_used_fraction": finite_mean(x["temporal_prior_used_fraction"] for x in trace),
            "fallback_fraction": finite_mean(x["fallback_fraction"] for x in trace),
            "semantic_centroid_displacement": finite_mean(displacements),
            "residual_temporal_cos": finite_mean(x.get("residual_temporal_cos") for x in trace),
            "residual_jerk": finite_mean(x.get("residual_jerk") for x in trace),
            "action_jerk": action_jerk, "action_jitter_index": action_jerk,
            "selector_trace": trace, **audit,
        }
        tmp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, summary_path)
        print(json.dumps({"task": args.task, "seed": seed, "arm": arm,
                          "success": summary["success"], "steps": steps,
                          "runtime_seconds": round(runtime, 2), "technical_pass": True}), flush=True)


if __name__ == "__main__":
    main()
