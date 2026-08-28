"""One-arm rollout and smoke test for Orthogonal Dual Attention-CD."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.orthogonal_policy import OrthogonalDualCDInference
from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode


PROTOCOL = "ORTHOGONAL_DUAL_ATTENTION_CD_SMOKE_V1"
ARM = "orthogonal_dual"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
)


def build_policy(base, lambda_geom: float, lambda_sem: float):
    policy = copy.copy(base)
    policy.__class__ = OrthogonalDualCDInference
    policy.alpha = lambda_geom
    policy.lambd = lambda_geom
    policy.lambda_geom = lambda_geom
    policy.lambda_sem = lambda_sem
    policy.orthogonal_eps = 1e-8
    policy.preserve_last_action_token = True
    policy.kmeans_K = 8
    policy.kmeans_seed = 0
    policy.selection_mode = "semantic"
    policy.attention_layer_start = 16
    policy.attention_layer_end = 32
    policy.attention_mask_value = -1e4
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    return policy


def write_arrays(path: Path, logits: list[dict], actions: np.ndarray) -> None:
    if not logits:
        raise RuntimeError("Orthogonal rollout produced no logits")
    np.savez_compressed(
        path,
        positive_logits=np.stack([record["positive"] for record in logits]),
        uniform_negative_logits=np.stack(
            [record["uniform_negative"] for record in logits]
        ),
        semantic_negative_logits=np.stack(
            [record["semantic_negative"] for record in logits]
        ),
        final_logits=np.stack([record["final"] for record in logits]),
        executed_actions=actions,
    )


def write_config(
    artifact: Path,
    task: str,
    seed: int,
    lambda_geom: float,
    lambda_sem: float,
) -> None:
    config = {
        "experiment": PROTOCOL,
        "stage": "one_process_one_seed_smoke",
        "task": task,
        "seed": seed,
        "arm": ARM,
        "lambda_geom": lambda_geom,
        "lambda_sem": lambda_sem,
        "orthogonal_eps": 1e-8,
        "uniform_geometry": "16x16 even-row/even-column strided grid; 64 tokens",
        "semantic_selector": "entity_set KMeans K=8 seed=0 n_init=10",
        "attention_layers": [16, 32],
        "attention_mask_value_requested": -10000.0,
        "attention_mask_value_bfloat16_effective": float(
            torch.tensor(-1e4, dtype=torch.bfloat16).item()
        ),
        "projection_space": "full OpenVLA logits; independently per action token",
        "diagnostic_space": "full vocabulary and 256-token action vocabulary",
        "last_action_token": "positive branch preserved to match existing PCD/OpenVLA protocol",
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("CONFIG_LOCK.json differs from requested smoke configuration")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="google_robot_pick_coke_can")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-geom", type=float, default=0.5)
    parser.add_argument("--lambda-sem", type=float, default=0.35)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 50:
        raise ValueError("Smoke seed must be in 0..49")
    if args.lambda_geom < 0 or args.lambda_sem < 0:
        raise ValueError("Lambda values must be non-negative")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    write_config(
        artifact, args.task, args.seed, args.lambda_geom, args.lambda_sem
    )
    arm_dir = artifact / "episodes" / args.task / ARM
    arm_dir.mkdir(parents=True, exist_ok=True)
    summary_path = arm_dir / f"episode_{args.seed:03d}_summary.json"
    arrays_path = arm_dir / f"episode_{args.seed:03d}_arrays.npz"
    if summary_path.exists() and arrays_path.exists():
        print(json.dumps({"skip": args.task, "seed": args.seed, "arm": ARM}), flush=True)
        return

    env, environment_id = make_environment(args.task)
    snapshot = capture_snapshot(env, args.seed)
    canonical = snapshot_sha(snapshot)
    obs, state_sha, rgb_sha = restore_snapshot(env, args.seed, snapshot)
    instruction = env.unwrapped.get_language_instruction()
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(
        OpenVLAInference(**policy_config), args.lambda_geom, args.lambda_sem
    )
    policy.reset(instruction, seed=args.seed)
    policy._episode_trace = []
    policy._episode_logits = []
    result, steps, reason, actions, jitter = run_episode(
        env, policy, instruction, obs
    )
    write_arrays(arrays_path, policy._episode_logits, actions)
    trace = jsonable(policy._episode_trace)
    if not trace:
        raise RuntimeError("Orthogonal rollout produced no selector trace")

    feature_equal = all(step["feature_equal"] for step in trace)
    uniform_hooks = all(
        step["uniform_attention_mask"]["hook_calls"] == 112 for step in trace
    )
    semantic_hooks = all(
        step["semantic_attention_mask"]["hook_calls"] == 112 for step in trace
    )
    scalar_diagnostic_keys = (
        "norm_r_geom",
        "norm_r_sem",
        "norm_r_sem_ortho",
        "cos_sim_raw",
        "ortho_ratio",
        "ortho_dot_max_abs",
        "ortho_relative_error_max",
        "projection_coefficient_mean",
    )
    finite_diagnostics = all(
        np.isfinite([
            diagnostics[key]
            for space in ("orthogonal_full_vocab", "orthogonal_action_vocab")
            for diagnostics in (step[space],)
            for key in scalar_diagnostic_keys
        ]).all()
        for step in trace
    )
    orthogonal_audit = all(
        step[space]["ortho_relative_error_max"] <= 1e-5
        for step in trace
        for space in ("orthogonal_full_vocab", "orthogonal_action_vocab")
    )
    technical_pass = bool(
        feature_equal
        and uniform_hooks
        and semantic_hooks
        and finite_diagnostics
        and orthogonal_audit
    )
    summary = {
        "protocol_id": PROTOCOL,
        "environment_id": environment_id,
        "task": args.task,
        "seed": args.seed,
        "episode_id": args.seed,
        "arm": ARM,
        "instruction": instruction,
        "success": bool(result["success"]),
        "failure_reason": reason,
        "control_steps": steps,
        "action_jitter_index": jitter,
        "lambda_geom": args.lambda_geom,
        "lambda_sem": args.lambda_sem,
        "canonical_snapshot_sha256": canonical,
        "initial_state_sha256": state_sha,
        "initial_rgb_sha256": rgb_sha,
        "arrays_file": arrays_path.name,
        "selector_trace": trace,
        "all_visual_features_bit_identical": feature_equal,
        "all_uniform_hook_audits_pass": uniform_hooks,
        "all_semantic_hook_audits_pass": semantic_hooks,
        "all_diagnostics_finite": finite_diagnostics,
        "all_orthogonal_error_audits_pass": orthogonal_audit,
        "technical_pass": technical_pass,
        "result": jsonable(result),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (artifact / "SMOKE_RESULT.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "seed": args.seed,
                "success": summary["success"],
                "control_steps": steps,
                "technical_pass": technical_pass,
                "summary": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "task": args.task,
                "seed": args.seed,
                "success": summary["success"],
                "steps": steps,
                "technical_pass": technical_pass,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not technical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
