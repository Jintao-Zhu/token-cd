"""Exact-paired four-arm rollout for the six-task Orthogonal Dual-CD study."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.orthogonal_dual_rollout import (
    build_policy as build_orthogonal_policy,
    write_arrays as write_orthogonal_arrays,
)
from research.semantic_token_cd.spatial_grid_policy import (
    SpatialGridAttentionCDInference,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays as write_standard_arrays,
)


PROTOCOL = "ORTHOGONAL_DUAL_FOUR_ARM_6TASK_50_V1"
TASKS = (
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
ARMS = ("vanilla", "attn_uniform", "attn_semantic", "orthogonal_dual")
EXPECTED_HOOK_CALLS = 112
ORTHOGONAL_ERROR_LIMIT = 1e-5


def build_vanilla_policy(base):
    policy = copy.copy(base)
    policy.__class__ = AuditedVanillaInference
    policy._episode_trace = []
    policy._episode_logits = []
    return policy


def build_attention_policy(base, mode: str):
    policy = copy.copy(base)
    policy.__class__ = SpatialGridAttentionCDInference
    policy.alpha = 0.5
    policy.lambd = 0.5
    policy.kmeans_K = 8
    policy.kmeans_seed = 0
    policy.selection_mode = "semantic"
    policy.spatial_selection_mode = mode
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


def build_policies(base):
    return {
        "vanilla": build_vanilla_policy(base),
        "attn_uniform": build_attention_policy(base, "grid_strided"),
        "attn_semantic": build_attention_policy(base, "semantic_hard"),
        "orthogonal_dual": build_orthogonal_policy(base, 0.5, 0.35),
    }


def locked_config() -> dict:
    return {
        "experiment": PROTOCOL,
        "benchmark": "SIMPLER six-task Google Robot and WidowX expansion",
        "tasks": list(TASKS),
        "seeds": list(range(50)),
        "arms": list(ARMS),
        "execution": (
            "one task per process; capture one snapshot per seed and restore the "
            "same snapshot before each of four sequential arms"
        ),
        "reuse_old_episodes": False,
        "lambda_attention": 0.5,
        "lambda_geom": 0.5,
        "lambda_sem": 0.35,
        "orthogonal_eps": 1e-8,
        "orthogonal_error_limit": ORTHOGONAL_ERROR_LIMIT,
        "attention_layers": [16, 32],
        "attention_mask_value_requested": -10000.0,
        "attention_mask_value_bfloat16_effective": float(
            torch.tensor(-1e4, dtype=torch.bfloat16).item()
        ),
        "uniform_geometry": (
            "16x16 even-row/even-column strided grid; exactly 64 tokens"
        ),
        "semantic_selector": "entity_set KMeans K=8 seed=0 n_init=10",
        "last_action_token": "positive branch preserved for all CD arms",
        "expected_attention_hook_calls_per_branch_per_control_step": (
            EXPECTED_HOOK_CALLS
        ),
    }


def write_config(artifact: Path) -> None:
    expected = locked_config()
    path = artifact / "CONFIG_LOCK.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError("CONFIG_LOCK.json differs from the locked protocol")
        return
    temporary = artifact / f".CONFIG_LOCK.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit_trace(arm: str, trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError(f"{arm} produced no trace")
    if arm == "vanilla":
        return {"technical_pass": True, "all_visual_features_bit_identical": True}
    feature_equal = all(step["feature_equal"] for step in trace)
    if arm in {"attn_uniform", "attn_semantic"}:
        hook_pass = all(
            step.get("negative_truncated", False)
            or step["attention_mask"]["hook_calls"] == EXPECTED_HOOK_CALLS
            for step in trace
        )
        token_count_pass = arm != "attn_uniform" or all(
            step["num_tokens"] == 64 for step in trace
        )
        technical_pass = feature_equal and hook_pass and token_count_pass
        audit = {
            "all_visual_features_bit_identical": feature_equal,
            "all_attention_hook_audits_pass": hook_pass,
            "uniform_token_count_audit_pass": token_count_pass,
            "technical_pass": technical_pass,
        }
    else:
        uniform_hooks = all(
            step.get("uniform_negative_truncated", False)
            or step["uniform_attention_mask"]["hook_calls"] == EXPECTED_HOOK_CALLS
            for step in trace
        )
        semantic_hooks = all(
            step.get("semantic_negative_truncated", False)
            or step["semantic_attention_mask"]["hook_calls"] == EXPECTED_HOOK_CALLS
            for step in trace
        )
        diagnostic_spaces = ("orthogonal_full_vocab", "orthogonal_action_vocab")
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
        finite = all(
            np.isfinite(
                [step[space][key] for key in scalar_diagnostic_keys]
            ).all()
            for step in trace
            for space in diagnostic_spaces
        )
        orthogonal = all(
            step[space]["ortho_relative_error_max"] <= ORTHOGONAL_ERROR_LIMIT
            for step in trace
            for space in diagnostic_spaces
        )
        technical_pass = (
            feature_equal and uniform_hooks and semantic_hooks and finite and orthogonal
        )
        audit = {
            "all_visual_features_bit_identical": feature_equal,
            "all_uniform_hook_audits_pass": uniform_hooks,
            "all_semantic_hook_audits_pass": semantic_hooks,
            "all_diagnostics_finite": finite,
            "all_orthogonal_error_audits_pass": orthogonal,
            "technical_pass": technical_pass,
        }
    if not audit["technical_pass"]:
        raise RuntimeError(f"Technical audit failed for {arm}: {audit}")
    return audit


def refuse_unsafe_resume(task_root: Path) -> bool:
    manifest = task_root / "pairing_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get("all_four_arm_exact_pairing") and len(data.get("pairs", [])) == 50:
            print(json.dumps({"task": task_root.name, "already_complete": True}), flush=True)
            return True
        raise RuntimeError(f"Invalid existing pairing manifest: {manifest}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    write_config(artifact)
    task_root = artifact / "episodes" / args.task
    if refuse_unsafe_resume(task_root):
        return
    task_root.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**policy_config))
    pairs = []
    completed_seeds = set()
    for seed in range(50):
        summary_paths = {
            arm: task_root / arm / f"episode_{seed:03d}_summary.json" for arm in ARMS
        }
        if all(p.exists() for p in summary_paths.values()):
            ref = json.loads(summary_paths["vanilla"].read_text())
            pairs.append(
                {
                    "seed": seed,
                    "canonical_snapshot_sha256": ref["canonical_snapshot_sha256"],
                    "initial_state_sha256": ref["initial_state_sha256"],
                    "initial_rgb_sha256": ref["initial_rgb_sha256"],
                    "exact_four_arm_pairing": True,
                }
            )
            completed_seeds.add(seed)
    if completed_seeds:
        print(
            json.dumps(
                {"task": args.task, "resume_skipping_seeds": sorted(completed_seeds)}
            ),
            flush=True,
        )

    for seed in range(50):
        if seed in completed_seeds:
            continue
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        summaries = {}
        initial_hashes = {}
        instructions = {}

        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial_hashes[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            instructions[arm] = instruction
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(
                env, policy, instruction, obs
            )
            if arm == "orthogonal_dual":
                write_orthogonal_arrays(arrays_path, policy._episode_logits, actions)
            else:
                write_standard_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            audit = audit_trace(arm, trace)
            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": args.task,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": jitter,
                "first_step_residual_norm": (
                    0.0 if arm == "vanilla" else float(trace[0]["residual_norm"])
                ),
                "lambda": 0.0 if arm == "vanilla" else 0.5,
                "lambda_geom": 0.5 if arm == "orthogonal_dual" else None,
                "lambda_sem": 0.35 if arm == "orthogonal_dual" else None,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
                **audit,
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            summaries[arm] = summary
            print(
                json.dumps(
                    {
                        "task": args.task,
                        "seed": seed,
                        "arm": arm,
                        "success": summary["success"],
                        "steps": steps,
                        "technical_pass": audit["technical_pass"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError(f"Four-arm initial state/RGB mismatch: {args.task} {seed}")
        if len(set(instructions.values())) != 1:
            raise RuntimeError(f"Four-arm instruction mismatch: {args.task} {seed}")
        if len({item["canonical_snapshot_sha256"] for item in summaries.values()}) != 1:
            raise RuntimeError(f"Four-arm canonical snapshot mismatch: {args.task} {seed}")
        pairs.append(
            {
                "seed": seed,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": next(iter(initial_hashes.values()))[0],
                "initial_rgb_sha256": next(iter(initial_hashes.values()))[1],
                "exact_four_arm_pairing": True,
            }
        )
        (task_root / "task_progress.json").write_text(
            json.dumps(
                {"task": args.task, "completed_seeds": seed + 1, "last_seed": seed},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    manifest = {
        "protocol_id": PROTOCOL,
        "task": args.task,
        "seeds": list(range(50)),
        "arms": list(ARMS),
        "pairs": pairs,
        "all_four_arm_exact_pairing": True,
    }
    (task_root / "pairing_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": 50}), flush=True)


if __name__ == "__main__":
    main()
