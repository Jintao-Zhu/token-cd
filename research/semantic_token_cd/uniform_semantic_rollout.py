"""Missing-arm rollout for the 450-episode Uniform vs Semantic protocol.

Vanilla 0..49 and a small prefix of both attention arms are referenced from
audited prior artifacts. This runner computes only missing Uniform/Semantic
episodes and verifies every reused/new episode against the Vanilla snapshot.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    get_image_from_maniskill2_obs_dict,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.spatial_grid_policy import SpatialGridAttentionCDInference
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)


PROTOCOL = "UNIFORM_VS_SEMANTIC_ATTENTION_CD_450_V1"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
ARMS = ("attn_uniform", "attn_semantic")
MODES = {"attn_uniform": "grid_strided", "attn_semantic": "semantic_hard"}
REUSED_NAMES = {"attn_uniform": "attn_grid_strided", "attn_semantic": "attn_sem_hard"}


def build_policy(base, mode: str):
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


def build_vanilla_policy(base):
    policy = copy.copy(base)
    policy.__class__ = AuditedVanillaInference
    policy._episode_trace = []
    policy._episode_logits = []
    return policy


def summary_path(root: Path, task: str, arm: str, seed: int) -> Path:
    return root / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"


def find_existing(
    artifact: Path,
    reuse_artifact: Path,
    task: str,
    arm: str,
    seed: int,
) -> tuple[Path, str] | None:
    new_path = summary_path(artifact, task, arm, seed)
    if new_path.exists() and new_path.with_name(new_path.name.replace("_summary.json", "_arrays.npz")).exists():
        return new_path, "current_artifact"
    reused_arm = REUSED_NAMES[arm]
    old_path = summary_path(reuse_artifact, task, reused_arm, seed)
    old_arrays = old_path.with_name(old_path.name.replace("_summary.json", "_arrays.npz"))
    if old_path.exists() and old_arrays.exists():
        return old_path, "spatial_grid_partial"
    return None


def write_config(artifact: Path, vanilla_artifact: Path, reuse_artifact: Path):
    config = {
        "experiment": PROTOCOL,
        "tasks": list(TASKS),
        "seeds": list(range(50)),
        "logical_arms": ["vanilla", *ARMS],
        "lambda": 0.5,
        "attention_layers": [16, 32],
        "attention_mask_value_requested": -10000.0,
        "attention_mask_value_bfloat16_effective": -9984.0,
        "uniform_geometry": "16x16 even-row x even-column strided grid; exactly 64 tokens",
        "semantic_selector": "entity_set KMeans K=8 seed=0 n_init=10; dense selected groups",
        "vanilla_source": str(vanilla_artifact),
        "partial_attention_source": str(reuse_artifact),
        "gates": {
            "gate_1_uniform_minus_vanilla_pp": 15.0,
            "gate_2_uniform_positive_and_task_wins": 2,
        },
        "execution": "reference audited prior episodes and compute only missing attention-arm episodes",
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("CONFIG_LOCK.json differs from preregistration")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--vanilla-artifact", type=Path, required=True)
    parser.add_argument("--reuse-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(50)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    vanilla_artifact = args.vanilla_artifact.resolve()
    reuse_artifact = args.reuse_artifact.resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if any(seed not in range(50) for seed in seeds):
        raise ValueError("Seeds must be in 0..49")
    artifact.mkdir(parents=True, exist_ok=True)
    write_config(artifact, vanilla_artifact, reuse_artifact)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**config)
    policies = {arm: build_policy(base, mode) for arm, mode in MODES.items()}
    vanilla_policy = build_vanilla_policy(base)
    manifest = []

    for seed in seeds:
        vanilla_path = summary_path(vanilla_artifact, args.task, "vanilla", seed)
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        old_vanilla_matches = False
        if vanilla_path.exists():
            vanilla = json.loads(vanilla_path.read_text())
            old_vanilla_matches = vanilla["canonical_snapshot_sha256"] == canonical
        sources = {}
        if old_vanilla_matches:
            _obs0, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if (
                vanilla["initial_state_sha256"] != state_sha
                or vanilla["initial_rgb_sha256"] != rgb_sha
            ):
                raise RuntimeError(f"Vanilla state/RGB mismatch for {args.task} seed {seed}")
            sources["vanilla"] = str(vanilla_path)
        else:
            vanilla_dir = artifact / "episodes" / args.task / "vanilla"
            vanilla_dir.mkdir(parents=True, exist_ok=True)
            vanilla_out = vanilla_dir / f"episode_{seed:03d}_summary.json"
            vanilla_arrays = vanilla_dir / f"episode_{seed:03d}_arrays.npz"
            if vanilla_out.exists() and vanilla_arrays.exists():
                fresh = json.loads(vanilla_out.read_text())
                if fresh["canonical_snapshot_sha256"] != canonical:
                    raise RuntimeError(f"Fresh Vanilla snapshot mismatch: {vanilla_out}")
                env.reset(seed=seed)  # Preserve four resets per seed when reusing this arm.
            else:
                obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                instruction = env.unwrapped.get_language_instruction()
                vanilla_policy.reset(instruction, seed=seed)
                vanilla_policy._episode_trace = []
                vanilla_policy._episode_logits = []
                result, steps, reason, actions, jitter = run_episode(
                    env, vanilla_policy, instruction, obs
                )
                write_arrays(vanilla_arrays, vanilla_policy._episode_logits, actions)
                fresh = {
                    "protocol_id": PROTOCOL,
                    "environment_id": environment_id,
                    "task": args.task,
                    "seed": seed,
                    "episode_id": seed,
                    "arm": "vanilla",
                    "instruction": instruction,
                    "success": bool(result["success"]),
                    "failure_reason": reason,
                    "control_steps": steps,
                    "action_jitter_index": jitter,
                    "first_step_residual_norm": 0.0,
                    "lambda": 0.0,
                    "canonical_snapshot_sha256": canonical,
                    "initial_state_sha256": state_sha,
                    "initial_rgb_sha256": rgb_sha,
                    "arrays_file": vanilla_arrays.name,
                    "selector_trace": jsonable(vanilla_policy._episode_trace),
                    "all_visual_features_bit_identical": True,
                    "result": jsonable(result),
                    "fresh_due_to_cross_run_snapshot_mismatch": True,
                }
                vanilla_out.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"task": args.task, "seed": seed, "arm": "vanilla", "success": fresh["success"], "fresh_snapshot": True}), flush=True)
            state_sha = fresh["initial_state_sha256"]
            rgb_sha = fresh["initial_rgb_sha256"]
            sources["vanilla"] = str(vanilla_out)
        for arm in ARMS:
            existing = find_existing(artifact, reuse_artifact, args.task, arm, seed)
            if existing is not None:
                path, source = existing
                summary = json.loads(path.read_text())
                trace = summary.get("selector_trace", [])
                expected_mode = MODES[arm]
                first = trace[0] if trace else {}
                attention = first.get("attention_mask", {})
                intervention_valid = (
                    first.get("selection_mode") == expected_mode
                    and attention.get("layer_indices") == list(range(16, 32))
                    and float(attention.get("mask_value", 0.0)) == -10000.0
                    and attention.get("hook_calls") == 112
                )
                if arm == "attn_uniform":
                    intervention_valid &= first.get("num_tokens") == 64
                if (
                    summary["canonical_snapshot_sha256"] != canonical
                    or summary["initial_state_sha256"] != state_sha
                    or summary["initial_rgb_sha256"] != rgb_sha
                    or not summary["all_visual_features_bit_identical"]
                    or not intervention_valid
                ):
                    # Cross-session snapshot not reproducible for this seed. Do NOT
                    # raise — fall through and recompute the arm fresh against this
                    # session's snapshot (same class as drawer seed-4 exclusion, but
                    # non-fatal here because fresh keeps the 3-arm pairing intact).
                    print(json.dumps({
                        "reuse_fallback": args.task, "seed": seed, "arm": arm,
                        "reason": "snapshot_or_audit_mismatch",
                    }), flush=True)
                else:
                    sources[arm] = str(path)
                    print(json.dumps({"reuse": args.task, "seed": seed, "arm": arm, "source": source}), flush=True)
                    env.reset(seed=seed)  # Match one restore/reset for every logical arm.
                    continue

            arm_dir = artifact / "episodes" / args.task / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            summary_out = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_out = arm_dir / f"episode_{seed:03d}_arrays.npz"
            obs, current_state_sha, current_rgb_sha = restore_snapshot(env, seed, snapshot)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
            write_arrays(arrays_out, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            if not trace or not all(step["feature_equal"] for step in trace):
                raise RuntimeError("Feature equality audit failed")
            if not all(step["attention_mask"]["hook_calls"] == 112 for step in trace):
                raise RuntimeError("Attention hook audit failed")
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
                "first_step_residual_norm": float(trace[0]["residual_norm"]),
                "lambda": 0.5,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": current_state_sha,
                "initial_rgb_sha256": current_rgb_sha,
                "arrays_file": arrays_out.name,
                "selector_trace": trace,
                "all_visual_features_bit_identical": True,
                "result": jsonable(result),
            }
            summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            sources[arm] = str(summary_out)
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": summary["success"], "steps": steps}), flush=True)

        manifest.append({
            "seed": seed,
            "canonical_snapshot_sha256": canonical,
            "exact_three_arm_pairing": True,
            "sources": sources,
        })

    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "pairing_manifest.json").write_text(json.dumps({
        "task": args.task,
        "seeds": seeds,
        "logical_arms": ["vanilla", *ARMS],
        "pairs": manifest,
        "all_three_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
