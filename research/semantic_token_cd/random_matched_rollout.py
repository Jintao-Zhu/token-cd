"""Matched-random attention arm for semantic-specificity validation."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from research.semantic_token_cd.attention_mask_policy import AttentionMaskEntityCDInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)


PROTOCOL = "SEMANTIC_SPECIFICITY_RANDOM_MATCHED_50_V1"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
)
ARM = "attn_random_matched"


def build_policy(base):
    policy = copy.copy(base)
    policy.__class__ = AttentionMaskEntityCDInference
    policy.alpha = 0.5
    policy.lambd = 0.5
    policy.kmeans_K = 8
    policy.kmeans_seed = 0
    policy.selection_mode = "random_matched"
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


def write_config(artifact: Path, reference_artifact: Path):
    config = {
        "experiment": PROTOCOL,
        "tasks": list(TASKS),
        "seeds": list(range(50)),
        "technical_exclusions": {
            "google_robot_open_drawer": [4],
            "google_robot_close_drawer": [4],
            "reason": "drawer seed 4 canonical snapshot is not reproducible across environment processes; excluded before Random-Matched outcome was observed",
        },
        "new_arm": ARM,
        "reference_artifact": str(reference_artifact),
        "reference_arms": ["vanilla", "attn_uniform", "attn_semantic"],
        "lambda": 0.5,
        "attention_layers": [16, 32],
        "attention_mask_value_requested": -10000.0,
        "attention_mask_value_bfloat16_effective": -9984.0,
        "selector": "entity_set KMeans K=8 seed=0 n_init=10",
        "random_control": "permute token-to-KMeans-group membership; preserve selected group IDs, group count, every selected group size, and total N",
        "gates": {
            "S1_semantic_minus_random_matched_pp": 5.0,
            "S2_semantic_task_wins": 2,
            "S3_semantic_rescue_harm_gt_random": True,
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("CONFIG_LOCK.json differs from preregistration")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(50)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    reference = args.reference_artifact.resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if any(seed not in range(50) for seed in seeds):
        raise ValueError("Seeds must be in 0..49")
    if args.task in {"google_robot_open_drawer", "google_robot_close_drawer"}:
        seeds = [seed for seed in seeds if seed != 4]
    artifact.mkdir(parents=True, exist_ok=True)
    write_config(artifact, reference)
    reference_manifest = json.loads(
        (reference / "episodes" / args.task / "pairing_manifest.json").read_text()
    )
    if not reference_manifest["all_three_arm_exact_pairing"]:
        raise RuntimeError("Reference three-arm pairing audit did not pass")
    reference_pairs = {int(pair["seed"]): pair for pair in reference_manifest["pairs"]}

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config))
    manifests = []

    for seed in seeds:
        reference_pair = reference_pairs[seed]
        snapshot = capture_snapshot(env, seed)  # Reset 1 of 4.
        canonical = snapshot_sha(snapshot)
        if canonical != reference_pair["canonical_snapshot_sha256"]:
            raise RuntimeError(f"Reference snapshot mismatch for {args.task} seed {seed}")

        arm_dir = artifact / "episodes" / args.task / ARM
        arm_dir.mkdir(parents=True, exist_ok=True)
        summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
        arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary["canonical_snapshot_sha256"] != canonical:
                raise RuntimeError(f"Existing episode snapshot mismatch: {summary_path}")
            env.reset(seed=seed)  # Reset 2; matches the completed arm.
            print(json.dumps({"skip": args.task, "seed": seed, "arm": ARM}), flush=True)
        else:
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)  # Reset 2.
            instruction = env.unwrapped.get_language_instruction()
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(
                env, policy, instruction, obs
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            if not trace or not all(step["feature_equal"] for step in trace):
                raise RuntimeError("Feature equality audit failed")
            for step in trace:
                attention = step["attention_mask"]
                if (
                    step["selection_mode"] != "random_matched"
                    or step["num_tokens"] != len(step["reference_semantic_token_ids"])
                    or attention["hook_calls"] != 112
                    or attention["layer_indices"] != list(range(16, 32))
                    or float(attention["mask_value"]) != -10000.0
                ):
                    raise RuntimeError("Matched-random intervention audit failed")
            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": args.task,
                "seed": seed,
                "episode_id": seed,
                "arm": ARM,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": jitter,
                "first_step_residual_norm": float(trace[0]["residual_norm"]),
                "lambda": 0.5,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "all_visual_features_bit_identical": True,
                "result": jsonable(result),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"task": args.task, "seed": seed, "arm": ARM, "success": summary["success"], "steps": steps}), flush=True)

        # The reference protocol consumed four resets per seed. Preserve that
        # cadence so drawer scene-instance selection matches every next seed.
        env.reset(seed=seed)  # Reset 3.
        env.reset(seed=seed)  # Reset 4.
        manifests.append({
            "seed": seed,
            "canonical_snapshot_sha256": canonical,
            "reference_pairing_manifest": str(reference / "episodes" / args.task / "pairing_manifest.json"),
            "exact_four_arm_pairing": True,
            "summary": str(summary_path),
        })

    task_root = artifact / "episodes" / args.task
    (task_root / "pairing_manifest.json").write_text(json.dumps({
        "task": args.task,
        "seeds": seeds,
        "new_arm": ARM,
        "pairs": manifests,
        "all_four_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
