"""Two new attention-mask arms for the five-arm SIMPLER Distractor pilot."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.attention_mask_policy import AttentionMaskEntityCDInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    get_image_from_maniskill2_obs_dict,
    jsonable,
    make_environment,
    restore_snapshot,
    run_episode,
    snapshot_sha,
    write_logits,
)


PROTOCOL = "CAUSAL_ATTENTION_MASKING_CD_PHASE0_V1"
ARMS = ("attn_random_cd", "attn_semantic_cd")


def build_policy(base, selection_mode):
    policy = copy.copy(base)
    policy.__class__ = AttentionMaskEntityCDInference
    policy.alpha = 0.5
    policy.lambd = 0.5
    policy.kmeans_K = 8
    policy.kmeans_seed = 0
    policy.selection_mode = selection_mode
    policy.attention_layer_start = 16
    policy.attention_layer_end = 32
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--legacy-artifact", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    seeds = [int(value) for value in args.seeds.split(",") if value]
    artifact = args.artifact.resolve()
    legacy_artifact = args.legacy_artifact.resolve()
    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)

    lock = {
        "experiment": PROTOCOL,
        "benchmark": "SIMPLER with distractors",
        "tasks": ["google_robot_pick_coke_can", "google_robot_open_drawer", "google_robot_close_drawer"],
        "seeds": list(range(10)),
        "new_arms": list(ARMS),
        "reused_arms": ["vanilla", "random_cd", "semantic_cd"],
        "legacy_artifact": str(legacy_artifact),
        "lambda": 0.5,
        "selector": "entity_set KMeans K=8, seed=0, n_init=10",
        "random_control": "size-preserving random permutation of KMeans membership",
        "attention_layers": [16, 32],
        "visual_key_offset": 1,
        "gates": {"random_inertness_pp": 5.0, "semantic_random_gain_pp": 8.0, "open_drawer_no_harm": True},
    }
    lock_path = artifact / "CONFIG_LOCK.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise RuntimeError("CONFIG_LOCK.json differs from preregistration")
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**config)
    policies = {
        "attn_random_cd": build_policy(base, "random_matched"),
        "attn_semantic_cd": build_policy(base, "semantic"),
    }

    pairs = []
    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        summaries = {}
        initial_hashes = {}
        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            logits_path = arm_dir / f"episode_{seed:03d}_logits.npz"
            if summary_path.exists() and logits_path.exists():
                summaries[arm] = json.loads(summary_path.read_text())
                print(json.dumps({"skip": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial_hashes[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, timestep, failure_reason = run_episode(env, policy, instruction, obs)
            write_logits(logits_path, policy._episode_logits)
            trace = jsonable(policy._episode_trace)
            if not trace or not all(step["feature_equal"] for step in trace):
                raise RuntimeError("Feature equality audit failed")
            summary = {
                "protocol_id": PROTOCOL,
                "benchmark": "SIMPLER with distractors",
                "environment_id": environment_id,
                "task": args.task,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": failure_reason,
                "control_steps": timestep,
                "lambda": 0.5,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "logits_file": logits_path.name,
                "logits_dtype": "float16",
                "logits_scope": "OpenVLA action vocabulary [7,256]",
                "selector_trace": trace,
                "selected_entities": trace[0]["selected_entities"],
                "mean_residual_norm": float(np.mean([step["residual_norm"] for step in trace])),
                "mean_num_tokens": float(np.mean([step["num_tokens"] for step in trace])),
                "all_visual_features_bit_identical": True,
                "result": jsonable(result),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            summaries[arm] = summary
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": summary["success"], "steps": timestep}), flush=True)

        for arm in ARMS:
            if arm not in initial_hashes:
                initial_hashes[arm] = (
                    summaries[arm]["initial_state_sha256"],
                    summaries[arm]["initial_rgb_sha256"],
                )
        old_summary = json.loads(
            (legacy_artifact / "episodes" / args.task / "vanilla" / f"episode_{seed:03d}_summary.json").read_text()
        )
        all_state = [value[0] for value in initial_hashes.values()] + [old_summary["initial_state_sha256"]]
        all_rgb = [value[1] for value in initial_hashes.values()] + [old_summary["initial_rgb_sha256"]]
        all_canonical = [summary["canonical_snapshot_sha256"] for summary in summaries.values()] + [old_summary["canonical_snapshot_sha256"]]
        if len(set(all_state)) != 1 or len(set(all_rgb)) != 1 or len(set(all_canonical)) != 1:
            raise RuntimeError(f"Five-arm snapshot pairing mismatch for {args.task} seed {seed}")
        pairs.append({"seed": seed, "canonical_snapshot_sha256": canonical, "exact_five_arm_pairing": True})

    (task_root / "pairing_manifest.json").write_text(json.dumps({
        "task": args.task, "seeds": seeds, "new_arms": list(ARMS), "pairs": pairs,
        "all_five_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
