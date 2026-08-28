"""Six-arm paired rollout for Spatial Grid vs Random Attention-CD."""

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
    flatten_action,
    get_image_from_maniskill2_obs_dict,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_policy import SpatialGridAttentionCDInference


PROTOCOL = "SPATIAL_GRID_VS_RANDOM_ATTENTION_CD_V1"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
)

# Widowx tasks for the 9-task benchmark expansion. This file's own six-arm
# protocol (TASKS) is google_robot-only, but make_environment is shared by
# uniform_semantic_rollout and orthogonal_dual_experiment which run all 9.
WIDOWX_TASKS = (
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
ARMS = (
    "vanilla",
    "attn_rand_64",
    "attn_grid_strided",
    "attn_grid_checkerboard",
    "attn_sem_hard",
    "attn_sem_sparse",
)
MODES = {
    "attn_rand_64": "random_64",
    "attn_grid_strided": "grid_strided",
    "attn_grid_checkerboard": "grid_checkerboard",
    "attn_sem_hard": "semantic_hard",
    "attn_sem_sparse": "semantic_sparse",
}


def make_environment(task: str):
    import gymnasium as gym
    import simpler_env

    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd",
            prepackaged_config=True,
            distractor_config="less",
        ), "GraspSingleOpenedCokeCanDistractorInScene-v0"
    if task in TASKS or task in WIDOWX_TASKS:
        return simpler_env.make(task), simpler_env.ENVIRONMENT_MAP[task][0]
    raise ValueError(f"Task is not preregistered: {task}")


def build_policies(base):
    vanilla = copy.copy(base)
    vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace = []
    vanilla._episode_logits = []
    policies = {"vanilla": vanilla}
    for arm, mode in MODES.items():
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
        policies[arm] = policy
    return policies


def run_episode(env, policy, instruction, obs):
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize

    predicted_terminated = truncated = False
    timestep = 0
    step_infos = []
    executed_actions = []
    image = get_image_from_maniskill2_obs_dict(env, obs)
    while not (predicted_terminated or truncated):
        _raw_action, actions, _aux = policy.step(
            image, None, instruction, proprio=obs["agent"]["eef_pos"]
        )
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            executed = flatten_action(action)
            if executed.shape != (7,) or not np.isfinite(executed).all():
                raise FloatingPointError(f"Invalid executed action: {executed}")
            executed_actions.append(executed.copy())
            obs, _reward, _success, truncated, info = env.step(executed)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            timestep += 1
            step_infos.append(convert_numpy_or_torch_to_python(info))
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not env.unwrapped.is_final_subtask():
                predicted_terminated = False
                env.advance_to_next_subtask()
            instruction = env.unwrapped.get_language_instruction()
    result = summarize(step_infos)
    result.update(stat_first(step_infos))
    result.update(stat_final(step_infos))
    failure_reason = None if result["success"] else (
        "environment_time_limit" if truncated else "policy_terminated_without_success"
    )
    actions_array = np.asarray(executed_actions, dtype=np.float32)
    jitter = float(np.linalg.norm(np.diff(actions_array, axis=0), axis=1).mean()) if len(actions_array) > 1 else 0.0
    return result, timestep, failure_reason, actions_array, jitter


def write_arrays(path: Path, logits: list[dict], actions: np.ndarray):
    positive = np.stack([record["positive"] for record in logits])
    payload = {"positive_logits": positive, "executed_actions": actions}
    if logits and "negative" in logits[0]:
        payload["negative_logits"] = np.stack([record["negative"] for record in logits])
    np.savez_compressed(path, **payload)


def config_lock(artifact: Path):
    lock = {
        "experiment": PROTOCOL,
        "benchmark": "SIMPLER Google Robot; coke uses less-distractor environment",
        "tasks": list(TASKS),
        "seeds": list(range(30)),
        "arms": list(ARMS),
        "lambda": 0.5,
        "attention_layers": [16, 32],
        "attention_mask_value_requested": -10000.0,
        "attention_mask_value_bfloat16_effective": float(torch.tensor(-1e4, dtype=torch.bfloat16).item()),
        "rand_64": "64 tokens without replacement; deterministic by episode seed and control step",
        "grid_strided": "(row,col) both even on 16x16 grid; 64 tokens",
        "grid_checkerboard": "(row+col)%2==0 on 16x16 grid; 128 tokens",
        "semantic_hard": "entity_set KMeans K=8 seed=0 n_init=10; all selected tokens",
        "semantic_sparse": "floor(N_sem/2), min 1, without replacement; deterministic by episode seed and control step",
        "gates": {"G1_grid_minus_random_pp": 5.0, "G2_grid_minus_vanilla_pp": 20.0, "G3_task_wins": 4, "G4_sparse_gt_hard": True},
        "note": "Previous 50.0% Attn-Random used semantic-size-matched dynamic N, not Rand-64, and is not reused.",
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError("CONFIG_LOCK.json differs from preregistration")
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(30)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    seeds = [int(value) for value in args.seeds.split(",") if value]
    if any(seed not in range(30) for seed in seeds):
        raise ValueError("Seeds must be in preregistered interval 0..29")
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    config_lock(artifact)
    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**policy_config))
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
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
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
            result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            if arm != "vanilla":
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
                "first_step_residual_norm": 0.0 if arm == "vanilla" else float(trace[0]["residual_norm"]),
                "lambda": 0.0 if arm == "vanilla" else 0.5,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "all_visual_features_bit_identical": arm == "vanilla" or all(step["feature_equal"] for step in trace),
                "result": jsonable(result),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            summaries[arm] = summary
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": summary["success"], "steps": steps, "jitter": jitter}), flush=True)

        for arm in ARMS:
            if arm not in initial_hashes:
                initial_hashes[arm] = (summaries[arm]["initial_state_sha256"], summaries[arm]["initial_rgb_sha256"])
        if len({value[0] for value in initial_hashes.values()}) != 1 or len({value[1] for value in initial_hashes.values()}) != 1:
            raise RuntimeError(f"Six-arm initial-state mismatch for {args.task} seed {seed}")
        if len({summary["canonical_snapshot_sha256"] for summary in summaries.values()}) != 1:
            raise RuntimeError(f"Six-arm snapshot mismatch for {args.task} seed {seed}")
        pairs.append({"seed": seed, "canonical_snapshot_sha256": canonical, "exact_six_arm_pairing": True})

    (task_root / "pairing_manifest.json").write_text(json.dumps({
        "task": args.task, "seeds": seeds, "arms": list(ARMS), "pairs": pairs,
        "all_six_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
