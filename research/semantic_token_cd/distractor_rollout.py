"""Three-arm paired rollout for SIMPLER Distractor Semantic Entity-CD V2."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_ROOT = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e")
PCD_SOURCE = PCD_ROOT / "source"
MEAN_PATH = (
    REPO_ROOT
    / "artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814"
    / "position_conditioned_visual_mean.pt"
)
PROTOCOL = "SIMPLER_DISTRACTOR_SEMANTIC_ENTITY_CD_PHASE0_V2"
ARMS = ("vanilla", "semantic_cd", "random_cd")

for path in (REPO_ROOT / "task1/shim_site", REPO_ROOT, PCD_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parallel_inference import get_image_from_maniskill2_obs_dict  # noqa: E402
from properties import get_policy_config  # noqa: E402
from simpler_env.policies.openvla.openvla_model import OpenVLAInference  # noqa: E402
from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize  # noqa: E402

from research.semantic_token_cd.distractor_policy import (  # noqa: E402
    AuditedEntityCDInference,
    AuditedVanillaInference,
)
from research.semantic_token_cd.rollout_pilot import (  # noqa: E402
    array_sha256,
    capture_snapshot,
    clone,
    flatten_action,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)


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
    if task in {"google_robot_open_drawer", "google_robot_close_drawer"}:
        return simpler_env.make(task), simpler_env.ENVIRONMENT_MAP[task][0]
    raise ValueError(f"Task is not preregistered: {task}")


def build_policies(vanilla, mean):
    policies = {}
    clean = copy.copy(vanilla)
    clean.__class__ = AuditedVanillaInference
    clean._episode_trace = []
    clean._episode_logits = []
    policies["vanilla"] = clean
    for arm, mode in (("semantic_cd", "semantic"), ("random_cd", "random_matched")):
        policy = copy.copy(vanilla)
        policy.__class__ = AuditedEntityCDInference
        policy.alpha = 0.5
        policy.lambd = 0.5
        policy.replacement_mean = mean
        policy.kmeans_K = 8
        policy.kmeans_seed = 0
        policy.selection_mode = mode
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
    predicted_terminated = truncated = False
    timestep = 0
    step_infos = []
    image = get_image_from_maniskill2_obs_dict(env, obs)
    while not (predicted_terminated or truncated):
        _raw_action, actions, _aux = policy.step(
            image, None, instruction, proprio=obs["agent"]["eef_pos"]
        )
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            executed = flatten_action(action)
            if not np.isfinite(executed).all():
                raise FloatingPointError("Non-finite action")
            obs, _reward, _success, truncated, info = env.step(executed)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            timestep += 1
            step_infos.append(convert_numpy_or_torch_to_python(info))
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not env.unwrapped.is_final_subtask():
                predicted_terminated = False
                env.advance_to_next_subtask()
            next_instruction = env.unwrapped.get_language_instruction()
            if next_instruction != instruction:
                instruction = next_instruction
    result = summarize(step_infos)
    result.update(stat_first(step_infos))
    result.update(stat_final(step_infos))
    if result["success"]:
        reason = None
    elif truncated:
        reason = "environment_time_limit"
    else:
        reason = "policy_terminated_without_success"
    return result, timestep, reason


def write_logits(path: Path, records: list[dict]):
    positive = np.stack([record["positive"] for record in records])
    payload = {"positive_logits": positive}
    if records and "negative" in records[0]:
        payload["negative_logits"] = np.stack([record["negative"] for record in records])
    np.savez_compressed(path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--mean-path", type=Path, default=MEAN_PATH)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seeds = [int(value) for value in args.seeds.split(",") if value]
    artifact = args.artifact.resolve()
    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    (artifact / "initial_states" / args.task).mkdir(parents=True, exist_ok=True)

    lock_path = artifact / "CONFIG_LOCK.json"
    config = {
        "experiment": PROTOCOL,
        "benchmark": "SIMPLER with distractors",
        "tasks": [
            "google_robot_pick_coke_can",
            "google_robot_open_drawer",
            "google_robot_close_drawer",
        ],
        "episodes_per_task": 50,
        "arms": list(ARMS),
        "lambda": 0.5,
        "selector": "entity_set KMeans K=8, seed=0, n_init=10",
        "random_control": "size-preserving random permutation of KMeans membership",
        "replacement": "position-conditioned visual mean",
        "gates": {"A_delta_semantic_random": 0.03, "B_rescue_harm": 1.5, "C_task_wins": 2},
    }
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != config:
            raise RuntimeError("CONFIG_LOCK.json differs from preregistration")
    else:
        temporary_lock = lock_path.with_name(f".{lock_path.name}.{os.getpid()}.tmp")
        temporary_lock.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_lock, lock_path)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base_policy = OpenVLAInference(**policy_config)
    mean = torch.load(args.mean_path.resolve(), map_location="cpu", weights_only=True)["mean"]
    policies = build_policies(base_policy, mean)

    manifests = []
    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        initial = {}
        summaries = {}
        initial_image_path = artifact / "initial_states" / args.task / f"episode_{seed:03d}.png"
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
            initial[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            if not initial_image_path.exists():
                Image.fromarray(get_image_from_maniskill2_obs_dict(env, obs)).save(initial_image_path)
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, timestep, failure_reason = run_episode(env, policy, instruction, obs)
            write_logits(logits_path, policy._episode_logits)
            trace = jsonable(policy._episode_trace)
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
                "lambda": 0.0 if arm == "vanilla" else 0.5,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "logits_file": logits_path.name,
                "logits_dtype": "float16",
                "logits_scope": "OpenVLA action vocabulary [7,256]",
                "selector_trace": trace,
                "result": jsonable(result),
            }
            if arm != "vanilla":
                summary["selected_entities"] = trace[0]["selected_entities"] if trace else []
                summary["mean_residual_norm"] = float(np.mean([step["residual_norm"] for step in trace]))
                summary["mean_num_tokens"] = float(np.mean([step["num_tokens"] for step in trace]))
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            summaries[arm] = summary
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": summary["success"],
                "steps": timestep,
            }), flush=True)

        state_hashes = {
            arm: initial[arm][0] if arm in initial else summaries[arm]["initial_state_sha256"]
            for arm in ARMS
        }
        rgb_hashes = {
            arm: initial[arm][1] if arm in initial else summaries[arm]["initial_rgb_sha256"]
            for arm in ARMS
        }
        canonical_hashes = {arm: summaries[arm]["canonical_snapshot_sha256"] for arm in ARMS}
        if len(set(state_hashes.values())) != 1 or len(set(rgb_hashes.values())) != 1:
            raise RuntimeError(f"Cross-arm pairing mismatch for {args.task} seed {seed}")
        if len(set(canonical_hashes.values())) != 1:
            raise RuntimeError(f"Snapshot mismatch for {args.task} seed {seed}")
        manifests.append({
            "seed": seed,
            "canonical_snapshot_sha256": canonical,
            "initial_state_sha256": state_hashes["vanilla"],
            "initial_rgb_sha256": rgb_hashes["vanilla"],
            "exact_pairing": True,
        })

    (task_root / "pairing_manifest.json").write_text(json.dumps({
        "task": args.task,
        "seeds": seeds,
        "arms": list(ARMS),
        "pairs": manifests,
        "all_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
