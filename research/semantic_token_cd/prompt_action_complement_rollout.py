"""Closed-loop rollout worker for Prompt/Action complement SHR v1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_action_complement_protocol import (
    ACTION_LAYERS, ARTIFACT, CANONICAL, LAMBDA, NEW_ARMS, PCD_SOURCE,
    PROMPT_LAYERS, PROTOCOL, TASKS, atomic_json,
)
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.xswap_rollout import make_environment


def parse_seeds(spec: str) -> list[int]:
    result = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); result.extend(range(lo, hi + 1))
        elif part.strip():
            result.append(int(part))
    return sorted(set(result))


def build_policy(base, task: str, arm: str):
    policy = copy.copy(base); policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = PROMPT_LAYERS
    policy.selection_count = None
    policy.selection_top_p = None
    policy.selector_instruction = None
    policy.selector_contrast_instruction = None
    policy.selector_difference_eta = None
    policy.complement_arm = arm
    policy.action_attention_layers = ACTION_LAYERS
    policy.save_prompt_attention = True
    return policy


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_loop(env, policy, instruction, observation, video_path: Path):
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import flatten_action
    from utils import convert_numpy_or_torch_to_python, summarize
    image = get_image_from_maniskill2_obs_dict(env, observation)
    infos, actions, frames = [], [], []
    predicted = truncated = False
    control = 0
    while not (predicted or truncated) and control < 140:
        frames.append(np.asarray(image, dtype=np.uint8))
        _raw, action_group, _meta = policy.step(
            image, None, instruction, proprio=observation["agent"]["eef_pos"]
        )
        if not isinstance(action_group, list): action_group = [action_group]
        for action in action_group:
            vector = flatten_action(action)
            if vector.shape != (7,) or not np.isfinite(vector).all():
                raise FloatingPointError("invalid action")
            actions.append(vector.copy())
            observation, _reward, _success, truncated, info = env.step(vector)
            image = get_image_from_maniskill2_obs_dict(env, observation)
            control += 1
            infos.append(convert_numpy_or_torch_to_python(info))
            predicted = bool(action["terminate_episode"][0] > 0)
            if predicted and not env.unwrapped.is_final_subtask():
                predicted = False; env.advance_to_next_subtask()
    frames.append(np.asarray(image, dtype=np.uint8))
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_path, frames, fps=10, codec="libx264", quality=7, macro_block_size=None)
    result = summarize(infos)
    result["failure_reason"] = None if result.get("success") else (
        "time_limit" if truncated else "policy_terminated"
    )
    return result, np.asarray(actions, dtype=np.float32), infos


def write_arrays(path: Path, records: list[dict], actions: np.ndarray) -> None:
    payload = {"executed_actions": actions}
    keys = (
        "positive", "negative", "selected_mask", "reference_shr_mask",
        "prompt_attention", "action_attention", "action_attention_per_layer",
        "action_attention_per_dimension", "action_attention_l11",
        "action_attention_full_layers", "core_mask", "supplement_mask",
        "per_token_perturbation_norm",
    )
    for key in keys:
        if records and all(key in row for row in records):
            payload[key] = np.stack([row[key] for row in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", required=True, type=int, choices=(2, 3))
    parser.add_argument("--arm", required=True, choices=NEW_ARMS)
    parser.add_argument("--worker-id", default="manual")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    env, env_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    policy_sha = file_sha(Path(__file__).with_name("prompt_attn_shr_policy.py"))
    for seed in parse_seeds(args.seeds):
        out = ARTIFACT / "closed_loop/episodes" / args.task / args.arm
        summary_path = out / f"episode_{seed:03d}_summary.json"
        arrays_path = out / f"episode_{seed:03d}_arrays.npz"
        video_path = ARTIFACT / "closed_loop/videos" / args.task / args.arm / f"episode_{seed:03d}.mp4"
        if summary_path.exists() and arrays_path.exists() and video_path.exists(): continue
        snapshot_path = CANONICAL / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
        with snapshot_path.open("rb") as handle: snapshot = pickle.load(handle)
        observation, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        policy = build_policy(base, args.task, args.arm)
        policy._episode_trace = []; policy._episode_logits = []
        policy._episode_seed = seed; policy._selector_step = 0
        policy.reset(instruction, seed=seed)
        start = time.monotonic()
        result, actions, infos = run_loop(env, policy, instruction, observation, video_path)
        runtime = time.monotonic() - start
        trace = policy._episode_trace
        checks = {
            "all_prompt_l11": all(row.get("attention_layers") == [11] for row in trace),
            "all_action_layers_16_31": all(row.get("action_attention_layers") == list(range(16, 32)) for row in trace),
            "all_action_dims_0_5": all(row.get("action_attention_dimensions") == list(range(6)) for row in trace),
            "all_action_attention_recorded": all(
                "teacher_forced_vs_autoregressive_changed_dims" in row for row in trace
            ),
            "all_exact_m": all(row.get("coverage_exact") is True for row in trace),
            "all_core_exact": all(row.get("core_exact") is True for row in trace),
            "all_pool_exact": all(row.get("candidate_pool_membership_exact") is True for row in trace),
            "all_feature_equal": all(row.get("feature_equal") is True for row in trace),
            "all_non_target_equal": all(row.get("non_target_bit_identical") is True for row in trace),
            "all_guided_prefix": all(row.get("guided_prefix") is True for row in trace),
        }
        checks["technical_pass"] = bool(trace) and all(checks.values())
        if not checks["technical_pass"]: raise RuntimeError(f"audit failure: {checks}")
        summary = {
            "protocol": PROTOCOL, "task": args.task, "seed": seed, "arm": args.arm,
            "instruction": instruction, "success": bool(result.get("success", False)),
            "result": jsonable(result), "control_steps": len(infos), "runtime_seconds": runtime,
            "worker_id": args.worker_id, "environment_id": env_id,
            "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
            "canonical_snapshot_sha256": snapshot_sha(snapshot), "policy_code_sha256": policy_sha,
            "mean_m": float(np.mean([row["m_t"] for row in trace])),
            "mean_r": float(np.mean([row["supplement_count_r"] for row in trace])),
            "mean_replacements": float(np.mean([row["actual_replacements_vs_original"] for row in trace])),
            "mean_feature_perturbation_norm": float(np.mean([row["feature_perturbation_norm"] for row in trace])),
            "mean_centered_residual_norm": float(np.mean([row["centered_logit_residual_norm"] for row in trace])),
            "max_teacher_clean_logits_abs_diff": float(max(row["teacher_forced_clean_action_logits_max_abs_diff"] for row in trace)),
            "selector_trace": jsonable(trace), "video_path": str(video_path.relative_to(ARTIFACT)),
            "video_sha256": file_sha(video_path), **checks,
        }
        write_arrays(arrays_path, policy._episode_logits, actions)
        atomic_json(summary_path, summary)
        print(json.dumps({"task": args.task, "seed": seed, "arm": args.arm,
                          "success": summary["success"], "seconds": round(runtime, 1)}), flush=True)
    env.close()


if __name__ == "__main__": main()
