#!/usr/bin/env python3
"""First-step same-seed audit: current C vs historical l11_matched."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.task_conditioned_contrast_protocol import (
    ATTENTION_LAYERS,
    CANONICAL,
    CONTROL_INSTRUCTIONS,
    KMEANS_K,
    KMEANS_SEED,
    LAMBDA0,
    PCD_SOURCE,
    TASKS,
)


HISTORICAL = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/prompt_attn_l11_token_count_v1")
CURRENT = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/task_conditioned_contrast_midsize_v1")
AUDIT_SEEDS = tuple(range(100, 105))


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def build_old_policy(base, task: str):
    from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common

    policy = copy.copy(base)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA0)
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = ATTENTION_LAYERS
    policy.selection_count = None
    policy.selection_top_p = None
    policy.selection_budget_schedule = None
    policy.selection_budget_scale = 1.0
    policy.selection_budget_label = None
    policy.selection_budget_source = "matched"
    policy.selection_budget_entities = None
    policy.save_prompt_attention = False
    return policy


def build_current_policy(base, task: str):
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
    from research.semantic_token_cd.task_conditioned_contrast_policy import TaskConditionedContrastInference

    policy = copy.copy(base)
    policy.__class__ = TaskConditionedContrastInference
    _init_common(policy, LAMBDA0)
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = ATTENTION_LAYERS
    policy.selection_count = None
    policy.selection_top_p = None
    policy.selection_budget_schedule = None
    policy.selection_budget_scale = 1.0
    policy.selection_budget_label = None
    policy.selection_budget_source = "matched"
    policy.selection_budget_entities = None
    policy.save_prompt_attention = False
    policy.arm = "C"
    policy.apc_beta = 0.1
    policy.control_instruction = CONTROL_INSTRUCTIONS[task]
    policy.task_key = task
    policy.save_condition_logits = True
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=CURRENT)
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seeds", default="100-104")
    parser.add_argument("--task", choices=TASKS, default=None)
    args = parser.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    repo = Path("/home/leju-suzhou/zjt_ws/token-cd")
    for path in (str(repo / "task1/shim_site"), str(repo), str(PCD_SOURCE)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import restore_snapshot, snapshot_sha
    from research.semantic_token_cd.spatial_grid_rollout import make_environment
    import research.semantic_token_cd.prompt_attn_shr_policy as old_policy_module
    import research.semantic_token_cd.st_shr_policy as st_policy_module

    captured = []

    def wrap_harmonic(original):
        def wrapped(features, region, *args, **kwargs):
            solved = original(features, region, *args, **kwargs)
            captured.append({
                "input_sha256": array_sha256(np.asarray(features)),
                "region": [int(x) for x in region],
                "output_sha256": array_sha256(np.asarray(solved)),
                "output_norm": float(np.linalg.norm(np.asarray(solved))),
            })
            return solved
        return wrapped

    original_old_harmonic = old_policy_module.harmonic_reconstruct
    original_st_harmonic = st_policy_module.harmonic_reconstruct
    old_policy_module.harmonic_reconstruct = wrap_harmonic(original_old_harmonic)
    st_policy_module.harmonic_reconstruct = wrap_harmonic(original_st_harmonic)

    tasks = [args.task] if args.task else list(TASKS)
    results = []
    for task in tasks:
        env, _environment_id = make_environment(task)
        checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
        config = get_policy_config("openvla", checkpoint, task, {}, False)
        base = OpenVLAInference(**config)
        old_policy = build_old_policy(base, task)
        current_policy = build_current_policy(base, task)
        for seed in seeds:
            historical_summary_path = args.historical / "episodes" / task / "l11_matched" / f"episode_{seed:03d}_summary.json"
            historical_arrays_path = args.historical / "episodes" / task / "l11_matched" / f"episode_{seed:03d}_arrays.npz"
            historical = json.loads(historical_summary_path.read_text())
            historical_arrays = np.load(historical_arrays_path)
            historical_first = historical["selector_trace"][0]
            snapshot_path = CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl"
            snapshot = pickle.load(snapshot_path.open("rb"))
            expected_snapshot = snapshot_sha(snapshot)
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            instruction = env.unwrapped.get_language_instruction()
            image = get_image_from_maniskill2_obs_dict(env, obs)

            captured.clear()
            old_policy.reset(instruction, seed=seed)
            old_policy._episode_trace = []
            old_policy._episode_logits = []
            _raw_old, _actions_old, old_meta = old_policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
            old_harmonic = captured[-1] if captured else None
            old_logits = old_policy._episode_logits[0]

            captured.clear()
            current_policy.reset(instruction, seed=seed)
            current_policy._episode_trace = []
            current_policy._episode_logits = []
            _raw_current, _actions_current, current_meta = current_policy.step(
                image, None, instruction, proprio=obs["agent"]["eef_pos"]
            )
            current_harmonic = captured[-1] if captured else None
            current_logits = current_policy._episode_logits[0]

            historical_positive_lp = torch.log_softmax(
                torch.from_numpy(historical_arrays["positive"][0]).float(), dim=-1
            ).numpy()
            historical_negative_lp = torch.log_softmax(
                torch.from_numpy(historical_arrays["negative"][0]).float(), dim=-1
            ).numpy()
            current_clean_lp = np.asarray(current_logits["LA"], dtype=np.float32)
            current_negative_lp = np.asarray(current_logits["LB"], dtype=np.float32)
            historical_mask = np.flatnonzero(np.asarray(historical_arrays["selected_mask"][0]) > 0).tolist()

            record = {
                "task": task,
                "seed": seed,
                "instruction": instruction,
                "historical_instruction": historical["instruction"],
                "task_key_match": historical["task"] == task,
                "canonical_snapshot_match": historical["canonical_snapshot_sha256"] == expected_snapshot,
                "initial_state_match": historical["initial_state_sha256"] == state_sha,
                "initial_rgb_match": historical["initial_rgb_sha256"] == rgb_sha,
                "historical_state_sha256": historical["initial_state_sha256"],
                "current_state_sha256": state_sha,
                "historical_rgb_sha256": historical["initial_rgb_sha256"],
                "current_rgb_sha256": rgb_sha,
                "clean_logprob_max_abs_diff": max_abs(current_clean_lp, historical_positive_lp),
                "negative_logprob_max_abs_diff": max_abs(current_negative_lp, historical_negative_lp),
                "clean_top1_historical": historical_positive_lp.argmax(axis=-1).tolist(),
                "clean_top1_current": current_clean_lp.argmax(axis=-1).tolist(),
                "negative_top1_historical": historical_negative_lp.argmax(axis=-1).tolist(),
                "negative_top1_current": current_negative_lp.argmax(axis=-1).tolist(),
                "attention_sha_match": current_meta["attention_meta"]["attention_sha256"] == historical_first["attention_sha256"],
                "m_match": int(current_meta["m_t"]) == int(historical_first["m_t"]),
                "selected_tokens_match": list(current_meta["selected_token_ids"]) == list(historical_first["selected_token_ids"]),
                "selected_tokens_match_mask": list(current_meta["selected_token_ids"]) == historical_mask,
                "historical_selected_token_ids": historical_first["selected_token_ids"],
                "current_selected_token_ids": current_meta["selected_token_ids"],
                "historical_m": historical_first["m_t"],
                "current_m": current_meta["m_t"],
                "reconstruction_delta_norm_historical": historical_first["feature_perturbation_norm"],
                "reconstruction_delta_norm_current": current_meta["reconstruction_delta_norm"],
                "reconstruction_delta_norm_abs_diff": abs(
                    float(historical_first["feature_perturbation_norm"]) - float(current_meta["reconstruction_delta_norm"])
                ),
                "old_harmonic": old_harmonic,
                "current_harmonic": current_harmonic,
                "harmonic_sha_match": bool(
                    old_harmonic and current_harmonic and old_harmonic["output_sha256"] == current_harmonic["output_sha256"]
                ),
                "historical_final_tokens": historical_first["final_token_ids"],
                "current_final_tokens": current_meta["final_action_tokens"],
                "final_tokens_match": list(historical_first["final_token_ids"]) == list(current_meta["final_action_tokens"]),
                "historical_guided_action": historical_first["guided_action"],
                "current_guided_action": current_meta["executed_action"],
                "guided_action_max_abs_diff": max_abs(historical_first["guided_action"], current_meta["executed_action"]),
                "historical_episode_steps": historical["control_steps"],
                "current_episode_steps": json.loads(
                    (args.artifact / "episodes" / task / "C" / f"episode_{seed:03d}_summary.json").read_text()
                )["episode_steps"]
                if (args.artifact / "episodes" / task / "C" / f"episode_{seed:03d}_summary.json").exists()
                else None,
                "old_rerun_positive_max_abs_diff": max_abs(
                    np.asarray(old_logits["positive"], dtype=np.float32),
                    np.asarray(historical_arrays["positive"][0], dtype=np.float32),
                ),
                "old_rerun_negative_max_abs_diff": max_abs(
                    np.asarray(old_logits["negative"], dtype=np.float32),
                    np.asarray(historical_arrays["negative"][0], dtype=np.float32),
                ),
                "old_rerun_final_tokens": old_meta["final_token_ids"],
                "old_rerun_final_tokens_match": list(old_meta["final_token_ids"]) == list(historical_first["final_token_ids"]),
            }
            record["first_divergence"] = next(
                (
                    key
                    for key in (
                        "canonical_snapshot_match",
                        "initial_state_match",
                        "initial_rgb_match",
                        "clean_logprob_max_abs_diff",
                        "attention_sha_match",
                        "m_match",
                        "selected_tokens_match",
                        "harmonic_sha_match",
                        "negative_logprob_max_abs_diff",
                        "final_tokens_match",
                        "guided_action_max_abs_diff",
                    )
                    if not (
                        record.get(key) is True
                        or (isinstance(record.get(key), (int, float)) and record.get(key) < 1e-5)
                    )
                ),
                None,
            )
            results.append(record)
            print(json.dumps({
                "task": task,
                "seed": seed,
                "first_divergence": record["first_divergence"],
                "clean_diff": record["clean_logprob_max_abs_diff"],
                "negative_diff": record["negative_logprob_max_abs_diff"],
                "mask_match": record["selected_tokens_match"],
                "harmonic_match": record["harmonic_sha_match"],
                "final_match": record["final_tokens_match"],
            }), flush=True)

    suffix = f"_{args.task}" if args.task else ""
    output = args.artifact / "analysis" / f"C_VS_HISTORICAL_FIRST_STEP_AUDIT{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "protocol_id": "C_VS_HISTORICAL_L11_MATCHED_FIRST_STEP_AUDIT",
        "historical": str(args.historical),
        "seeds": seeds,
        "n": len(results),
        "results": results,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n": len(results), "output": str(output)}))


if __name__ == "__main__":
    main()
