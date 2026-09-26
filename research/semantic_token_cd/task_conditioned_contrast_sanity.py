#!/usr/bin/env python3
"""Focused implementation sanity checks for task-conditioned contrast + APC."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.task_conditioned_contrast_protocol import (
    APC_BETA,
    ARTIFACT,
    ATTENTION_LAYERS,
    CANONICAL,
    CONTROL_INSTRUCTIONS,
    KMEANS_K,
    KMEANS_SEED,
    LAMBDA0,
    PCD_SOURCE,
    PROTOCOL,
    TASKS,
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_policy(base, task: str, arm: str, *, identity: bool = False, zero_control: bool = False):
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
    policy.arm = arm
    policy.apc_beta = APC_BETA
    policy.control_instruction = CONTROL_INSTRUCTIONS[task]
    policy.task_key = task
    policy.save_condition_logits = True
    policy.debug_identity_reconstruction = identity
    policy.debug_zero_control_residual = zero_control
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task", choices=TASKS, default="google_robot_open_drawer")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    repo = Path("/home/leju-suzhou/zjt_ws/token-cd")
    for path in (str(repo / "task1/shim_site"), str(repo), str(PCD_SOURCE)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import restore_snapshot
    from research.semantic_token_cd.spatial_grid_rollout import make_environment

    task = args.task
    seed = args.seed
    snapshot_path = CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl"
    snapshot = pickle.load(snapshot_path.open("rb"))
    env, _environment_id = make_environment(task)
    obs, _state_sha, _rgb_sha = restore_snapshot(env, seed, snapshot)
    instruction = env.unwrapped.get_language_instruction()
    image = get_image_from_maniskill2_obs_dict(env, obs)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    base = OpenVLAInference(**config)

    results = {}
    masks = {}
    for arm in ("C", "C_APC", "T", "T_APC"):
        policy = build_policy(base, task, arm)
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        _raw, _actions, meta = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        results[arm] = meta
        masks[arm] = meta["selected_token_ids"]
    results["mask_invariance"] = len({tuple(value) for value in masks.values()}) == 1

    # Identity reconstruction => reconstruction is exactly clean and the final
    # action is unchanged.  A and B use different numerical forward paths
    # (generation vs teacher-forced), so their logits are not expected to be
    # bitwise identical; the contract is that the visual feature is identity.
    for arm in ("C", "T"):
        policy = build_policy(base, task, arm, identity=True)
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        _raw, _actions, meta = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        results[f"identity_{arm}"] = {
            "S_visual_rms": meta["S_visual_rms"],
            "S_control_rms": meta["S_control_rms"],
            "S_task_rms": meta["S_task_rms"],
            "reconstruction_delta_norm": meta["reconstruction_delta_norm"],
            "changed_dims": meta["changed_dims_vs_clean"],
            "pass": meta["reconstruction_delta_norm"] < 1e-6 and meta["changed_dims_vs_clean"] == 0,
        }

    # Remove control residual => T must equal C.
    policy = build_policy(base, task, "T", zero_control=True)
    policy.reset(instruction, seed=seed)
    policy._episode_trace = []
    policy._episode_logits = []
    _raw, _actions, meta = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
    results["remove_control"] = {
        "same_final_tokens": meta["final_action_tokens"] == results["C"]["final_action_tokens"],
        "pass": meta["final_action_tokens"] == results["C"]["final_action_tokens"],
    }

    # APC beta limits.
    from research.semantic_token_cd.task_conditioned_contrast_policy import _apc_apply
    scores = torch.tensor([[0.0, -1.0, -3.0]], dtype=torch.float32)
    clean = torch.log_softmax(scores, dim=-1)
    constrained_one, _ = _apc_apply(scores, clean, 1.0)
    constrained_zero, _ = _apc_apply(scores, clean, 1e-6)
    results["apc_limits"] = {
        "beta1_top1_only": int(constrained_one.argmax(-1).item()) == int(clean.argmax(-1).item()),
        "beta_small_matches_raw": int(constrained_zero.argmax(-1).item()) == int(scores.argmax(-1).item()),
        "pass": int(constrained_one.argmax(-1).item()) == 0 and int(constrained_zero.argmax(-1).item()) == 0,
    }

    results["technical_pass"] = (
        results["mask_invariance"]
        and all(results[f"identity_{arm}"]["pass"] for arm in ("C", "T"))
        and results["remove_control"]["pass"]
        and results["apc_limits"]["pass"]
    )
    output = args.artifact / "sanity" / "SANITY_RESULTS.json"
    atomic_json(output, {"protocol_id": PROTOCOL, "task": task, "seed": seed, "results": results})
    print(json.dumps({"technical_pass": results["technical_pass"], "output": str(output)}, indent=2))
    if not results["technical_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
