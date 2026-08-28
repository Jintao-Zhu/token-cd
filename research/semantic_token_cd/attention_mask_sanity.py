"""Single-state preregistered sanity check for Attention-Mask Random CD."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.distractor_rollout import (
    MEAN_PATH,
    PCD_SOURCE,
    build_policies,
    capture_snapshot,
    get_image_from_maniskill2_obs_dict,
    make_environment,
    restore_snapshot,
)
from research.semantic_token_cd.attention_mask_policy import (
    AttentionMaskEntityCDInference,
)


RATIO_THRESHOLD = 0.5


def build_attention_random(base_policy):
    policy = copy.copy(base_policy)
    policy.__class__ = AttentionMaskEntityCDInference
    policy.alpha = 0.5
    policy.lambd = 0.5
    policy.kmeans_K = 8
    policy.kmeans_seed = 0
    policy.selection_mode = "random_matched"
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
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean-path", type=Path, default=MEAN_PATH)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    env, environment_id = make_environment(args.task)
    snapshot = capture_snapshot(env, args.seed)
    obs, state_sha, rgb_sha = restore_snapshot(env, args.seed, snapshot)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    instruction = env.unwrapped.get_language_instruction()

    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**config)
    mean = torch.load(args.mean_path.resolve(), map_location="cpu", weights_only=True)["mean"]
    legacy_random = build_policies(base, mean)["random_cd"]
    attention_random = build_attention_random(base)

    outputs = {}
    for name, policy in (
        ("legacy_random", legacy_random),
        ("attention_random", attention_random),
    ):
        policy.reset(instruction, seed=args.seed)
        policy._episode_trace = []
        policy._episode_logits = []
        policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        outputs[name] = policy._episode_trace[0]

    legacy_norm = float(outputs["legacy_random"]["residual_norm"])
    attention_norm = float(outputs["attention_random"]["residual_norm"])
    ratio = attention_norm / legacy_norm if legacy_norm > 0 else float("inf")
    same_selection = (
        outputs["legacy_random"]["selected_token_ids"]
        == outputs["attention_random"]["selected_token_ids"]
    )
    result = {
        "protocol_id": "CAUSAL_ATTENTION_MASKING_CD_SANITY_V1",
        "task": args.task,
        "environment_id": environment_id,
        "seed": args.seed,
        "initial_state_sha256": state_sha,
        "initial_rgb_sha256": rgb_sha,
        "layer_interval": [16, 32],
        "ratio_threshold_preregistered": RATIO_THRESHOLD,
        "legacy_random_residual_norm": legacy_norm,
        "attention_random_residual_norm": attention_norm,
        "attention_to_legacy_ratio": ratio,
        "same_matched_random_selection": same_selection,
        "visual_features_bit_identical": bool(outputs["attention_random"]["feature_equal"]),
        "attention_mask_audit": outputs["attention_random"]["attention_mask"],
        "pass": bool(
            same_selection
            and outputs["attention_random"]["feature_equal"]
            and np.isfinite(ratio)
            and ratio < RATIO_THRESHOLD
        ),
    }
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    output = artifact / "sanity.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
