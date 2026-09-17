"""One-observation fail-closed preflight for Prompt-Attn-SHR v1."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import (
    build_policies,
    load_reference,
    make_environment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", default="google_robot_open_drawer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    source = args.snapshot_artifact.resolve()
    reference = load_reference(source, args.task, args.seed)
    snapshot_path = source / "snapshots" / args.task / f"seed_{args.seed:03d}.pkl"
    with snapshot_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    if snapshot_sha(snapshot) != reference["canonical_snapshot_sha256"]:
        raise RuntimeError("preflight snapshot SHA mismatch")
    env, _ = make_environment(args.task, args.gpu)
    obs, state_sha, rgb_sha = restore_snapshot(env, args.seed, snapshot)
    if (state_sha, rgb_sha) != (
        reference["initial_state_sha256"], reference["initial_rgb_sha256"]
    ):
        raise RuntimeError("preflight restored-state mismatch")
    instruction = env.unwrapped.get_language_instruction()
    image = get_image_from_maniskill2_obs_dict(env, obs)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task)

    outputs = {}
    for arm in ("standard_shr", "prompt_attn_shr", "random_shr"):
        policy = policies[arm]
        policy.reset(instruction, seed=args.seed)
        policy._episode_trace = []
        policy._episode_logits = []
        _, _, meta = policy.step(image, None, instruction)
        outputs[arm] = meta

    canonical_first = reference["selector_trace"][0]
    standard = outputs["standard_shr"]
    for key in ("positive_token_ids", "negative_token_ids", "final_token_ids", "selected_token_ids"):
        if standard[key] != canonical_first[key]:
            raise RuntimeError(f"standard SHR no longer matches canonical first step: {key}")
    prompt = outputs["prompt_attn_shr"]
    random = outputs["random_shr"]
    if prompt["positive_token_ids"] != canonical_first["positive_token_ids"]:
        raise RuntimeError("attention recording changed clean greedy action")
    if prompt["m_t"] != len(prompt["selected_token_ids"]):
        raise RuntimeError("prompt coverage mismatch")
    if random["m_t"] != len(random["selected_token_ids"]):
        raise RuntimeError("random coverage mismatch")
    result = {
        "technical_pass": True,
        "task": args.task,
        "seed": args.seed,
        "instruction": instruction,
        "standard_exact_canonical": True,
        "clean_action_unchanged_with_attention": True,
        "prompt_query_tokens": prompt["prompt_query_tokens"],
        "prompt_text_indices": prompt["prompt_text_indices"],
        "prompt_multimodal_query_indices": prompt["prompt_multimodal_query_indices"],
        "visual_key_indices": prompt["visual_key_indices"],
        "attention_layers": prompt["attention_layers"],
        "attention_head_count": prompt["attention_head_count"],
        "m_t": prompt["m_t"],
        "prompt_shr_overlap_ratio": prompt["prompt_shr_overlap_ratio"],
        "random_seed": random["random_seed"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
