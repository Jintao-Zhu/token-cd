#!/usr/bin/env python3
"""Box-free action-level semantic invariance/selectivity scan for all layers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_action_layer_causal_worker import (
    SOURCE,
    action_logits,
    atomic_json,
    load_cases,
    log_probability_residual,
    negative_scores,
    selected_indices,
)


PROTOCOL = "PROMPT_ACTION_LAYER_SEMANTIC_CONSISTENCY_V1"


def cosine_per_dimension(left, right) -> tuple[float, list[float]]:
    import torch

    left = left.double(); right = right.double()
    numerator = (left * right).sum(-1)
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    values = torch.where(
        denominator > 1e-12, numerator / denominator.clamp_min(1e-12), torch.zeros_like(denominator)
    )
    return float(values.mean().item()), [float(value) for value in values.tolist()]


def jaccard(left: list[int], right: list[int]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def process_case(policy, case: dict) -> dict:
    import torch

    from research.ar_token_counterfactual.intervention import projector_intervention
    from research.semantic_token_cd.prompt_attn_shr_policy import extract_prompt_attention_per_layer

    arrays = np.load(case["npz"])
    image = arrays["image"]
    original_attention = arrays["original"].astype(np.float64)
    synonym_attention = arrays["synonym"].astype(np.float64)
    metadata = json.loads(Path(case["npz"]).with_suffix(".json").read_text())
    synonym_instruction = metadata["synonym_instruction"]
    count = int(case["count"])

    policy.reset(case["instruction"], seed=case["seed"])
    inputs = policy.process_inputs(image, task_description=case["instruction"])
    with projector_intervention(policy.vla) as trace:
        clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
    if trace.before is None or clean_scores.shape[0] != 7:
        raise RuntimeError("positive branch capture failed")
    visual = trace.before
    clean_action = action_logits(policy, clean_scores)

    target_inputs = policy.process_inputs(image, task_description=case["target_instruction"])
    target_attention, _ = extract_prompt_attention_per_layer(
        policy, target_inputs, case["target_instruction"], visual
    )
    target_attention = target_attention.astype(np.float64)

    layers = []
    for layer in range(32):
        masks = {
            "original": selected_indices(original_attention[layer], count, "top"),
            "synonym": selected_indices(synonym_attention[layer], count, "top"),
            "target": selected_indices(target_attention[layer], count, "top"),
        }
        residuals = {}
        perturbations = {}
        for name, selected in masks.items():
            branch, perturbation = negative_scores(policy, inputs, clean_scores, visual, selected)
            residuals[name] = log_probability_residual(clean_action, action_logits(policy, branch))
            perturbations[name] = perturbation
        synonym_cosine, synonym_per_dimension = cosine_per_dimension(
            residuals["original"], residuals["synonym"]
        )
        target_cosine, target_per_dimension = cosine_per_dimension(
            residuals["original"], residuals["target"]
        )
        layers.append({
            "layer": layer,
            "synonym_residual_cosine": synonym_cosine,
            "target_residual_cosine": target_cosine,
            "semantic_separation": synonym_cosine - target_cosine,
            "synonym_residual_cosine_per_dimension": synonym_per_dimension,
            "target_residual_cosine_per_dimension": target_per_dimension,
            "synonym_mask_jaccard": jaccard(masks["original"], masks["synonym"]),
            "target_mask_jaccard": jaccard(masks["original"], masks["target"]),
            "mask_jaccard_separation": (
                jaccard(masks["original"], masks["synonym"])
                - jaccard(masks["original"], masks["target"])
            ),
            "feature_perturbation_norms": perturbations,
        })
    return {
        "protocol_id": PROTOCOL,
        **{key: case[key] for key in (
            "state_id", "task", "seed", "split", "instruction", "target_instruction", "count"
        )},
        "synonym_instruction": synonym_instruction,
        "uses_manual_target_boxes": False,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
    from research.semantic_token_cd.prompt_attn_shr_rollout import build_policies
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    cases = [case for index, case in enumerate(load_cases()) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        cases = cases[:args.limit]
    artifact = args.artifact.resolve()
    current_task, policy = None, None
    for case in cases:
        output = artifact / "results" / f"{case['state_id']}.json"
        if output.exists():
            print(json.dumps({"skip": case["state_id"]}), flush=True); continue
        if case["task"] != current_task:
            if policy is not None:
                del policy
                import torch
                torch.cuda.empty_cache()
            checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
            config = get_policy_config("openvla", checkpoint, case["task"], {}, False)
            policy = build_policies(OpenVLAInference(**config), case["task"])["prompt_attn_shr"]
            current_task = case["task"]
        result = process_case(policy, case)
        atomic_json(output, result)
        print(json.dumps({
            "done": case["state_id"], "split": case["split"],
            "l11_synonym_cosine": result["layers"][11]["synonym_residual_cosine"],
            "l11_semantic_separation": result["layers"][11]["semantic_separation"],
        }), flush=True)


if __name__ == "__main__":
    main()
