#!/usr/bin/env python3
"""Offline matched-budget robustness scan for shortlisted prompt-attention layers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_action_layer_causal_worker import (
    action_logits,
    atomic_json,
    load_cases,
    log_probability_residual,
    negative_scores,
    residual_metrics,
    selected_indices,
)
from research.semantic_token_cd.prompt_action_layer_semantic_consistency_worker import (
    cosine_per_dimension,
    jaccard,
)


PROTOCOL = "PROMPT_ACTION_LAYER_BUDGET_ROBUSTNESS_V2"
LAYERS = (9, 11, 14)
SCALES = (0.75, 1.0, 1.25)


def scaled_count(count: int, scale: float) -> int:
    return min(256, max(1, int(np.rint(count * scale))))


def process_case(policy, case: dict) -> dict:
    import torch

    from research.ar_token_counterfactual.intervention import projector_intervention
    from research.semantic_token_cd.global_merge_policy import guided_forward_scores
    from research.semantic_token_cd.prompt_attn_shr_policy import extract_prompt_attention_per_layer

    arrays = np.load(case["npz"])
    image = arrays["image"]
    original_attention = arrays["original"].astype(np.float64)
    synonym_attention = arrays["synonym"].astype(np.float64)
    cached_clean = arrays["clean_positive"][:6].astype(np.float32)
    metadata = json.loads(Path(case["npz"]).with_suffix(".json").read_text())

    policy.reset(case["instruction"], seed=case["seed"])
    inputs = policy.process_inputs(image, task_description=case["instruction"])
    with projector_intervention(policy.vla) as trace:
        clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
    if trace.before is None or clean_scores.shape[0] != 7:
        raise RuntimeError("positive branch capture failed")
    visual = trace.before
    clean_action = action_logits(policy, clean_scores)
    cached_max_abs = float(np.max(np.abs(clean_action.numpy() - cached_clean)))

    target_inputs = policy.process_inputs(image, task_description=case["target_instruction"])
    target_attention, _ = extract_prompt_attention_per_layer(
        policy, target_inputs, case["target_instruction"], visual
    )
    target_attention = target_attention.astype(np.float64)
    with projector_intervention(policy.vla) as target_trace:
        target_scores = guided_forward_scores(
            policy.vla, target_inputs, clean_scores.argmax(dim=-1), visual.shape[1]
        )
    if target_trace.before is None or not torch.equal(target_trace.before, visual):
        raise RuntimeError("target-prompt visual features differ from original prompt")
    prompt_residual = log_probability_residual(clean_action, action_logits(policy, target_scores))

    results = []
    random_seed = int.from_bytes(
        hashlib.sha256(case["state_id"].encode()).digest()[:8], "little"
    )
    random_order = np.random.default_rng(random_seed).permutation(256)
    for scale in SCALES:
        count = scaled_count(int(case["count"]), scale)
        random_selected = sorted(random_order[:count].tolist())
        random_scores, _ = negative_scores(policy, inputs, clean_scores, visual, random_selected)
        random_alignment = residual_metrics(
            log_probability_residual(clean_action, action_logits(policy, random_scores)),
            prompt_residual,
        )["alignment"]

        for layer in LAYERS:
            masks = {
                "original": selected_indices(original_attention[layer], count, "top"),
                "synonym": selected_indices(synonym_attention[layer], count, "top"),
                "target": selected_indices(target_attention[layer], count, "top"),
            }
            residuals = {}
            perturbations = {}
            for name, selected in masks.items():
                branch, perturbation = negative_scores(
                    policy, inputs, clean_scores, visual, selected
                )
                residuals[name] = log_probability_residual(
                    clean_action, action_logits(policy, branch)
                )
                perturbations[name] = perturbation
            alignment = residual_metrics(residuals["original"], prompt_residual)["alignment"]
            synonym_cosine, _ = cosine_per_dimension(
                residuals["original"], residuals["synonym"]
            )
            target_cosine, _ = cosine_per_dimension(
                residuals["original"], residuals["target"]
            )
            results.append({
                "layer": layer,
                "scale": scale,
                "count": count,
                "top_alignment": alignment,
                "random_alignment": random_alignment,
                "top_minus_random": alignment - random_alignment,
                "synonym_residual_cosine": synonym_cosine,
                "target_residual_cosine": target_cosine,
                "semantic_separation": synonym_cosine - target_cosine,
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
        "synonym_instruction": metadata["synonym_instruction"],
        "cached_clean_action_max_abs_difference": cached_max_abs,
        "uses_manual_target_boxes": False,
        "layers": list(LAYERS),
        "scales": list(SCALES),
        "results": results,
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
    import torch

    cases = [
        case for index, case in enumerate(load_cases())
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        cases = cases[:args.limit]
    artifact = args.artifact.resolve()
    current_task = None
    policy = None
    for case in cases:
        output = artifact / "results" / f"{case['state_id']}.json"
        if output.exists():
            print(json.dumps({"skip": case["state_id"]}), flush=True)
            continue
        if case["task"] != current_task:
            if policy is not None:
                del policy
                torch.cuda.empty_cache()
            checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
            config = get_policy_config("openvla", checkpoint, case["task"], {}, False)
            policy = build_policies(OpenVLAInference(**config), case["task"])["prompt_attn_shr"]
            current_task = case["task"]
        result = process_case(policy, case)
        atomic_json(output, result)
        l11 = next(
            row for row in result["results"]
            if row["layer"] == 11 and row["scale"] == 1.0
        )
        print(json.dumps({
            "done": case["state_id"],
            "split": case["split"],
            "l11_alignment": l11["top_alignment"],
            "l11_separation": l11["semantic_separation"],
        }), flush=True)


if __name__ == "__main__":
    main()
