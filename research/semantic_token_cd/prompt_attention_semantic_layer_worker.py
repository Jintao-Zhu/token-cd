#!/usr/bin/env python3
"""Box-free semantic-selectivity scan of all prompt-attention layers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_action_layer_causal_worker import (
    atomic_json,
    load_cases,
)


PROTOCOL = "PROMPT_ATTENTION_SEMANTIC_LAYER_SCAN_V1"


def irrelevant_rewording(instruction: str) -> str:
    return f"please {instruction}"


def attach_mismatched_images(cases: list[dict]) -> list[dict]:
    output = []
    for case in cases:
        candidates = [item for item in cases if item["task"] != case["task"]]
        candidates.sort(key=lambda item: item["state_id"])
        digest = hashlib.sha256(case["state_id"].encode()).digest()
        selected = candidates[int.from_bytes(digest[:8], "little") % len(candidates)]
        output.append({**case, "mismatched_npz": selected["npz"], "mismatched_state_id": selected["state_id"]})
    return output


def normalize(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    denominator = values.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0) or not np.isfinite(values).all():
        raise FloatingPointError("invalid attention distribution")
    return values / denominator


def js_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize(left)
    right = normalize(right)
    middle = 0.5 * (left + right)
    left_term = np.sum(left * np.log(np.maximum(left, 1e-15) / np.maximum(middle, 1e-15)), axis=-1)
    right_term = np.sum(right * np.log(np.maximum(right, 1e-15) / np.maximum(middle, 1e-15)), axis=-1)
    return np.sqrt(np.maximum(0.0, 0.5 * (left_term + right_term)))


def extract(policy, image: np.ndarray, instruction: str, clean_visual):
    from research.semantic_token_cd.prompt_attn_shr_policy import extract_prompt_attention_per_layer

    inputs = policy.process_inputs(image, task_description=instruction)
    attention, _ = extract_prompt_attention_per_layer(policy, inputs, instruction, clean_visual)
    return attention.astype(np.float64)


def capture_visual(policy, image: np.ndarray, instruction: str):
    from research.ar_token_counterfactual.intervention import projector_intervention

    inputs = policy.process_inputs(image, task_description=instruction)
    with projector_intervention(policy.vla) as trace:
        policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
    if trace.before is None:
        raise RuntimeError("projector capture failed")
    return trace.before


def process_case(policy, case: dict) -> dict:
    real_arrays = np.load(case["npz"])
    real_image = real_arrays["image"]
    real = {
        "original": real_arrays["original"].astype(np.float64),
        "synonym": real_arrays["synonym"].astype(np.float64),
    }
    metadata = json.loads(Path(case["npz"]).with_suffix(".json").read_text())
    prompts = {
        "original": case["instruction"],
        "synonym": metadata["synonym_instruction"],
        "semantic": case["target_instruction"],
        "irrelevant": irrelevant_rewording(case["instruction"]),
    }

    policy.reset(case["instruction"], seed=case["seed"])
    real_visual = capture_visual(policy, real_image, prompts["original"])
    reproduced = extract(policy, real_image, prompts["original"], real_visual)
    reproduction_max_abs = float(np.max(np.abs(reproduced - real["original"])))
    real["semantic"] = extract(policy, real_image, prompts["semantic"], real_visual)
    real["irrelevant"] = extract(policy, real_image, prompts["irrelevant"], real_visual)

    mismatch_arrays = np.load(case["mismatched_npz"])
    mismatch_image = mismatch_arrays["image"]
    mismatch_visual = capture_visual(policy, mismatch_image, prompts["original"])
    mismatch = {
        name: extract(policy, mismatch_image, instruction, mismatch_visual)
        for name, instruction in prompts.items()
    }

    real_para = js_distance(real["original"], real["synonym"])
    real_semantic = js_distance(real["original"], real["semantic"])
    real_irrelevant = js_distance(real["original"], real["irrelevant"])
    mismatch_para = js_distance(mismatch["original"], mismatch["synonym"])
    mismatch_semantic = js_distance(mismatch["original"], mismatch["semantic"])
    mismatch_irrelevant = js_distance(mismatch["original"], mismatch["irrelevant"])
    real_surface = np.maximum(real_para, real_irrelevant)
    mismatch_surface = np.maximum(mismatch_para, mismatch_irrelevant)
    real_margin = real_semantic - real_surface
    mismatch_margin = mismatch_semantic - mismatch_surface

    layers = []
    for layer in range(32):
        layers.append({
            "layer": layer,
            "real_paraphrase_distance": float(real_para[layer]),
            "real_semantic_distance": float(real_semantic[layer]),
            "real_irrelevant_distance": float(real_irrelevant[layer]),
            "real_semantic_margin": float(real_margin[layer]),
            "real_semantic_ratio": float(real_semantic[layer] / max(real_surface[layer], 1e-12)),
            "mismatch_paraphrase_distance": float(mismatch_para[layer]),
            "mismatch_semantic_distance": float(mismatch_semantic[layer]),
            "mismatch_irrelevant_distance": float(mismatch_irrelevant[layer]),
            "mismatch_semantic_margin": float(mismatch_margin[layer]),
            "grounded_semantic_margin": float(real_margin[layer] - mismatch_margin[layer]),
            "real_visual_attention_mass": float(real["original"][layer].sum()),
        })
    return {
        "protocol_id": PROTOCOL,
        **{key: case[key] for key in ("state_id", "task", "seed", "split", "instruction", "target_instruction")},
        "synonym_instruction": prompts["synonym"],
        "irrelevant_instruction": prompts["irrelevant"],
        "mismatched_state_id": case["mismatched_state_id"],
        "uses_manual_target_boxes": False,
        "attention_reproduction_max_abs_difference": reproduction_max_abs,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
    from research.semantic_token_cd.prompt_attn_shr_rollout import build_policies
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    import torch

    cases = attach_mismatched_images(load_cases())
    cases = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
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
        l11 = result["layers"][11]
        print(json.dumps({
            "done": case["state_id"], "split": case["split"],
            "l11_real_margin": l11["real_semantic_margin"],
            "l11_grounded_margin": l11["grounded_semantic_margin"],
        }), flush=True)


if __name__ == "__main__":
    main()
