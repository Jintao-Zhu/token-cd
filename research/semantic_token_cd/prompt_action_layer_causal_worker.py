#!/usr/bin/env python3
"""Box-free offline layer screen using prompt-to-action intervention alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "artifacts/prompt_attn_layer_selection_v1"
PROTOCOL = "PROMPT_ACTION_LAYER_CAUSAL_V1"
ACTION_BINS = 256
LAYERS = tuple(range(32))
CONTROLS = ("top", "bottom", "wrong_prompt")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def action_logits(policy, scores):
    start = int(policy.vla.vocab_size) - ACTION_BINS
    values = scores[:6, start : start + ACTION_BINS].detach().float().cpu()
    if tuple(values.shape) != (6, ACTION_BINS):
        raise RuntimeError(f"invalid action logits: {tuple(values.shape)}")
    return values


def log_probability_residual(clean, branch):
    import torch

    return torch.log_softmax(clean, dim=-1) - torch.log_softmax(branch, dim=-1)


def residual_metrics(residual, prompt_residual) -> dict:
    import torch

    residual = residual.double()
    prompt_residual = prompt_residual.double()
    numerator = (residual * prompt_residual).sum(dim=-1)
    denominator = torch.linalg.vector_norm(residual, dim=-1) * torch.linalg.vector_norm(
        prompt_residual, dim=-1
    )
    per_dimension = torch.where(
        denominator > 1e-12, numerator / denominator.clamp_min(1e-12), torch.zeros_like(denominator)
    )
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    prompt_norm = float(torch.linalg.vector_norm(prompt_residual).item())
    return {
        "alignment": float(per_dimension.mean().item()),
        "alignment_per_dimension": [float(value) for value in per_dimension.tolist()],
        "residual_norm": residual_norm,
        "prompt_residual_norm": prompt_norm,
        "strength_ratio": residual_norm / max(1e-12, prompt_norm),
    }


def selected_indices(scores: np.ndarray, count: int, mode: str) -> list[int]:
    from research.semantic_token_cd.prompt_attn_shr_policy import stable_top_m

    values = np.asarray(scores, dtype=np.float64)
    if mode == "top":
        return stable_top_m(values, count)
    if mode == "bottom":
        return stable_top_m(-values, count)
    raise ValueError(mode)


def negative_scores(policy, inputs, clean_scores, visual, selected: list[int]):
    import torch

    from research.semantic_token_cd.global_merge_policy import (
        guided_forward_scores,
        projector_merge_intervention,
    )
    from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct

    features = visual[0].numpy().astype(np.float32)
    negative = features.copy()
    selected_array = np.asarray(selected, dtype=np.int64)
    negative[selected_array] = harmonic_reconstruct(features, selected_array, beta=0.0)
    with projector_merge_intervention(
        policy.vla, torch.from_numpy(negative).unsqueeze(0)
    ) as trace:
        scores = guided_forward_scores(
            policy.vla, inputs, clean_scores.argmax(dim=-1), visual.shape[1]
        )
    if trace["before"] is None or not torch.equal(trace["before"], visual):
        raise RuntimeError("negative branch visual features differ from positive branch")
    return scores, float(np.linalg.norm(negative - features))


def load_cases() -> list[dict]:
    manifest = json.loads((SOURCE / "SPLIT_AND_CONTROLS.json").read_text())
    controls = {item["state_id"]: item for item in manifest["target_controls"]}
    cases = []
    for path in sorted((SOURCE / "states").glob("**/step_*.json")):
        metadata = json.loads(path.read_text())
        control = controls.get(metadata["state_id"])
        cases.append({
            "state_id": metadata["state_id"],
            "task": metadata["task"],
            "seed": int(metadata["seed"]),
            "split": metadata["split"],
            "instruction": metadata["instruction"],
            "target_instruction": alternate_instruction(metadata["instruction"]),
            "count": int(metadata["m"]),
            "npz": str(path.with_suffix(".npz").resolve()),
            "has_original_manual_control": control is not None,
        })
    return sorted(cases, key=lambda item: item["state_id"])


def alternate_instruction(instruction: str) -> str:
    drawer = {
        "open top drawer": "open middle drawer",
        "open middle drawer": "open bottom drawer",
        "open bottom drawer": "open top drawer",
    }
    if instruction in drawer:
        return drawer[instruction]
    if instruction == "pick coke can":
        return "pick pepsi can"
    if instruction.startswith("move ") and " near " in instruction:
        source, target = instruction.removeprefix("move ").split(" near ", 1)
        return f"move {target} near {source}"
    raise ValueError(f"no valid alternate instruction rule for: {instruction}")


def process_case(policy, case: dict) -> dict:
    import torch

    from research.ar_token_counterfactual.intervention import projector_intervention
    from research.semantic_token_cd.global_merge_policy import guided_forward_scores

    arrays = np.load(case["npz"])
    image = arrays["image"]
    original_attention = arrays["original"].astype(np.float64)
    cached_clean = arrays["clean_positive"][:6].astype(np.float32)
    count = int(case["count"])

    policy.reset(case["instruction"], seed=case["seed"])
    original_inputs = policy.process_inputs(image, task_description=case["instruction"])
    with projector_intervention(policy.vla) as positive_trace:
        clean_scores = policy._forward_scores(original_inputs, policy.unnorm_key, do_sample=False)
    if positive_trace.before is None or clean_scores.shape[0] != 7:
        raise RuntimeError("positive branch capture failed")
    visual = positive_trace.before
    clean_action = action_logits(policy, clean_scores)
    cached_max_abs = float(np.max(np.abs(clean_action.numpy() - cached_clean)))

    target_inputs = policy.process_inputs(image, task_description=case["target_instruction"])
    from research.semantic_token_cd.prompt_attn_shr_policy import extract_prompt_attention_per_layer
    target_attention, target_attention_meta = extract_prompt_attention_per_layer(
        policy, target_inputs, case["target_instruction"], visual
    )
    target_attention = target_attention.astype(np.float64)
    with projector_intervention(policy.vla) as target_trace:
        target_shared_scores = guided_forward_scores(
            policy.vla, target_inputs, clean_scores.argmax(dim=-1), visual.shape[1]
        )
    if target_trace.before is None or not torch.equal(target_trace.before, visual):
        raise RuntimeError("target-prompt visual features differ from original prompt")
    target_action = action_logits(policy, target_shared_scores)
    prompt_residual = log_probability_residual(clean_action, target_action)
    prompt_norm = float(torch.linalg.vector_norm(prompt_residual).item())
    if not np.isfinite(prompt_norm) or prompt_norm <= 1e-8:
        raise RuntimeError(f"degenerate prompt residual: {prompt_norm}")

    random_seed = int.from_bytes(
        hashlib.sha256(case["state_id"].encode()).digest()[:8], "little"
    )
    random_selected = sorted(
        np.random.default_rng(random_seed).choice(256, size=count, replace=False).tolist()
    )
    random_scores, random_perturbation = negative_scores(
        policy, original_inputs, clean_scores, visual, random_selected
    )
    random_metrics = residual_metrics(
        log_probability_residual(clean_action, action_logits(policy, random_scores)), prompt_residual
    )
    random_metrics.update({
        "selected_token_ids": random_selected,
        "feature_perturbation_norm": random_perturbation,
    })

    layers = []
    for layer in LAYERS:
        branches = {}
        masks = {
            "top": selected_indices(original_attention[layer], count, "top"),
            "bottom": selected_indices(original_attention[layer], count, "bottom"),
            "wrong_prompt": selected_indices(target_attention[layer], count, "top"),
        }
        for mode in CONTROLS:
            branch_scores, perturbation = negative_scores(
                policy, original_inputs, clean_scores, visual, masks[mode]
            )
            metrics = residual_metrics(
                log_probability_residual(clean_action, action_logits(policy, branch_scores)),
                prompt_residual,
            )
            metrics.update({
                "selected_token_ids": masks[mode],
                "feature_perturbation_norm": perturbation,
            })
            branches[mode] = metrics
        layers.append({
            "layer": layer,
            "branches": branches,
            "top_minus_random": branches["top"]["alignment"] - random_metrics["alignment"],
            "top_minus_bottom": branches["top"]["alignment"] - branches["bottom"]["alignment"],
            "top_minus_wrong_prompt": (
                branches["top"]["alignment"] - branches["wrong_prompt"]["alignment"]
            ),
            "causal_margin": branches["top"]["alignment"] - max(
                random_metrics["alignment"],
                branches["bottom"]["alignment"],
                branches["wrong_prompt"]["alignment"],
            ),
        })
    return {
        "protocol_id": PROTOCOL,
        **{key: case[key] for key in (
            "state_id", "task", "seed", "split", "instruction", "target_instruction", "count"
        )},
        "uses_manual_target_boxes": False,
        "has_original_manual_control": bool(case["has_original_manual_control"]),
        "target_attention_sha256": hashlib.sha256(
            np.ascontiguousarray(target_attention.astype(np.float32)).tobytes()
        ).hexdigest(),
        "target_attention_meta": target_attention_meta,
        "prompt_residual_norm": prompt_norm,
        "cached_clean_action_max_abs_difference": cached_max_abs,
        "random": random_metrics,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-original-controls", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
    from research.semantic_token_cd.prompt_attn_shr_rollout import build_policies
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    cases = load_cases()
    if args.only_original_controls:
        cases = [case for case in cases if case["has_original_manual_control"]]
    cases = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        cases = cases[: args.limit]
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
                import torch
                torch.cuda.empty_cache()
            checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
            config = get_policy_config("openvla", checkpoint, case["task"], {}, False)
            policy = build_policies(OpenVLAInference(**config), case["task"])["prompt_attn_shr"]
            current_task = case["task"]
        result = process_case(policy, case)
        atomic_json(output, result)
        print(json.dumps({
            "done": case["state_id"],
            "split": case["split"],
            "l11_alignment": result["layers"][11]["branches"]["top"]["alignment"],
            "l11_margin": result["layers"][11]["causal_margin"],
        }), flush=True)


if __name__ == "__main__":
    main()
