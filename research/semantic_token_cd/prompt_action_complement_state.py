"""Shared same-state extraction/evaluation for Prompt/Action complement v1."""
from __future__ import annotations

import copy

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import ACTION_VOCAB_SIZE, _action_logits
from research.semantic_token_cd.global_merge_policy import guided_forward_scores, projector_merge_intervention
from research.semantic_token_cd.prompt_action_complement_protocol import ACTION_LAYERS, LAMBDA, PROMPT_LAYERS
from research.semantic_token_cd.prompt_attn_shr_policy import (
    PromptAttentionSHRInference, construct_prompt_action_masks,
    extract_action_attention, extract_prompt_attention,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct


def build_policy(base, task: str):
    policy = copy.copy(base); policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0; policy.selector_mode = "prompt_attention"; policy.task_index = TASK_INDEX[task]
    policy.attention_layers = PROMPT_LAYERS; policy.action_attention_layers = ACTION_LAYERS
    policy.selection_count = None; policy.selection_top_p = None; policy.save_prompt_attention = True
    policy.selector_instruction = None; policy.selector_contrast_instruction = None
    policy.selector_difference_eta = None; policy.complement_arm = "original"
    return policy


@torch.inference_mode()
def extract_state(policy, image: np.ndarray, instruction: str, seed: int, step: int):
    policy._episode_seed = seed; policy._selector_step = step; policy.reset(instruction, seed=seed)
    inputs = policy.process_inputs(image, task_description=instruction)
    with projector_intervention(policy.vla) as trace:
        clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
    if clean_scores.shape[0] != 7 or trace.before is None:
        raise RuntimeError("clean state extraction did not produce 7 action logits and visual features")
    visual = trace.before
    features = visual[0].numpy().astype(np.float32)
    prompt, prompt_meta = extract_prompt_attention(policy, inputs, instruction, visual, layers=PROMPT_LAYERS)
    action, action_meta, action_arrays = extract_action_attention(
        policy, inputs, clean_scores.argmax(-1), clean_scores, visual, layers=ACTION_LAYERS
    )
    labels, _, per_entity = policy._semantic_clusters(features)
    groups = policy._entity_groups(features, labels)
    reference = sorted(set(int(index) for group in groups for index in np.flatnonzero(labels == group)))
    m = len(reference)
    if not 1 <= m <= 204:
        raise RuntimeError(f"coverage m={m} cannot satisfy locked candidate-pool construction")
    masks = construct_prompt_action_masks(prompt, action, m)
    return {
        "inputs": inputs, "clean_scores": clean_scores, "visual": visual, "features": features,
        "prompt": prompt, "action": action, "prompt_meta": prompt_meta,
        "action_meta": action_meta, "action_arrays": action_arrays,
        "reference": reference, "m": m, "masks": masks,
        "labels": labels, "per_entity_score": per_entity, "entity_groups": groups,
    }


def centered_residual(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    residual = np.asarray(positive[:6], dtype=np.float64) - np.asarray(negative[:6], dtype=np.float64)
    return residual - residual.mean(axis=-1, keepdims=True)


@torch.inference_mode()
def evaluate_mask(policy, state: dict, selected: list[int]):
    features = state["features"]
    negative_features = features.copy()
    negative_features[selected] = harmonic_reconstruct(
        features, np.asarray(selected, dtype=np.int64), beta=0.0
    )
    outside = np.asarray([index for index in range(256) if index not in set(selected)])
    if not np.array_equal(negative_features[outside], features[outside]):
        raise RuntimeError("mask-outside features changed")
    with projector_merge_intervention(
        policy.vla, torch.from_numpy(negative_features).unsqueeze(0)
    ) as negative_trace:
        negative_scores = guided_forward_scores(
            policy.vla, state["inputs"], state["clean_scores"].argmax(-1), 256
        )
    if negative_trace["before"] is None or not torch.equal(state["visual"], negative_trace["before"]):
        raise RuntimeError("negative projector input differs from clean features")
    final_scores, _guidance = policy._combine_action_scores(state["clean_scores"], negative_scores)
    eos_id = int(policy.vla.generation_config.eos_token_id)
    invalid = ~torch.isfinite(final_scores)
    if invalid.any():
        expected = torch.zeros_like(invalid, dtype=torch.bool); expected[:, eos_id] = True
        if (invalid & ~expected).any(): raise FloatingPointError("non-EOS final logits are non-finite")
        final_scores = final_scores.clone(); final_scores[:, eos_id] = torch.finfo(final_scores.dtype).min
    token_ids = final_scores.argmax(-1)
    clean_ids = state["clean_scores"].argmax(-1)
    clean_action = np.asarray(policy._decode_actions(clean_ids, policy.unnorm_key), dtype=np.float64)
    guided_action = np.asarray(policy._decode_actions(token_ids, policy.unnorm_key), dtype=np.float64)
    positive = _action_logits(policy, state["clean_scores"]).astype(np.float32)
    negative = _action_logits(policy, negative_scores).astype(np.float32)
    guided = _action_logits(policy, final_scores).astype(np.float32)
    residual = centered_residual(positive, negative)
    clean_top2 = np.sort(positive[:6], axis=-1)[:, -2:]
    guided_top2 = np.sort(guided[:6], axis=-1)[:, -2:]
    return {
        "metrics": {
            "feature_perturbation_norm": float(np.linalg.norm(negative_features - features)),
            "feature_perturbation_relative": float(np.linalg.norm(negative_features - features) /
                                                    (np.linalg.norm(features) + 1e-12)),
            "centered_residual_norm": float(np.linalg.norm(residual)),
            "positive_token_ids": clean_ids.detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "guided_changed_dims": int((token_ids[:6] != clean_ids[:6]).sum().item()),
            "clean_action": clean_action.tolist(), "guided_action": guided_action.tolist(),
            "guided_clean_action_l2": float(np.linalg.norm(guided_action[:6] - clean_action[:6])),
            "clean_top2_margin": (clean_top2[:, 1] - clean_top2[:, 0]).astype(float).tolist(),
            "guided_top2_margin": (guided_top2[:, 1] - guided_top2[:, 0]).astype(float).tolist(),
        },
        "arrays": {"positive": positive, "negative": negative, "guided": guided,
                   "centered_residual": residual.astype(np.float32),
                   "per_token_perturbation_norm": np.linalg.norm(
                       negative_features - features, axis=-1).astype(np.float32)},
    }

