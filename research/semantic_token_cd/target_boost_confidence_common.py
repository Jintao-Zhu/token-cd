"""Policy builder for confidence-gated Target Positive-Boost."""
from __future__ import annotations

import copy

from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.target_boost_confidence_protocol import LAMBDA, arm_config


def build_policy(base, task: str, arm: str):
    policy = copy.copy(base); policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0; policy.selector_mode = "prompt_attention"; policy.task_index = TASK_INDEX[task]
    policy.attention_layers = (11,); policy.selection_count = None; policy.selection_top_p = None
    policy.save_prompt_attention = True; policy.selector_instruction = None
    policy.complement_arm = None; policy.prompt_action_rerank_multiplier = None
    policy.prompt_action_bounded_mode = None; policy.prompt_action_full_joint = False
    cfg = arm_config(task, arm)
    policy.selector_contrast_instruction = cfg["contrast_instruction"]
    policy.selector_difference_eta = cfg["eta"]
    policy.selector_difference_formula = cfg["formula"]
    policy.selector_difference_kind = "generic_positive_boost_confidence_gate"
    policy.selector_difference_confidence_threshold = cfg["confidence_threshold"]
    return policy
