"""Shared policy construction for target-specific attention experiments."""
from __future__ import annotations

import copy

from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.target_specific_protocol import ATTENTION_LAYERS, LAMBDA, selector_config


def build_policy(base, task: str, arm: str):
    policy = copy.copy(base); policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0; policy.selector_mode = "prompt_attention"; policy.task_index = TASK_INDEX[task]
    policy.attention_layers = tuple(ATTENTION_LAYERS); policy.selection_count = None
    policy.selection_top_p = None; policy.save_prompt_attention = True; policy.selector_instruction = None
    policy.complement_arm = None; policy.prompt_action_rerank_multiplier = None
    policy.prompt_action_bounded_mode = None; policy.prompt_action_full_joint = False
    cfg = selector_config(task, arm)
    policy.selector_contrast_instruction = cfg["contrast_instruction"]
    policy.selector_difference_eta = cfg["eta"]
    policy.selector_difference_formula = cfg["formula"] or "log_boost"
    policy.selector_difference_kind = "generic_frame" if cfg["formula"] else None
    return policy

