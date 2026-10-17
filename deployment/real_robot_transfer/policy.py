"""Construct the exact simulation VTCG policy for camera inference."""
from __future__ import annotations

import os

import numpy as np
import torch

from .openvla_runtime import OpenVLAContrastInference


def load_policy(checkpoint: str, unnorm_key: str, instruction: str, *, seed: int = 0,
                device: str = "cuda:0"):
    """Load once, then reuse the canonical research class without SIMPLER.

    The research rollout also changes a loaded base object's class. All
    instance state normally prepared by ``_init_common`` is initialized here.
    """
    os.environ["TOKEN_CD_REAL_ROBOT"] = "1"
    from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference

    base = OpenVLAContrastInference(checkpoint, unnorm_key, device)
    base.__class__ = PromptAttentionSHRInference
    base.alpha = base.lambd = 0.5
    base.beta = 0.0
    base.selection_mode = "semantic"
    base.kmeans_K = 8
    base.kmeans_seed = 0
    base.selector_mode = "prompt_attention"
    base.attention_layers = (11,)
    base.task_index = 0
    base.selection_count = None
    base.selection_top_p = None
    base.selection_budget_schedule = None
    base.selection_budget_scale = 1.0
    base.selection_budget_label = None
    base.selection_budget_source = "matched"
    base.counterfactual_constructor = "harmonic"
    base.guidance_execution = "contrast"
    base.save_prompt_attention = False
    base._selector_instr = None
    base._entities = []
    base._entity_emb = []
    base._emb_cache = {}
    base._episode_trace = []
    base._episode_logits = []
    base._episode_seed = seed
    base._selector_step = 0
    base.reset(instruction, seed=seed)
    if not base._entities:
        raise ValueError(
            "The rule-based entity parser found no source/target noun in the instruction; "
            "use an imperative task description supported by extract_entities()."
        )
    return base


def infer(policy, image: np.ndarray, instruction: str) -> tuple[dict, dict]:
    """Return decoded OpenVLA action and per-step method trace.

    The policy's second return value is the simulator action on the source
    runtime and None here; it must never be passed to a real robot.
    """
    with torch.inference_mode():
        raw_action, _, trace = policy.step(image, task_description=instruction)
    if trace.get("coverage_exact") is not True or trace.get("guided_prefix") is not True:
        raise RuntimeError("VTCG policy audit failed")
    if trace.get("feature_equal") is not True or trace.get("non_target_bit_identical") is not True:
        raise RuntimeError("projector intervention audit failed")
    if trace["positive_token_ids"][6] != trace["final_token_ids"][6]:
        raise RuntimeError("gripper action was modified by guidance")
    return raw_action, trace
