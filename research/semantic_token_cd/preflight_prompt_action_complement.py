"""One-state fail-closed audit for Prompt/Action complement v1."""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from research.semantic_token_cd.prompt_action_complement_protocol import (
    ACTION_LAYERS, ARMS, ARTIFACT, LAMBDA, PCD_SOURCE, PROMPT_LAYERS,
    STATE_SOURCE, atomic_json,
)
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common


TASK = "google_robot_open_drawer"


def build(base, arm):
    policy = copy.copy(base); policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0; policy.selector_mode = "prompt_attention"; policy.task_index = TASK_INDEX[TASK]
    policy.attention_layers = PROMPT_LAYERS; policy.action_attention_layers = ACTION_LAYERS
    policy.selection_count = None; policy.selection_top_p = None; policy.save_prompt_attention = True
    policy.selector_instruction = None; policy.selector_contrast_instruction = None
    policy.selector_difference_eta = None; policy.complement_arm = arm
    return policy


def run(policy, image, instruction):
    policy._episode_trace = []; policy._episode_logits = []
    policy._episode_seed = 0; policy._selector_step = 0; policy.reset(instruction, seed=0)
    policy.step(image, None, instruction, proprio=np.zeros(8, dtype=np.float64))
    return policy._episode_trace[-1], policy._episode_logits[-1]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    image_path = STATE_SOURCE / "runs/emitted_states/google_robot_open_drawer/seed_000/step_0013.npz"
    if not image_path.exists():
        image_path = sorted((STATE_SOURCE / "runs/emitted_states/google_robot_open_drawer").glob("seed_*/*.npz"))[0]
    offline = STATE_SOURCE / "runs/offline/google_robot_open_drawer" / image_path.parent.name / image_path.name
    meta = json.loads(offline.with_suffix(".json").read_text())
    image = np.asarray(np.load(image_path)["image"], dtype=np.uint8)
    instruction = meta["instruction"]
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, TASK, {}, False))
    baseline_trace, baseline_record = run(build(base, None), image, instruction)
    traces = {}; records = {}
    for arm in ARMS:
        traces[arm], records[arm] = run(build(base, arm), image, instruction)
    original = traces["original"]
    audit = {
        "baseline_original_mask_exact": baseline_trace["selected_token_ids"] == original["selected_token_ids"],
        "baseline_original_final_tokens_exact": baseline_trace["final_token_ids"] == original["final_token_ids"],
        "baseline_original_guided_action_exact": baseline_trace["guided_action"] == original["guided_action"],
        "clean_actions_equal_all_arms": all(row["clean_action"] == original["clean_action"] for row in traces.values()),
        "m_equal_all_arms": len({row["m_t"] for row in traces.values()}) == 1,
        "masks_exact_m_all_arms": all(len(set(row["selected_token_ids"])) == row["m_t"] for row in traces.values()),
        "common_core_exact_all_arms": all(row["core_token_ids"] == original["core_token_ids"] for row in traces.values()),
        "candidate_membership_exact_all_arms": all(row["candidate_pool_membership_exact"] for row in traces.values()),
        "action_attention_recorded_all_arms": all(
            "teacher_forced_vs_autoregressive_changed_dims" in row for row in traces.values()
        ),
        "prompt_zero_based_l11_all_arms": all(row["attention_layers"] == [11] for row in traces.values()),
        "action_zero_based_16_31_all_arms": all(row["action_attention_layers"] == list(range(16, 32)) for row in traces.values()),
        "action_dims_0_5_all_arms": all(row["action_attention_dimensions"] == list(range(6)) for row in traces.values()),
        "visual_features_exact_all_arms": all(row["feature_equal"] for row in traces.values()),
        "outside_features_exact_all_arms": all(row["non_target_bit_identical"] for row in traces.values()),
    }
    audit["technical_pass"] = all(audit.values())
    result = {
        "task": TASK, "seed": int(meta["seed"]), "control_step": int(meta["control_step"]),
        "instruction": instruction, "source_image": str(image_path), "audit": audit,
        "m": original["m_t"], "r": original["supplement_count_r"],
        "max_teacher_clean_logits_abs_diff": max(
            row["teacher_forced_clean_action_logits_max_abs_diff"] for row in traces.values()
        ),
        "arms": {arm: {
            "selected_token_ids": row["selected_token_ids"],
            "supplement_token_ids": row["supplement_token_ids"],
            "actual_replacements_vs_original": row["actual_replacements_vs_original"],
            "guided_changed_dims": row["guided_changed_dims"],
        } for arm, row in traces.items()},
    }
    atomic_json(ARTIFACT / "preflight/report.json", result)
    if not audit["technical_pass"]: raise RuntimeError(json.dumps(audit, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
