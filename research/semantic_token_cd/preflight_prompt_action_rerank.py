"""One-state fail-closed GPU audit for PA-Rerank; does not run an episode."""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from research.semantic_token_cd.prompt_action_rerank_protocol import (
    ACTION_LAYERS, ARTIFACT, CANDIDATE_MULTIPLIER, LAMBDA, PCD_SOURCE,
    PROMPT_LAYERS,
)
from research.semantic_token_cd.prompt_attn_shr_policy import (
    PromptAttentionSHRInference, construct_prompt_action_rerank,
)
from research.semantic_token_cd.prompt_action_complement_protocol import STATE_SOURCE
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.prompt_action_rerank_protocol import atomic_json

TASK = "google_robot_open_drawer"


def build(base, rerank: bool):
    policy = copy.copy(base)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[TASK]
    policy.attention_layers = PROMPT_LAYERS
    policy.action_attention_layers = ACTION_LAYERS
    policy.selection_count = None
    policy.selection_top_p = None
    policy.save_prompt_attention = True
    policy.selector_instruction = None
    policy.selector_contrast_instruction = None
    policy.selector_difference_eta = None
    policy.complement_arm = None
    policy.prompt_action_rerank_multiplier = CANDIDATE_MULTIPLIER if rerank else None
    return policy


def one_step(policy, image, instruction):
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    policy.reset(instruction, seed=0)
    policy.step(image, None, instruction, proprio=np.zeros(8, dtype=np.float64))
    return policy._episode_trace[-1], policy._episode_logits[-1]


def main() -> None:
    parser_gpu = int(os.environ.get("PA_RERANK_PREFLIGHT_GPU", "2"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parser_gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    image_path = STATE_SOURCE / "runs/emitted_states/google_robot_open_drawer/seed_000/step_0013.npz"
    if not image_path.exists():
        image_path = sorted((STATE_SOURCE / "runs/emitted_states/google_robot_open_drawer").glob("seed_*/*.npz"))[0]
    offline = STATE_SOURCE / "runs/offline/google_robot_open_drawer" / image_path.parent.name / image_path.name
    instruction = json.loads(offline.with_suffix(".json").read_text())["instruction"]
    image = np.asarray(np.load(image_path)["image"], dtype=np.uint8)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, TASK, {}, False))
    baseline_trace, baseline_record = one_step(build(base, False), image, instruction)
    rerank_trace, rerank_record = one_step(build(base, True), image, instruction)
    emulated = construct_prompt_action_rerank(
        rerank_record["prompt_attention"], rerank_record["prompt_attention"],
        rerank_trace["m_t"], CANDIDATE_MULTIPLIER,
    )
    checks = {
        "same_m_as_l11_matched": rerank_trace["m_t"] == baseline_trace["m_t"],
        "exact_final_k": len(set(rerank_trace["selected_token_ids"])) == rerank_trace["m_t"],
        "selected_inside_prompt_pool": rerank_trace["rerank_subset_of_candidate_pool"] is True,
        "candidate_size_exact": rerank_trace["candidate_pool_size"] == min(
            256, 3 * rerank_trace["m_t"]
        ),
        "clean_positive_logits_exact": np.array_equal(
            baseline_record["positive"], rerank_record["positive"]
        ),
        "clean_positive_action_exact": baseline_trace["clean_action"] == rerank_trace["clean_action"],
        "prompt_as_second_score_recovers_matched_mask": (
            emulated["selected"] == baseline_trace["selected_token_ids"]
        ),
        "prompt_zero_based_l11": rerank_trace["attention_layers"] == [11],
        "action_zero_based_l16_31": rerank_trace["action_attention_layers"] == list(range(16, 32)),
        "action_clean_positive_dims_0_5": rerank_trace["action_attention_dimensions"] == list(range(6)),
        "same_harmonic_and_prefix_path": (
            rerank_trace["beta"] == 0.0 and rerank_trace["lambda"] == 0.5
            and rerank_trace["guided_prefix"] is True
            and rerank_trace["non_target_bit_identical"] is True
        ),
    }
    checks["technical_pass"] = all(checks.values())
    report = {
        "task": TASK, "instruction": instruction, "source_image": str(image_path),
        "m": rerank_trace["m_t"],
        "candidate_pool_size": rerank_trace["candidate_pool_size"],
        "overlap_ratio": rerank_trace["rerank_overlap_ratio"],
        "jaccard": rerank_trace["rerank_jaccard"],
        "checks": checks,
    }
    atomic_json(ARTIFACT / "preflight/report.json", report)
    if not checks["technical_pass"]:
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
