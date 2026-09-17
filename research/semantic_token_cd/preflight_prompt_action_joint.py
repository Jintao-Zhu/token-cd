"""One-state fail-closed audit for PA-A10 and PA-Joint10."""
from __future__ import annotations

import json
import os

import numpy as np

from research.semantic_token_cd.prompt_action_complement_protocol import STATE_SOURCE
from research.semantic_token_cd.prompt_action_joint_protocol import (
    ARMS, ARTIFACT, PCD_SOURCE, atomic_json,
)
from research.semantic_token_cd.prompt_action_joint_rollout import build_policy
from research.semantic_token_cd.prompt_action_rerank_rollout import build_policy as build_baseline
from research.semantic_token_cd.prompt_attn_shr_policy import construct_prompt_action_bounded

TASK = "google_robot_open_drawer"


def run(policy, image, instruction):
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    policy.reset(instruction, seed=0)
    policy.step(image, None, instruction, proprio=np.zeros(8, dtype=np.float64))
    return policy._episode_trace[-1], policy._episode_logits[-1]


def main() -> None:
    gpu = int(os.environ.get("PA_JOINT_PREFLIGHT_GPU", "2"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
    # The imported PA-Rerank builder enables full rerank; disable it to obtain
    # the exact historical L11-Matched path for this audit.
    baseline_policy = build_baseline(base, TASK)
    baseline_policy.prompt_action_rerank_multiplier = None
    baseline_trace, baseline_record = run(baseline_policy, image, instruction)
    traces, records = {}, {}
    for arm in ARMS:
        traces[arm], records[arm] = run(build_policy(base, TASK, arm), image, instruction)
    m = baseline_trace["m_t"]
    emulated = construct_prompt_action_bounded(
        records["pa_joint10"]["prompt_attention"],
        records["pa_joint10"]["prompt_attention"], m, "joint",
    )
    checks = {
        "same_m_all_arms": all(x["m_t"] == m for x in traces.values()),
        "exact_k_all_arms": all(len(set(x["selected_token_ids"])) == m for x in traces.values()),
        "core_90pct_preserved": all(x["bounded_core_preserved"] is True for x in traces.values()),
        "replacement_cap_all_arms": all(x["bounded_max_replacements_respected"] is True for x in traces.values()),
        "inside_prompt_top2k": all(x["bounded_subset_of_candidate_pool"] is True for x in traces.values()),
        "clean_logits_exact": all(np.array_equal(records[a]["positive"], baseline_record["positive"]) for a in ARMS),
        "clean_action_exact": all(traces[a]["clean_action"] == baseline_trace["clean_action"] for a in ARMS),
        "prompt_equal_action_recovers_l11": emulated["selected"] == baseline_trace["selected_token_ids"],
        "locked_attention_layers": all(x["attention_layers"] == [11] and x["action_attention_layers"] == list(range(16, 32)) for x in traces.values()),
        "locked_downstream": all(x["beta"] == 0.0 and x["lambda"] == 0.5 and x["guided_prefix"] is True and x["non_target_bit_identical"] is True for x in traces.values()),
    }
    checks["technical_pass"] = all(checks.values())
    report = {"task": TASK, "instruction": instruction, "m": m,
              "arms": {a: {"overlap": traces[a]["bounded_overlap_ratio"],
                            "selected": traces[a]["selected_token_ids"]} for a in ARMS},
              "checks": checks}
    atomic_json(ARTIFACT / "preflight/report.json", report)
    if not checks["technical_pass"]:
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
