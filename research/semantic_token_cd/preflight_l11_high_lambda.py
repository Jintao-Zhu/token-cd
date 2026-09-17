"""One-state branch-invariance gates for the high-lambda extension."""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from research.semantic_token_cd.l11_high_lambda_protocol import ARTIFACT, PCD_SOURCE, atomic_json
from research.semantic_token_cd.l11_high_lambda_rollout import initialize_pure_l11
from research.semantic_token_cd.dtp_l11_cd_policy import initialize_dtp_l11_cd

TASK = "google_robot_open_drawer"


def run(policy, image, instruction):
    policy._episode_seed = 0
    policy._selector_step = 0
    policy.reset(instruction, seed=0)
    policy._episode_trace = []
    policy._episode_logits = []
    raw, action, _ = policy.step(image, None, instruction, proprio=np.zeros(8))
    return raw, action, policy._episode_trace[-1], policy._episode_logits[-1]


def main() -> None:
    gpu = int(os.environ.get("L11_HIGH_PREFLIGHT_GPU", "1"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    sample = np.load(ARTIFACT.parent / "prompt_attn_layer_selection_v1/states/google_robot_open_drawer/seed_000/step_011.npz")
    image = sample["image"]
    instruction = "open top drawer"
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, TASK, {}, False))
    families = {}
    for family, init in (("pure_l11", initialize_pure_l11), ("dtp_l11", initialize_dtp_l11_cd)):
        outputs = {}
        for value in (0.5, 0.6, 0.75):
            policy = copy.copy(base)
            init(policy, TASK, value)
            outputs[value] = run(policy, image, instruction)
        ref = outputs[0.5][3]
        checks = {
            "positive_logits_invariant": all(np.array_equal(outputs[v][3]["positive"], ref["positive"]) for v in (0.6, 0.75)),
            "negative_logits_invariant": all(np.array_equal(outputs[v][3]["negative"], ref["negative"]) for v in (0.6, 0.75)),
            "selected_mask_invariant": all(np.array_equal(outputs[v][3]["selected_mask"], ref["selected_mask"]) for v in (0.6, 0.75)),
            "gripper_action_invariant": all(outputs[v][2]["final_token_ids"][6] == outputs[v][2]["positive_token_ids"][6] for v in (0.5, 0.6, 0.75)),
            "lambda_locked": all(abs(float(outputs[v][2]["lambda"]) - v) < 1e-12 for v in (0.5, 0.6, 0.75)),
        }
        if family == "dtp_l11":
            checks["dtp_mask_invariant"] = all(np.array_equal(outputs[v][3]["dtp_pruned_mask"], ref["dtp_pruned_mask"]) for v in (0.6, 0.75))
            checks["dtp_fixed_config"] = all(outputs[v][2]["dtp_fixed_mask"] is True and outputs[v][2]["dtp_k"] == 64 for v in (0.5, 0.6, 0.75))
        checks["technical_pass"] = all(checks.values())
        families[family] = checks
    report = {"protocol": "L11_MATCHED_AND_DTP_POSITIVE_HIGH_LAMBDA_V1",
              "task": TASK, "instruction": instruction, "families": families,
              "technical_pass": all(x["technical_pass"] for x in families.values())}
    atomic_json(ARTIFACT / "preflight/GATES.json", report)
    print(json.dumps(report, indent=2))
    if not report["technical_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
