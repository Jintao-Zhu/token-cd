"""One-state equivalence and lambda-scaling gates for DTP/L11 CD."""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from research.semantic_token_cd.dtp_closed_loop_policy import build_policy as build_dtp
from research.semantic_token_cd.dtp_l11_cd_policy import initialize_dtp_l11_cd
from research.semantic_token_cd.dtp_l11_cd_protocol import ARTIFACT, PCD_SOURCE, atomic_json

TASK = "google_robot_open_drawer"


def run(policy, image, instruction):
    policy.reset(instruction, seed=0)
    policy._episode_trace = []
    policy._episode_logits = []
    raw, action, _ = policy.step(image, None, instruction, proprio=np.zeros(8))
    record = policy._episode_logits[-1] if policy._episode_logits else None
    return raw, action, policy._episode_trace[-1], record


def flattened(action):
    value = action[0] if isinstance(action, list) else action
    return np.concatenate([value["world_vector"], value["rot_axangle"], value["gripper"]])


def main() -> None:
    gpu = int(os.environ.get("DTP_L11_PREFLIGHT_GPU", "2"))
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
    dtp_raw, dtp_action, dtp_trace, _ = run(build_dtp(base, TASK, "v2_fix_k64_t05"), image, instruction)
    outputs = {}
    for value in (0.0, 0.10, 0.25, 0.50):
        policy = copy.copy(base)
        initialize_dtp_l11_cd(policy, TASK, value)
        outputs[value] = run(policy, image, instruction)

    zero_raw, zero_action, _, zero_record = outputs[0.0]
    checks = {
        "lambda0_raw_equals_dtp": all(
            np.array_equal(np.asarray(zero_raw[key]), np.asarray(dtp_raw[key])) for key in zero_raw
        ),
        "lambda0_executed_equals_dtp": np.array_equal(flattened(zero_action), flattened(dtp_action)),
        "positive_logits_equal_across_lambdas": all(
            np.array_equal(outputs[v][3]["positive"], zero_record["positive"]) for v in (0.10, 0.25, 0.50)
        ),
        "negative_logits_equal_across_lambdas": all(
            np.array_equal(outputs[v][3]["negative"], zero_record["negative"]) for v in (0.10, 0.25, 0.50)
        ),
        "selected_mask_equal_across_lambdas": all(
            np.array_equal(outputs[v][3]["selected_mask"], zero_record["selected_mask"]) for v in (0.10, 0.25, 0.50)
        ),
        "dtp_mask_equal_across_lambdas": all(
            np.array_equal(outputs[v][3]["dtp_pruned_mask"], zero_record["dtp_pruned_mask"]) for v in (0.10, 0.25, 0.50)
        ),
    }
    positive = zero_record["positive"]
    negative = zero_record["negative"]
    for value in (0.10, 0.25, 0.50):
        expected = positive.copy()
        expected[:6] = (1.0 + value) * positive[:6] - value * negative[:6]
        max_abs = float(np.max(np.abs(outputs[value][3]["final"][:6] - expected[:6])))
        checks[f"lambda_{value}_fusion_max_abs"] = max_abs
        # Records use the repository's historical float16 log format while
        # execution fuses float32 logits.  Permit at most one observed fp16
        # quantization step; the exact runtime formula is applied in policy.
        checks[f"lambda_{value}_fusion_exact"] = bool(max_abs <= 0.015625)
        checks[f"lambda_{value}_gripper_exact"] = bool(np.array_equal(
            outputs[value][3]["final"][6], positive[6]
        ))
    checks["technical_pass"] = all(
        value for key, value in checks.items() if not key.endswith("_max_abs")
    )
    report = {"protocol": "DTP_FIXED_POSITIVE_L11_MATCHED_NEGATIVE_LAMBDA_SWEEP_V1",
              "task": TASK, "instruction": instruction, "checks": checks}
    atomic_json(ARTIFACT / "preflight" / "GATES.json", report)
    print(json.dumps(report, indent=2))
    if not checks["technical_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
