"""Equivalence gates for the P50 + Prompt-CD factorial D arm."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
ARTIFACT = REPO / "artifacts/vla_pruner_x_promptcd_v1"
for path in (REPO, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def flatten_executed(actions) -> np.ndarray:
    action = actions[0] if isinstance(actions, list) else actions
    return np.concatenate([
        np.asarray(action["world_vector"], dtype=np.float64),
        np.asarray(action["rot_axangle"], dtype=np.float64),
        np.asarray(action["gripper"], dtype=np.float64),
    ])


def build_prompt(base, task):
    from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
    from research.semantic_token_cd.semantic_recon_rollout import _init_common

    policy = copy.copy(base)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, 0.5)
    policy.beta = 0.0
    policy.selector_mode = "prompt_attention"
    policy.attention_layers = (11,)
    return policy


def build_combined(base, task, lambd=0.5, pruner=True, cd=True):
    from research.semantic_token_cd.semantic_recon_rollout import _init_common
    from research.semantic_token_cd.vla_pruner_promptcd_policy import initialize_p50_prompt_cd

    policy = copy.copy(base)
    _init_common(policy, lambd)
    initialize_p50_prompt_cd(policy, task, lambd)
    policy.pruner_enabled = pruner
    policy.cd_enabled = cd
    return policy


def build_p50(base, task):
    from research.semantic_token_cd.vla_pruner_policy import build_vla_pruner_policy
    return build_vla_pruner_policy(base, "vla_pruner_prune50", task)


def run_sequence(policy, image, instruction, steps=4):
    policy.reset(instruction, seed=0)
    rows = []
    for step in range(steps):
        raw, actions, returned_meta = policy.step(image, None, instruction)
        meta = policy._episode_trace[-1] if policy._episode_trace else returned_meta
        tokens = meta.get("final_token_ids", meta.get("token_ids"))
        rows.append({
            "step": step,
            "token_ids": list(tokens),
            "raw": np.concatenate([
                np.asarray(raw["world_vector"], dtype=np.float64),
                np.asarray(raw["rotation_delta"], dtype=np.float64),
                np.asarray(raw["open_gripper"], dtype=np.float64),
            ]),
            "executed": flatten_executed(actions),
            "history_before": meta.get("history_len_before", meta.get("history_len")),
            "history_after": meta.get("history_len_after", meta.get("history_after")),
            "keep": meta.get("p50_keep_token_ids", meta.get("kept_visual_token_ids")),
        })
    return rows


def compare(left, right):
    result = {
        "token_ids_equal": True,
        "raw_max_abs": 0.0,
        "executed_max_abs": 0.0,
        "keep_equal": True,
        "per_step": [],
    }
    for lrow, rrow in zip(left, right):
        token_equal = lrow["token_ids"] == rrow["token_ids"]
        raw_abs = float(np.max(np.abs(lrow["raw"] - rrow["raw"])))
        executed_abs = float(np.max(np.abs(lrow["executed"] - rrow["executed"])))
        keep_equal = lrow["keep"] == rrow["keep"]
        result["token_ids_equal"] &= token_equal
        result["raw_max_abs"] = max(result["raw_max_abs"], raw_abs)
        result["executed_max_abs"] = max(result["executed_max_abs"], executed_abs)
        result["keep_equal"] &= keep_equal
        result["per_step"].append({
            "step": lrow["step"], "token_ids_equal": token_equal,
            "raw_max_abs": raw_abs, "executed_max_abs": executed_abs,
            "keep_equal": keep_equal,
        })
    result["passed"] = bool(
        result["token_ids_equal"]
        and result["raw_max_abs"] == 0.0
        and result["executed_max_abs"] == 0.0
        and result["keep_equal"]
    )
    return result


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    task = "google_robot_open_drawer"
    instruction = "open top drawer"
    sample = np.load(
        REPO / "artifacts/prompt_attn_layer_selection_v1/states/"
        "google_robot_open_drawer/seed_000/step_011.npz"
    )
    image = sample["image"]
    checkpoint = str(SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, task, {}, False))

    gates = {}
    p50 = run_sequence(build_p50(base, task), image, instruction)
    combined_lambda0 = run_sequence(build_combined(base, task, lambd=0.0), image, instruction)
    gates["gate1_lambda0_combined_equals_p50"] = compare(combined_lambda0, p50)

    prompt = run_sequence(build_prompt(base, task), image, instruction)
    combined_pruner_off = run_sequence(
        build_combined(base, task, lambd=0.5, pruner=False), image, instruction
    )
    prompt_comparison = compare(combined_pruner_off, prompt)
    prompt_comparison.pop("keep_equal")
    prompt_comparison["passed"] = bool(
        prompt_comparison["token_ids_equal"]
        and prompt_comparison["raw_max_abs"] == 0.0
        and prompt_comparison["executed_max_abs"] == 0.0
    )
    gates["gate2_pruner_off_equals_prompt_cd"] = prompt_comparison

    p50_again = run_sequence(build_p50(base, task), image, instruction)
    combined_cd_off = run_sequence(
        build_combined(base, task, lambd=0.5, cd=False), image, instruction
    )
    gates["gate3_cd_off_equals_p50"] = compare(combined_cd_off, p50_again)
    payload = {
        "protocol": "VLA_PRUNER_X_PROMPT_CD_V1",
        "task": task,
        "instruction": instruction,
        "steps": 4,
        "active_pruning_checked_at_step": 3,
        "gates": gates,
        "all_passed": all(item["passed"] for item in gates.values()),
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "EQUIVALENCE_GATES.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
