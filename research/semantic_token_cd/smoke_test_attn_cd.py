"""Smoke test for the H100: verify the attention-CD pipeline runs end-to-end.

Two-arm (vanilla + attn_semantic K8 L8-15) capture-once-reuse on a small number
of seeds. Self-contained: it captures fresh snapshots per seed (no frozen
reference), so it works on a machine with no prior artifacts.

Success criteria (printed as SMOKE_TEST_PASS):
  * the environment + OpenVLA-7B checkpoint load,
  * a rollout completes for BOTH arms on every seed,
  * the technical audit passes on every attn_semantic episode:
      - all_visual_features_bit_identical  (CD subtraction is "clean"),
      - hook_calls == 56 per control step   (mask hits exactly the K8 x 7 keys
        across layers 8..15).

Run (from the repo root, with PCD_SOURCE + venv configured):
  PY=task1/.venvs/openvla-ar/bin/python            # H100: your venv python
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" research/semantic_token_cd/smoke_test_attn_cd.py \
      --task google_robot_close_drawer --seeds 3 --gpu 0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_policy import (
    SpatialGridAttentionCDInference,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
)

TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
KMEANS_K = 8
LAYER_START = 8
LAYER_END = 16  # layers 8..15 inclusive
EXPECTED_HOOK_CALLS = 7 * (LAYER_END - LAYER_START)  # 56


def build_vanilla_policy(base):
    policy = copy.copy(base)
    policy.__class__ = AuditedVanillaInference
    policy._episode_trace = []
    policy._episode_logits = []
    return policy


def build_attention_policy(base):
    policy = copy.copy(base)
    policy.__class__ = SpatialGridAttentionCDInference
    policy.alpha = 0.5
    policy.lambd = 0.5
    policy.kmeans_K = KMEANS_K
    policy.kmeans_seed = 0
    policy.selection_mode = "semantic"
    policy.spatial_selection_mode = "semantic_hard"
    policy.attention_layer_start = LAYER_START
    policy.attention_layer_end = LAYER_END
    policy.attention_mask_value = -1e4
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="google_robot_close_drawer")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**policy_config)
    policies = {
        "vanilla": build_vanilla_policy(base),
        "attn_semantic": build_attention_policy(base),
    }

    audit_failures = []
    results = []
    for seed in range(args.seeds):
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        for arm in ("vanilla", "attn_semantic"):
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(
                env, policy, instruction, obs
            )
            trace = jsonable(policy._episode_trace)
            if arm == "attn_semantic":
                feature_equal = all(step["feature_equal"] for step in trace)
                hook_pass = all(
                    step.get("negative_truncated", False)
                    or step["attention_mask"]["hook_calls"] == EXPECTED_HOOK_CALLS
                    for step in trace
                )
                ok = feature_equal and hook_pass
                if not ok:
                    audit_failures.append(
                        {
                            "seed": seed,
                            "feature_equal": feature_equal,
                            "hook_pass": hook_pass,
                        }
                    )
            results.append(
                {"seed": seed, "arm": arm, "success": bool(result["success"])}
            )
            print(
                json.dumps(
                    {
                        "task": args.task,
                        "seed": seed,
                        "arm": arm,
                        "success": bool(result["success"]),
                        "steps": steps,
                        "technical_ok": (
                            None if arm == "vanilla" else ok
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    passed = not audit_failures
    print(json.dumps({"SMOKE_TEST_PASS": passed, "audit_failures": audit_failures}))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
