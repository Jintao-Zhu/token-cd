"""Closed-loop causal test of entity-conditioned L11 intervention budgets."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_attn_l11_budget_causal_rollout import (
    code_version,
    make_environment,
    parse_seeds,
)
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import (
    KMEANS_K,
    KMEANS_SEED,
    LAMBDA,
    atomic_json,
    finite_mean,
    load_reference,
    write_arrays,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_L11_BUDGET_PROVENANCE_CLOSED_LOOP_V1"
ARMS = ("wrong_entity", "random_cluster", "within_task_shuffle")
WRONG_ENTITIES = {
    "google_robot_open_drawer": ("coke can",),
    "google_robot_close_drawer": ("redbull can",),
    "google_robot_pick_coke_can": ("top drawer",),
    "google_robot_move_near": ("top drawer", "apple"),
}


def build_policies(base, task: str, arms: tuple[str, ...]) -> dict:
    policies = {}
    for arm in arms:
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, LAMBDA)
        policy.beta = 0.0
        policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,)
        policy.selection_count = None
        policy.selection_top_p = None
        policy.selection_budget_schedule = None
        policy.selection_budget_scale = 1.0
        policy.selection_budget_label = None
        policy.selection_budget_source = arm if arm != "within_task_shuffle" else "matched"
        policy.selection_budget_entities = WRONG_ENTITIES[task] if arm == "wrong_entity" else None
        policy.save_prompt_attention = False
        policies[arm] = policy
    return policies


def audit(trace: list[dict], arm: str, expected_schedule: list[int] | None) -> dict:
    checks = {
        "technical_nonempty": bool(trace),
        "all_feature_equal": bool(trace) and all(item.get("feature_equal") is True for item in trace),
        "all_guided_prefix": bool(trace) and all(item.get("guided_prefix") is True for item in trace),
        "all_reconstruction_finite": bool(trace) and all(item.get("reconstruction_finite") is True for item in trace),
        "all_lambda_locked": bool(trace) and all(abs(float(item.get("lambda", -1)) - .5) < 1e-12 for item in trace),
        "all_beta_zero": bool(trace) and all(abs(float(item.get("beta", -1))) < 1e-12 for item in trace),
        "all_coverage_exact": bool(trace) and all(item.get("coverage_exact") is True for item in trace),
        "all_non_target_bit_identical": bool(trace) and all(item.get("non_target_bit_identical") is True for item in trace),
        "all_l11_locked": bool(trace) and all(item.get("attention_layers") == [11] for item in trace),
        "all_post_softmax": bool(trace) and all(item.get("attention_post_softmax") is True for item in trace),
        "all_visual_keys_locked": bool(trace) and all(item.get("visual_key_indices") == [1, 256] for item in trace),
        "all_gripper_clean": bool(trace) and all(
            item["positive_token_ids"][6] == item["final_token_ids"][6] for item in trace
        ),
    }
    if arm == "within_task_shuffle":
        checks.update({
            "schedule_counts_exact": bool(trace) and all(
                item["actual_selected_count"] == expected_schedule[index]
                for index, item in enumerate(trace)
            ),
            "schedule_kmeans_bypassed": bool(trace) and all(
                not item.get("reference_shr_token_ids") for item in trace
            ),
            "schedule_label_exact": bool(trace) and all(
                item.get("budget_schedule_label") == arm for item in trace
            ),
            "budget_source_exact": bool(trace) and all(
                item.get("budget_source") == "frozen_schedule" for item in trace
            ),
        })
    else:
        checks.update({
            "budget_source_exact": bool(trace) and all(
                item.get("budget_source") == arm for item in trace
            ),
            "kmeans_budget_nonempty": bool(trace) and all(
                item.get("reference_shr_token_ids") for item in trace
            ),
            "budget_groups_nonempty": bool(trace) and all(
                item.get("selected_group_ids") for item in trace
            ),
        })
        if arm == "wrong_entity":
            expected_entities = list(WRONG_ENTITIES[trace[0]["task_key"]]) if trace and "task_key" in trace[0] else None
            checks["wrong_entities_present"] = bool(trace) and all(item.get("budget_entities") for item in trace)
        if arm == "random_cluster":
            checks["random_group_cardinality_exact"] = bool(trace) and all(
                len(item.get("selected_group_ids", [])) == len(item.get("matched_budget_group_ids", []))
                for item in trace
            )
    checks["technical_pass"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--schedule-file", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"invalid arms: {arms}")
    seeds = parse_seeds(args.seeds)
    schedules = json.loads(args.schedule_file.read_text())["schedules"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    environment, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task, arms)
    version = code_version([Path(__file__), Path(__file__).with_name("prompt_attn_shr_policy.py")])

    for seed in seeds:
        with (source / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(source, args.task, seed)
        expected = (
            reference["canonical_snapshot_sha256"], reference["initial_state_sha256"],
            reference["initial_rgb_sha256"],
        )
        if snapshot_sha(snapshot) != expected[0]:
            raise RuntimeError("canonical snapshot mismatch")
        for arm, policy in policies.items():
            output = artifact / "episodes" / args.task / arm
            summary_path = output / f"episode_{seed:03d}_summary.json"
            arrays_path = output / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            observation, state_sha, rgb_sha = restore_snapshot(environment, seed, snapshot)
            if (state_sha, rgb_sha) != expected[1:]:
                raise RuntimeError("restored snapshot mismatch")
            instruction = environment.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError("instruction mismatch")
            schedule = schedules["within_task_shuffle"][args.task][str(seed)] if arm == "within_task_shuffle" else None
            policy.selection_budget_schedule = tuple(schedule) if schedule is not None else None
            policy.selection_budget_label = arm if schedule is not None else None
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(environment, policy, instruction, observation)
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, arm, schedule)
            output.mkdir(parents=True, exist_ok=True)
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "evaluation_seed": seed,
                "episode_id": seed,
                "arm": arm,
                "attention_layers": [11],
                "instruction": instruction,
                "success": bool(result["success"]),
                "result": jsonable(result),
                "failure_reason": reason,
                "control_steps": steps,
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": expected[0],
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "lambda": LAMBDA,
                "beta": 0.0,
                "budget_source": arm,
                "wrong_entities": list(WRONG_ENTITIES[args.task]) if arm == "wrong_entity" else None,
                "budget_schedule_arm": arm if schedule is not None else None,
                "kmeans_K": None if schedule is not None else KMEANS_K,
                "kmeans_seed": None if schedule is not None else KMEANS_SEED,
                "code_version": version,
                "arrays_file": arrays_path.name,
                "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(item.get("actual_selected_count") for item in trace),
                "mean_feature_perturbation_relative": finite_mean(item.get("feature_perturbation_relative") for item in trace),
                "mean_centered_logit_residual_norm": finite_mean(item.get("centered_logit_residual_norm") for item in trace),
                "mean_guided_change_ratio": finite_mean(item.get("guided_change_ratio") for item in trace),
                "selector_trace": trace,
                **checks,
            }
            atomic_json(summary_path, summary)
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": summary["success"],
                "runtime_seconds": round(runtime, 2),
                "technical_pass": summary["technical_pass"],
            }), flush=True)


if __name__ == "__main__":
    main()
