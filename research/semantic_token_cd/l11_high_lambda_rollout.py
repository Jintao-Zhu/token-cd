"""Closed-loop rollout for pure-L11 and DTP-positive high-lambda arms."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.dtp_l11_cd_policy import initialize_dtp_l11_cd
from research.semantic_token_cd.l11_high_lambda_protocol import (
    ARMS, ARM_CONFIG, ARTIFACT, CANONICAL, PCD_SOURCE, PROTOCOL, TASKS, atomic_json,
)
from research.semantic_token_cd.prompt_action_rerank_rollout import file_sha, parse_seeds, run_loop
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.xswap_rollout import make_environment


def initialize_pure_l11(policy, task: str, lambd: float) -> None:
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, float(lambd))
    policy.beta = 0.0
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = (11,)
    policy.selection_count = None
    policy.selection_top_p = None
    policy.selector_instruction = None
    policy.selector_contrast_instruction = None
    policy.selector_difference_eta = None
    policy.selector_difference_kind = None
    policy.complement_arm = None
    policy.prompt_action_rerank_multiplier = None
    policy.prompt_action_bounded_mode = None
    policy.prompt_action_full_joint = False
    policy.save_prompt_attention = True


def build_policy(base, task: str, arm: str):
    cfg = ARM_CONFIG[arm]
    policy = copy.copy(base)
    if cfg["family"] == "pure_l11":
        initialize_pure_l11(policy, task, cfg["lambda"])
    else:
        initialize_dtp_l11_cd(policy, task, cfg["lambda"])
    return policy


def write_arrays(path: Path, records: list[dict], actions: np.ndarray) -> None:
    payload = {"executed_actions": actions}
    keys = (
        "positive", "negative", "final", "selected_mask", "reference_shr_mask",
        "prompt_attention", "prompt_attention_l11", "dtp_pruned_mask",
    )
    for key in keys:
        if records and all(key in row for row in records):
            payload[key] = np.stack([row[key] for row in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--gpu", required=True, type=int, choices=tuple(range(8)))
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"invalid arms: {arms}")
    artifact = args.artifact.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    env, env_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    policy_sha = file_sha(Path(__file__))
    for seed in parse_seeds(args.seeds):
        snapshot_path = CANONICAL / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical_sha = snapshot_sha(snapshot)
        hashes = []
        for arm in arms:
            cfg = ARM_CONFIG[arm]
            out = artifact / "closed_loop/episodes" / args.task / arm
            summary_path = out / f"episode_{seed:03d}_summary.json"
            arrays_path = out / f"episode_{seed:03d}_arrays.npz"
            video_path = artifact / "closed_loop/videos" / args.task / arm / f"episode_{seed:03d}.mp4"
            if summary_path.exists() and arrays_path.exists() and video_path.exists():
                continue
            observation, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            instruction = env.unwrapped.get_language_instruction()
            policy = build_policy(base, args.task, arm)
            policy._episode_seed = seed
            policy._selector_step = 0
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, actions, infos = run_loop(env, policy, instruction, observation, video_path)
            runtime = time.monotonic() - started
            trace = policy._episode_trace
            common = {
                "trace_nonempty": bool(trace),
                "all_lambda_locked": all(abs(float(row.get("lambda", -1)) - cfg["lambda"]) < 1e-12 for row in trace),
                "all_l11_matched": all(row.get("attention_layers") == [11] and row.get("coverage_exact") is True for row in trace),
                "all_guided_prefix": all(row.get("guided_prefix") is True for row in trace),
                "all_gripper_positive": all(
                    row.get("gripper_positive_exact", True) is True for row in trace
                ),
                "all_non_target_equal": all(row.get("non_target_bit_identical") is True for row in trace),
                "all_reconstruction_finite": all(row.get("reconstruction_finite") is True for row in trace),
            }
            if cfg["family"] == "dtp_l11":
                common["all_dtp_fixed"] = all(
                    row.get("dtp_fixed_mask") is True and row.get("dtp_layer") == 11
                    and row.get("dtp_k") == 64 and abs(float(row.get("dtp_tau", -1)) - 0.5) < 1e-12
                    for row in trace
                )
                common["all_dtp_prefix"] = all(row.get("prefix_source") == "DTP positive greedy tokens" for row in trace)
            else:
                common["all_pure_l11"] = all("dtp_fixed_mask" not in row for row in trace)
            common["technical_pass"] = bool(trace) and all(common.values())
            if not common["technical_pass"]:
                raise RuntimeError(f"high-lambda audit failure: {common}")
            mean = lambda key: float(np.mean([row[key] for row in trace]))
            summary = {
                "protocol": PROTOCOL, "task": args.task, "seed": seed, "arm": arm,
                "family": cfg["family"], "lambda": cfg["lambda"], "instruction": instruction,
                "success": bool(result.get("success", False)), "result": jsonable(result),
                "control_steps": len(infos), "runtime_seconds": runtime,
                "worker_id": args.worker_id, "environment_id": env_id,
                "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                "canonical_snapshot_sha256": canonical_sha, "policy_code_sha256": policy_sha,
                "mean_m": mean("m_t"),
                "mean_feature_perturbation_norm": mean("feature_perturbation_norm"),
                "mean_centered_residual_norm": mean("centered_logit_residual_norm"),
                "mean_guided_changed_dims": mean("guided_changed_dims"),
                "dtp_trigger_rate": mean("dtp_triggered") if cfg["family"] == "dtp_l11" else None,
                "selector_trace": jsonable(trace),
                "video_path": str(video_path.relative_to(artifact)), "video_sha256": file_sha(video_path),
                **common,
            }
            write_arrays(arrays_path, policy._episode_logits, actions)
            atomic_json(summary_path, summary)
            hashes.append((canonical_sha, state_sha, rgb_sha))
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm,
                              "success": summary["success"], "seconds": round(runtime, 1)}), flush=True)
        if len(set(hashes)) not in (0, 1):
            raise RuntimeError(f"within-seed snapshot mismatch: {args.task} seed={seed}")
    env.close()


if __name__ == "__main__":
    main()

