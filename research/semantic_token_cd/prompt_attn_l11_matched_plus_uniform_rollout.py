"""Closed-loop: own-state L11 matched mask UNION a 16-token spatially uniform grid.

Purpose: matched's budget alone is compared against matched + an extra evenly
strided 16-token grid drawn from the WHOLE 16x16 patch grid (so those tokens may
already be inside matched's own selection; this is NOT "+16 unique tokens").

Everything else is locked: L11 readout, own-state matched budget, L11 ranking,
lambda=0.5, harmonic beta=0 reconstruction, shared greedy prefix, gripper dim 6.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_attn_shr_policy_floor import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_layer_rollout import SUPPORTED_TASKS, make_environment
from research.semantic_token_cd.prompt_attn_l11_count_rollout import parse_count_sweep_seeds
from research.semantic_token_cd.prompt_attn_shr_rollout import (
    KMEANS_K, KMEANS_SEED, LAMBDA, atomic_json, finite_mean, load_reference, write_arrays,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode


PROTOCOL = "PROMPT_ATTN_L11_MATCHED_PLUS_UNIFORM16_V1"
UNIFORM_COUNT = 16
ARMS = ("matched_uniform16",)
TASKS = SUPPORTED_TASKS


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ensure_config(artifact: Path, canonical: Path, matched: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    code = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "policy_sha256": file_sha(repo / "research/semantic_token_cd/prompt_attn_shr_policy_floor.py"),
        "rollout_sha256": file_sha(Path(__file__).resolve()),
    }
    payload = {
        "protocol_id": PROTOCOL,
        "purpose": "union matched's own-state mask with a 16-token evenly strided grid from the whole 16x16 patch grid",
        "tasks": list(TASKS),
        "seeds_by_task": {t: list(range(100, 200)) for t in TASKS},
        "new_arms": list(ARMS),
        "new_episode_count": len(TASKS) * 100,
        "uniform_count": UNIFORM_COUNT,
        "uniform_geometry": "16x16 grid, rows/cols 0,4,8,12 -> exactly 16 tokens; drawn from the WHOLE grid",
        "reference_arm": {"arm": "l11_matched", "artifact": str(matched)},
        "canonical_snapshot_artifact": str(canonical),
        "code_version": code,
        "locked_downstream": {
            "attention_layer": 11,
            "query": "full instruction excluding special/template/padding tokens",
            "head_query_aggregation": "equal arithmetic mean",
            "tie_break": "ascending visual-token index",
            "budget": "own-state KMeans entity-matched cluster size (matched)",
            "spatial_postprocessing": False,
            "harmonic": "16x16 four-neighbor Dirichlet beta=0/gamma=1",
            "prefix": "shared clean greedy prefix",
            "lambda": .5,
            "guided_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper": "clean positive dimension 6",
            "sampling": False,
        },
    }
    artifact.mkdir(parents=True, exist_ok=True)
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("uniform config lock differs")
    atomic_json(path, payload)
    return code


def build_policies(base, task: str, arms: tuple[str, ...]) -> dict:
    out = {}
    for arm in arms:
        p = copy.copy(base)
        p.__class__ = PromptAttentionSHRInference
        _init_common(p, LAMBDA)
        p.beta = 0.0
        p.selector_mode = "prompt_attention"
        p.task_index = TASK_INDEX[task]
        p.attention_layers = (11,)
        p.selection_count = None
        p.selection_top_p = None
        p.selection_budget_schedule = None
        p.selection_budget_scale = 1.0
        p.selection_budget_label = arm
        p.selection_budget_source = "matched"
        p.selection_budget_entities = None
        p.selection_budget_min_count = None
        p.selection_budget_max_count = None
        p.selection_budget_small_threshold = None
        p.selection_budget_small_scale = 1.0
        p.selection_extra_uniform_count = UNIFORM_COUNT
        p.save_prompt_attention = False
        out[arm] = p
    return out


def audit(trace: list[dict], arm: str) -> dict:
    def ok(fn):
        return bool(trace) and all(fn(x) for x in trace)
    checks = {
        "technical_nonempty": bool(trace),
        "all_feature_equal": ok(lambda x: x.get("feature_equal") is True),
        "all_guided_prefix": ok(lambda x: x.get("guided_prefix") is True),
        "all_reconstruction_finite": ok(lambda x: x.get("reconstruction_finite") is True),
        "all_lambda_locked": ok(lambda x: abs(float(x.get("lambda", -1)) - .5) < 1e-12),
        "all_beta_zero": ok(lambda x: abs(float(x.get("beta", -1))) < 1e-12),
        "all_coverage_exact": ok(lambda x: x.get("coverage_exact") is True),
        "all_non_target_bit_identical": ok(lambda x: x.get("non_target_bit_identical") is True),
        "all_l11_locked": ok(lambda x: x.get("attention_layers") == [11]),
        "all_post_softmax": ok(lambda x: x.get("attention_post_softmax") is True),
        "all_visual_keys_locked": ok(lambda x: x.get("visual_key_indices") == [1, 256]),
        "all_gripper_clean": ok(lambda x: x["positive_token_ids"][6] == x["final_token_ids"][6]),
        "all_budget_source_matched": ok(lambda x: x.get("budget_source") == "matched"),
        "all_kmeans_used": ok(lambda x: bool(x.get("reference_shr_token_ids"))),
        # uniform grid must be present in every selected set (union, not replacement)
        "all_uniform_subset": ok(lambda x: set(x.get("extra_uniform_ids", [])) <= set(x.get("selected_token_ids", []))),
        "all_uniform_count_exact": ok(lambda x: len(x.get("extra_uniform_ids", [])) == UNIFORM_COUNT),
        # selected size must equal matched m + newly added unique tokens
        "all_union_size_consistent": ok(
            lambda x: len(x.get("selected_token_ids", []))
            == int(x.get("budget_matched_m_t", -1)) + int(x.get("extra_added_count", -1))
        ),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"uniform rollout audit failed: {checks}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--matched-artifact", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve(); canonical = args.canonical.resolve(); matched = args.matched_artifact.resolve()
    code = ensure_config(artifact, canonical, matched)
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not arms or any(a not in ARMS for a in arms):
        raise ValueError(f"invalid arms: {arms}")
    seeds = parse_count_sweep_seeds(args.seeds)
    env, environment_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**config), args.task, arms)

    for seed in seeds:
        with (canonical / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as fh:
            snapshot = pickle.load(fh)
        reference = load_reference(canonical, args.task, seed)
        expected = (reference["canonical_snapshot_sha256"], reference["initial_state_sha256"], reference["initial_rgb_sha256"])
        if snapshot_sha(snapshot) != expected[0]:
            raise RuntimeError("canonical snapshot mismatch")
        for arm, policy in policies.items():
            out = artifact / "episodes" / args.task / arm
            sp = out / f"episode_{seed:03d}_summary.json"
            apth = out / f"episode_{seed:03d}_arrays.npz"
            if sp.exists() and apth.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True); continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if (state_sha, rgb_sha) != expected[1:]:
                raise RuntimeError("restored snapshot mismatch")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError("instruction mismatch")
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []; policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs)
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, arm)
            out.mkdir(parents=True, exist_ok=True)
            write_arrays(apth, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL, "task": args.task, "environment_id": environment_id,
                "seed": seed, "evaluation_seed": seed, "episode_id": seed, "arm": arm,
                "uniform_count": UNIFORM_COUNT, "attention_layers": [11], "instruction": instruction,
                "success": bool(result["success"]), "result": jsonable(result), "failure_reason": reason,
                "control_steps": steps, "runtime_seconds": runtime, "gpu_id": args.gpu, "worker_id": args.worker_id,
                "canonical_snapshot_sha256": expected[0], "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                "lambda": LAMBDA, "beta": 0.0, "kmeans_K": KMEANS_K, "kmeans_seed": KMEANS_SEED,
                "code_version": code, "arrays_file": apth.name, "action_jitter_index": jerk,
                "mean_m_t": finite_mean(x.get("actual_selected_count") for x in trace),
                "mean_matched_m_t": finite_mean(x.get("budget_matched_m_t") for x in trace),
                "mean_extra_added": finite_mean(x.get("extra_added_count") for x in trace),
                "mean_feature_perturbation_relative": finite_mean(x.get("feature_perturbation_relative") for x in trace),
                "mean_guided_change_ratio": finite_mean(x.get("guided_change_ratio") for x in trace),
                "mean_centered_logit_residual_norm": finite_mean(x.get("centered_logit_residual_norm") for x in trace),
                "selector_trace": trace, **checks,
            }
            atomic_json(sp, summary)
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": summary["success"],
                              "runtime_seconds": round(runtime, 2), "mean_m_t": round(summary["mean_m_t"], 2),
                              "mean_matched_m_t": round(summary["mean_matched_m_t"], 2),
                              "mean_extra_added": round(summary["mean_extra_added"], 2)}), flush=True)


if __name__ == "__main__":
    main()
