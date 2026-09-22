"""Closed-loop: Matched with an episode-internal PERMUTED budget tape (state-coupling ablation).

Hypothesis under test:
    H1: SR(matched, m_t = f(x_t))  >  SR(matched, m_t = m_{pi(t)})
The two arms share the identical episode-level budget multiset (mean/std/min/max/
histogram/value set).  The only thing destroyed is which m belongs to which state.

The tape is frozen BEFORE the rollout (SHUFFLE_TAPES.json).  The current state's
own matched budget is still evaluated every step, but only logged - never used.
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
from research.semantic_token_cd.prompt_attn_l11_count_rollout import parse_count_sweep_seeds
from research.semantic_token_cd.prompt_attn_shr_rollout import (
    KMEANS_K, KMEANS_SEED, LAMBDA, atomic_json, finite_mean, load_reference, write_arrays,
)

# NOTE: prompt_attn_shr_rollout.TASKS only contains three tasks (no close_drawer).
# This protocol is the four google_robot tasks, matching the matched reference.
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode

PROTOCOL = "PROMPT_ATTN_L11_MATCHED_STATE_COUPLING_V1"
ARMS = ("matched_shuffle",)
EXTENSION_PAD = 100


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ensure_config(artifact: Path, canonical: Path, matched: Path, tape_file: Path, tape_sha: str) -> dict:
    repo = Path(__file__).resolve().parents[2]
    code = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "policy_sha256": file_sha(repo / "research/semantic_token_cd/prompt_attn_shr_policy_floor.py"),
        "rollout_sha256": file_sha(Path(__file__).resolve()),
        "tape_sha256": file_sha(tape_file),
    }
    payload = {
        "protocol_id": PROTOCOL,
        "hypothesis": "H1: matched (m_t = f(x_t)) > matched with a permuted budget tape",
        "purpose": "state-coupling ablation: keep the budget multiset, destroy the m<->state correspondence",
        "tasks": list(TASKS),
        "seeds_by_task": {t: list(range(100, 200)) for t in TASKS},
        "new_arms": list(ARMS),
        "new_episode_count": len(TASKS) * 100,
        "reference_arm": {"arm": "l11_matched", "artifact": str(matched)},
        "tape_file": str(tape_file),
        "tape_sha256_declared": tape_sha,
        "canonical_snapshot_artifact": str(canonical),
        "code_version": code,
        "locked_downstream": {
            "attention_layer": 11,
            "query": "full instruction excluding special/template/padding tokens",
            "head_query_aggregation": "equal arithmetic mean",
            "tie_break": "ascending visual-token index",
            "budget": "FROZEN per-seed permuted tape (matched multiset, order destroyed)",
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
        raise RuntimeError("state-coupling config lock differs")
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
        p.selection_budget_scale = 1.0
        p.selection_budget_label = arm
        p.selection_budget_source = "matched"
        p.selection_budget_entities = None
        p.selection_budget_min_count = None
        p.selection_budget_max_count = None
        p.selection_budget_small_threshold = None
        p.selection_budget_small_scale = 1.0
        p.selection_extra_uniform_count = None
        p.selection_spatial_mode = None
        p.selection_spatial_zero_ids = None
        p.log_current_matched_m = True
        p.save_prompt_attention = False
        out[arm] = p
    return out


def audit(trace: list[dict], tape: list[int]) -> dict:
    def ok(fn):
        return bool(trace) and all(fn(x) for x in trace)
    used = [int(x["actual_selected_count"]) for x in trace]
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
        "all_schedule_used": ok(lambda x: x.get("budget_source") == "frozen_schedule"),
        # the tape must be consumed verbatim while it lasts
        "tape_consumed_verbatim": all(used[i] == tape[i] for i in range(min(len(used), len(tape)))),
        "current_matched_logged": ok(lambda x: x.get("current_state_matched_m") is not None),
    }
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"state-coupling audit failed: {checks}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--tape-file", type=Path, required=True)
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
    tape_file = args.tape_file.resolve()
    tape_doc = json.loads(tape_file.read_text())
    tapes = tape_doc["tapes"]
    ext_pool = {t: list(v) for t, v in tape_doc["extension_pool"].items()}
    code = ensure_config(artifact, canonical, matched, tape_file, tape_doc.get("sha256", ""))
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not arms or any(a not in ARMS for a in arms):
        raise ValueError(f"invalid arms: {arms}")
    seeds = parse_count_sweep_seeds(args.seeds)

    from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
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
        base_tape = [int(v) for v in tapes[args.task][str(seed)]]
        rng = np.random.default_rng([20260918, 977, seed])
        pad = [int(v) for v in rng.choice(ext_pool[args.task], size=EXTENSION_PAD, replace=True)]
        full_tape = base_tape + pad

        for arm, policy in policies.items():
            out = artifact / "episodes" / args.task / arm
            sp = out / f"episode_{seed:03d}_summary.json"
            apth = out / f"episode_{seed:03d}_arrays.npz"
            if sp.exists() and apth.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True); continue
            policy.selection_budget_schedule = tuple(full_tape)
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
            checks = audit(trace, full_tape)
            used = [int(x["actual_selected_count"]) for x in trace]
            cur = [x.get("current_state_matched_m") for x in trace]
            cur_ok = [c for c in cur if c is not None]
            deltas = [abs(int(u) - int(c)) for u, c in zip(used, cur) if c is not None]
            n_ext = max(0, len(trace) - len(base_tape))
            out.mkdir(parents=True, exist_ok=True)
            write_arrays(apth, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL, "task": args.task, "environment_id": environment_id,
                "seed": seed, "evaluation_seed": seed, "episode_id": seed, "arm": arm,
                "attention_layers": [11], "instruction": instruction,
                "success": bool(result["success"]), "result": jsonable(result), "failure_reason": reason,
                "control_steps": steps, "runtime_seconds": runtime, "gpu_id": args.gpu, "worker_id": args.worker_id,
                "canonical_snapshot_sha256": expected[0], "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                "lambda": LAMBDA, "beta": 0.0, "kmeans_K": KMEANS_K, "kmeans_seed": KMEANS_SEED,
                "code_version": code, "arrays_file": apth.name, "action_jitter_index": jerk,
                "tape_length": len(base_tape),
                "num_steps_from_exact_permutation": len(trace) - n_ext,
                "num_steps_from_extension_pool": n_ext,
                "mean_m_t": finite_mean(used),
                "mean_matched_m_t": finite_mean(cur_ok),
                "mean_abs_delta_vs_current": finite_mean(deltas),
                "max_abs_delta_vs_current": (float(max(deltas)) if deltas else None),
                "selector_trace": trace, **checks,
            }
            atomic_json(sp, summary)
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": summary["success"],
                              "runtime_seconds": round(runtime, 2), "steps": steps,
                              "mean_tape_m": round(summary["mean_m_t"], 2),
                              "mean_current_m": round(summary["mean_matched_m_t"], 2),
                              "mean_abs_delta": round(summary["mean_abs_delta_vs_current"], 2),
                              "ext_steps": n_ext}), flush=True)


if __name__ == "__main__":
    main()
