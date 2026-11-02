"""OpenVLA budget-rule rollout: Matched vs Attention-TopP vs Fixed budget (all at lambda=1.0).

Purpose
-------
Hold EVERYTHING of the lambda=1.0 "Ours" protocol fixed and vary only how the per-step
budget m_t is determined:

    matched : own-state KMeans entity-matched cluster size      (the existing Ours arm)
    top_p   : smallest k whose cumulative L11 attention mass >= p, UNCLIPPED
              m_t = min{ k : sum_{i<=k} a~_(pi_i) >= p },  p in {0.75, 0.80, 0.85, 0.90}
    fixed   : m_t = m, constant, m in {16, 24, 32, 40, 48, 56, 64}

All three rules select the top-m_t tokens from the SAME L11 prompt-attention ranking, so the
only thing that changes is the number.

Everything else is inherited byte-for-byte from
`openvla_global_lambda_grid_rollout.py` (which produced
`artifacts/openvla_lambda1_extension_200_299_v1`, the 900-episode lambda=1.0 Ours set):
same checkpoint, same canonical snapshot artifact, same per-task RT mode
(depth 8 for open/close drawer, depth 0 for the other seven), same harmonic counterfactual,
same contrast execution, same projector intervention point, same audit, same output schema.

TopP clipping
-------------
`prompt_attn_shr_policy` clips the TopP count into [selection_min_count, selection_max_count],
which the earlier lambda=0.5 TopP study set to [16, 64].  Here the bounds are set to
[1, N_VISUAL=256], i.e. the clip is inert and the rule is exactly the formula above; the trace
records the bounds (`selection_min_count`/`selection_max_count`) so an unclipped run is provable
from the episode rows.

Usage
-----
    python -m research.semantic_token_cd.openvla_budget_rule_rollout \
        --artifact artifacts/openvla_budget_rule_9task_200_299_v1 \
        --task widowx_carrot_on_plate --seeds 200-219 --gpu 1 --worker-id smoke_w1 \
        --arms lambda_100,top_p085_l100,fixed032_l100 \
        --protocol-id OPENVLA_BUDGET_RULE_9TASK_200_299_V1 --lock-seeds 200-299
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd import prompt_attn_l11_budget_causal_rollout as shared
from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.distractor_policy import _action_logits
from research.semantic_token_cd.prompt_attn_shr_policy import N_VISUAL, PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json, finite_mean, write_arrays
from research.semantic_token_cd.semantic_recon_rollout import TASKS as ALL_TASKS, TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode

TASKS = tuple(ALL_TASKS)
RT_DEPTH = {
    **shared.RT_DEPTH,
    "google_robot_place_apple_in_closed_top_drawer": 0,
    "widowx_carrot_on_plate": 0,
    "widowx_put_eggplant_in_basket": 0,
    "widowx_spoon_on_towel": 0,
    "widowx_stack_cube": 0,
}
shared.RT_DEPTH.update(RT_DEPTH)
RT0_TASKS = set(RT_DEPTH) - {"google_robot_open_drawer", "google_robot_close_drawer"}

STRENGTH = 1.0                  # default lambda; arms may override it with a _lXYZ suffix
CONSTRUCTOR = "harmonic"
EXECUTION = "contrast"
LOCATION = "projector"

RANDOM_SALT = 0xB0D6E7

# Per-task mean of Matched's own per-episode mean m_t, measured on the lambda=1.0 reference
# (artifacts/openvla_lambda1_extension_200_299_v1, seeds 200-299).  The `taskmean` rule uses
# these as a CONSTANT count: same task-level average as Matched, but no per-step adaptation.
TASKMEAN_BUDGETS = {
    "google_robot_close_drawer": 35,
    "google_robot_open_drawer": 37,
    "google_robot_pick_coke_can": 29,
    "google_robot_move_near": 33,
    "google_robot_place_apple_in_closed_top_drawer": 57,
    "widowx_put_eggplant_in_basket": 46,
    "widowx_carrot_on_plate": 52,
    "widowx_spoon_on_towel": 54,
    "widowx_stack_cube": 49,
}


def parse_arm(arm: str) -> tuple[str, float, float | int | None]:
    """arm -> (budget rule, lambda, parameter).

    Naming: <rule>_l<lambda*100>, e.g. matched_l100, top_p075_l050, fixed016_l100,
    random_fixed48_l100.  The legacy Ours arm name `lambda_100` is kept as an alias for
    matched at lambda=1.0 so the historical reference can be reproduced bit-for-bit.
    """
    if arm == "lambda_100":
        return "matched", 1.00, None
    rule_part, _, lam_part = arm.rpartition("_l")
    if not rule_part or not lam_part.isdigit():
        raise ValueError(f"cannot parse arm name: {arm}")
    strength = int(lam_part) / 100.0
    if rule_part == "matched":
        return "matched", strength, None
    if rule_part.startswith("top_p"):
        return "top_p", strength, int(rule_part[5:]) / 100.0
    if rule_part == "taskmean":
        return "taskmean", strength, None
    if rule_part.startswith("fixed"):
        return "fixed", strength, int(rule_part[5:])
    if rule_part.startswith("random_fixed"):
        return "random_fixed", strength, int(rule_part[12:])
    raise ValueError(f"unknown budget rule in arm name: {arm}")


def budget_arms(names: tuple[str, ...] | list[str]) -> dict[str, tuple[str, float, float | int | None]]:
    return {a: parse_arm(a) for a in names}

SELECTOR_LABEL = {
    "matched": "L11 prompt attention + canonical dynamic Matched budget",
    "top_p": "L11 prompt attention + visual Top-p (unclipped)",
    "fixed": "L11 prompt attention + fixed token budget",
    "random_fixed": "random visual tokens + fixed token budget",
    "taskmean": "L11 prompt attention + fixed per-task mean budget (no per-step adaptation)",
}


def write_budget_arrays(path: Path, records: list[dict], actions: np.ndarray, lambd: float) -> None:
    positive = np.stack([record["positive"] for record in records]).astype(np.float32)
    negative = np.stack([record["negative"] for record in records]).astype(np.float32)
    guided = np.stack([record["guided"] for record in records]).astype(np.float32)
    np.savez_compressed(
        path,
        positive_logits=positive,
        negative_logits=negative,
        guided_logits=guided,
        executed_actions=actions,
    )


def build_policies(base, task: str, arms: tuple[str, ...]):
    policies = {}
    for arm in arms:
        rule, strength, param = parse_arm(arm)
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, strength)
        policy.beta = 0.0
        policy.selector_mode = "random_matched" if rule == "random_fixed" else "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,)
        policy.selection_budget_schedule = None
        policy.selection_budget_scale = 1.0
        policy.selection_budget_label = arm
        # defaults: no budget override at all (matched)
        policy.selection_count = None
        policy.selection_top_p = None
        policy.selection_min_count = 16
        policy.selection_max_count = 64
        policy.selection_budget_source = "matched"
        policy.selection_budget_random_salt = RANDOM_SALT
        if rule == "top_p":
            policy.selection_top_p = float(param)
            # unclipped: the policy clips into [min, max], so widen the bounds to the full
            # visual grid and the clip becomes inert.
            policy.selection_min_count = 1
            policy.selection_max_count = int(N_VISUAL)
            policy.selection_budget_source = "top_p"
        elif rule in ("fixed", "random_fixed", "taskmean"):
            policy.selection_count = int(TASKMEAN_BUDGETS[task]) if rule == "taskmean" else int(param)
            policy.selection_budget_source = {"fixed": "fixed", "random_fixed": "random_fixed",
                                              "taskmean": "taskmean"}[rule]
        policy.counterfactual_constructor = CONSTRUCTOR
        policy.guidance_execution = EXECUTION
        policy.intervention_location = LOCATION
        policy.save_prompt_attention = False

        # logging-only capture of the post-combination action logits (same as the base runner)
        original_combine = policy._combine_action_scores

        def capture_combined_scores(clean_scores, negative_scores, *, _policy=policy, _combine=original_combine):
            final_scores, meta = _combine(clean_scores, negative_scores)
            _policy._budget_rule_final_scores = final_scores.detach()
            return final_scores, meta

        policy._combine_action_scores = capture_combined_scores
        original_step = policy.step

        def step_and_record_guided_logits(*step_args, _policy=policy, _step=original_step, **step_kwargs):
            _policy._budget_rule_final_scores = None
            result = _step(*step_args, **step_kwargs)
            final_scores = _policy._budget_rule_final_scores
            if final_scores is None:
                raise RuntimeError("contrastive guided scores were not captured")
            final_scores = final_scores.clone()
            eos_id = int(_policy.vla.generation_config.eos_token_id)
            invalid = ~torch.isfinite(final_scores)
            if invalid.any():
                expected = torch.zeros_like(invalid, dtype=torch.bool)
                expected[:, eos_id] = True
                if (invalid & ~expected).any():
                    raise FloatingPointError("non-finite guided logits")
                final_scores[:, eos_id] = torch.finfo(final_scores.dtype).min
            _policy._episode_logits[-1]["guided"] = _action_logits(_policy, final_scores)
            _policy._budget_rule_final_scores = None
            return result

        policy.step = step_and_record_guided_logits
        policies[arm] = policy
    return policies


def audit(trace: list[dict], arm: str, TASK: str) -> dict:
    rule, strength, param = parse_arm(arm)
    checks = {
        "nonempty_trace": bool(trace),
        "budget_count_exact": bool(trace) and all(
            int(x.get("actual_selected_count", -1)) == int(x.get("m_t", -2)) for x in trace
        ),
        "clean_features_preserved": bool(trace) and all(x.get("feature_equal") is True for x in trace),
        "unselected_features_unchanged": bool(trace) and all(
            x.get("non_target_bit_identical") is True for x in trace
        ),
        "finite_reconstruction": bool(trace) and all(x.get("reconstruction_finite") is True for x in trace),
        "lambda_exact": bool(trace) and all(abs(float(x.get("lambda", -1)) - strength) < 1e-12 for x in trace),
        # the attention-based rules must all use the L11 prompt-attention ranking; the random
        # rule deliberately does not compute it (its trace carries no attention_layers field)
        "layer_exact": bool(trace) and (
            all(x.get("attention_layers") == [11] for x in trace) if rule != "random_fixed"
            else all(x.get("attention_layers") in (None, [], ()) for x in trace)
        ),
        "constructor_exact": bool(trace) and all(
            x.get("counterfactual_constructor") == CONSTRUCTOR for x in trace
        ),
        "execution_exact": bool(trace) and all(x.get("guidance_execution") == EXECUTION for x in trace),
        "intervention_location_exact": bool(trace) and all(
            x.get("intervention_location") == LOCATION for x in trace
        ),
        "seven_action_tokens": bool(trace) and all(
            len(x.get("positive_token_ids", [])) == 7
            and len(x.get("negative_token_ids", [])) == 7
            and len(x.get("final_token_ids", [])) == 7 for x in trace
        ),
    }
    if rule == "fixed":
        checks["fixed_budget_exact"] = bool(trace) and all(
            int(x.get("m_t", -1)) == int(param) for x in trace
        )
    if rule == "taskmean":
        checks["taskmean_budget_exact"] = bool(trace) and all(
            int(x.get("m_t", -1)) == int(TASKMEAN_BUDGETS[TASK]) for x in trace
        )
    if rule == "random_fixed":
        checks["random_budget_exact"] = bool(trace) and all(
            int(x.get("m_t", -1)) == int(param) for x in trace
        )
        checks["random_selector_used"] = bool(trace) and all(
            str(x.get("coverage_mode", "")).startswith("fixed_top_") for x in trace
        )
    if rule == "top_p":
        checks["top_p_threshold_exact"] = bool(trace) and all(
            abs(float(x.get("top_p_threshold") or -1.0) - float(param)) < 1e-12 for x in trace
        )
        checks["top_p_unclipped_bounds"] = bool(trace) and all(
            int(x.get("selection_min_count", -1)) == 1
            and int(x.get("selection_max_count", -1)) == int(N_VISUAL) for x in trace
        )
        checks["top_p_no_clip_triggered"] = bool(trace) and all(
            not x.get("lower_bound_triggered") and not x.get("upper_bound_triggered") for x in trace
        )
        checks["top_p_within_grid"] = bool(trace) and all(
            1 <= int(x.get("m_t", 0)) <= int(N_VISUAL) for x in trace
        )
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"budget-rule audit failed for {arm}: {checks}")
    return checks


def load_scan_reference(snapshot_root: Path, task: str, seed: int) -> dict:
    if task not in RT0_TASKS:
        return shared.load_task_reference(snapshot_root, task, seed)
    path = shared.RT0_REFERENCE / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing RT-depth0 reference: {path}")
    reference = json.loads(path.read_text())
    if "depth0" not in str(reference.get("renderer", "")).lower():
        raise RuntimeError(f"RT-depth0 reference has wrong renderer: {path}")
    canonical = shared.load_reference(snapshot_root, task, seed)
    reference["instruction"] = canonical["instruction"]
    return reference


def parse_seeds(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    result = sorted(set(values))
    if not result or any(seed < 0 or seed > 299 for seed in result):
        raise ValueError("seeds must be within 0..299")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path,
                    default=Path("artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--protocol-id", default="OPENVLA_BUDGET_RULE_9TASK_200_299_V1")
    ap.add_argument("--lock-seeds", default="200-299")
    args = ap.parse_args()

    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    try:
        arm_table = budget_arms(arms)
    except ValueError as exc:
        raise ValueError(f"invalid arms {arms}: {exc}") from exc
    seeds = parse_seeds(args.seeds)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    locked_seeds = parse_seeds(args.lock_seeds)
    lock = {
        "protocol_id": args.protocol_id,
        "model": "OpenVLA Fractal SIMPLER checkpoint",
        "tasks": list(TASKS),
        "seeds": locked_seeds,
        "arms": list(arms),
        "budget_rule": {a: arm_table[a][0] for a in arms},
        "budget_param": {a: arm_table[a][2] for a in arms},
        "rendering": {t: {"shader_dir": "rt", "rt_max_path_depth": RT_DEPTH[t],
                          "rt_samples_per_pixel": 32, "rt_use_denoiser": True} for t in TASKS},
        "snapshot_artifact": str(source),
        "paired_reference": "task-matched RT baseline; RT0 reference for Pick/Move",
        "attention_layer_zero_based": 11,
        "lambda": {a: arm_table[a][1] for a in arms},
        "constructor": {a: CONSTRUCTOR for a in arms},
        "execution": {a: EXECUTION for a in arms},
        "intervention_location": {a: LOCATION for a in arms},
        "top_p_unclipped": {a: (arm_table[a][0] == "top_p") for a in arms},
        "vary": "only the budget rule m_t; everything else identical to the lambda=1.0 Ours run",
    }
    lock_path = artifact / "CONFIG_LOCK.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text())
        # only the study-wide fields must agree between workers; everything arm-scoped is merged
        for key in ("protocol_id", "model", "seeds", "snapshot_artifact", "rendering",
                    "paired_reference", "attention_layer_zero_based", "vary"):
            if key in existing and existing.get(key) != lock[key]:
                raise RuntimeError(f"CONFIG_LOCK mismatch on {key}: {lock_path}")
        merged = dict(existing)
        merged["arms"] = sorted(set(existing.get("arms", [])) | set(arms))
        for key in ("lambda", "constructor", "execution", "intervention_location",
                    "budget_rule", "budget_param", "top_p_unclipped"):
            merged[key] = {**existing.get(key, {}), **lock[key]}
        lock = merged
    atomic_json(lock_path, lock)

    env, environment_id = shared.make_environment(args.task)
    model_config = get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), args.task, {}, False)
    policies = build_policies(OpenVLAInference(**model_config), args.task, arms)
    for seed in seeds:
        snapshot = pickle.loads((source / "snapshots" / args.task / f"seed_{seed:03d}.pkl").read_bytes())
        reference = load_scan_reference(source, args.task, seed)
        expected = (reference["canonical_snapshot_sha256"], reference["initial_state_sha256"],
                    reference["initial_rgb_sha256"])
        if snapshot_sha(snapshot) != expected[0]:
            raise RuntimeError(f"snapshot hash mismatch: {args.task}:{seed}")
        pair_hashes = []
        for arm, policy in policies.items():
            rule, strength, param = arm_table[arm]
            out = artifact / "episodes" / args.task / arm
            summary_path = out / f"episode_{seed:03d}_summary.json"
            arrays_path = out / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                old = json.loads(summary_path.read_text())
                if old.get("protocol_id") != lock["protocol_id"]:
                    raise RuntimeError(f"incompatible existing output: {summary_path}")
                pair_hashes.append((old["initial_state_sha256"], old["initial_rgb_sha256"]))
                continue
            obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
            if (state_hash, rgb_hash) != expected[1:]:
                raise RuntimeError(f"paired snapshot mismatch {args.task}:{seed}")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError(f"instruction mismatch: {args.task}:{seed}")
            policy.reset(instruction, seed=seed)
            policy._episode_trace, policy._episode_logits = [], []
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs)
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, arm, args.task)
            out.mkdir(parents=True, exist_ok=True)
            write_budget_arrays(arrays_path, policy._episode_logits, actions, strength)
            counts = [int(x.get("actual_selected_count", 0)) for x in trace]
            payload = {
                "protocol_id": lock["protocol_id"], "task": args.task, "seed": seed, "episode_id": seed,
                "arm": arm, "budget_rule": rule, "budget_param": param,
                "success": bool(result["success"]), "result": jsonable(result),
                "failure_reason": reason, "steps": steps,
                "runtime_seconds": time.monotonic() - started,
                "worker_id": args.worker_id, "gpu_id": args.gpu, "environment_id": environment_id,
                "canonical_snapshot_sha256": expected[0], "initial_state_sha256": state_hash,
                "initial_rgb_sha256": rgb_hash, "instruction": instruction,
                "lambda": strength, "attention_layers": [11],
                "rendering": {"shader_dir": "rt", "rt_max_path_depth": RT_DEPTH[args.task],
                              "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
                "selector": SELECTOR_LABEL[rule] + ("" if param is None else f" {param}"),
                "counterfactual_constructor": CONSTRUCTOR, "guidance_execution": EXECUTION,
                "arrays_file": arrays_path.name, "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(counts),
                "min_selected_token_count": min(counts) if counts else None,
                "max_selected_token_count": max(counts) if counts else None,
                "selector_trace": trace, **checks,
            }
            atomic_json(summary_path, payload)
            pair_hashes.append((state_hash, rgb_hash))
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "rule": rule,
                              "lambda": strength, "param": param, "success": payload["success"], "steps": steps,
                              "mean_m_t": round(payload["mean_selected_token_count"], 2),
                              "runtime_seconds": round(payload["runtime_seconds"], 2)},
                             sort_keys=True), flush=True)
        if len(set(pair_hashes)) != 1:
            raise RuntimeError(f"arm pair identity mismatch: {args.task}:{seed}: {pair_hashes}")


if __name__ == "__main__":
    main()
