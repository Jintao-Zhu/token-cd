"""Canonical OpenVLA global lambda-grid rollout for SIMPLER, seeds 0-199.

Cards 4 (budget scaling) uses the existing budget-causal runner directly.
This entry point deliberately reuses that runner's audited RT environment,
snapshot restore, and matched-reference validation.
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
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
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
# The shared renderer factory consumes this per-task map. Extend it only in
# this runner process so other studies keep their four-Google-task lock.
shared.RT_DEPTH.update(RT_DEPTH)
RT0_TASKS = set(RT_DEPTH) - {"google_robot_open_drawer", "google_robot_close_drawer"}


def write_lambda_grid_arrays(path: Path, records: list[dict], actions: np.ndarray, lambd: float) -> None:
    """Save both branches and the exact contrastive logits used for analysis."""
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
CARD_ARMS = {
    "1": {
        "zero": ("zero", "contrast", 0.5),
        "clean_global_mean": ("clean_global_mean", "contrast", 0.5),
        "local_neighbor_mean": ("local_neighbor_mean", "contrast", 0.5),
    },
    "2": {
        "hard_intervention_only": ("zero", "hard_intervention_only", 0.5),
        "harmonic_negative_only": ("harmonic", "negative_only", 0.5),
    },
    "3": {
        "layer5": ("harmonic", "contrast", 0.5),
        "layer11": ("harmonic", "contrast", 0.5),
    },
    "6": {
        "lambda_000": ("harmonic", "contrast", 0.00),
        "lambda_025": ("harmonic", "contrast", 0.25),
        "lambda_050": ("harmonic", "contrast", 0.50),
        "lambda_075": ("harmonic", "contrast", 0.75),
        "lambda_100": ("harmonic", "contrast", 1.00),
        "lambda_125": ("harmonic", "contrast", 1.25),
        "lambda_150": ("harmonic", "contrast", 1.50),
        "lambda_175": ("harmonic", "contrast", 1.75),
        "lambda_200": ("harmonic", "contrast", 2.00),
        "lambda_225": ("harmonic", "contrast", 2.25),
        "lambda_250": ("harmonic", "contrast", 2.50),
    },
}
CARD_LOCATION = {("3", "layer5"): "layer5", ("3", "layer11"): "layer11"}


def parse_lambda_grid_seeds(spec: str) -> list[int]:
    values = []
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
        raise ValueError("global lambda-grid seeds must be within 0..299")
    return result


def build_policies(base, task: str, card: str, arms: tuple[str, ...]):
    policies = {}
    for arm in arms:
        constructor, execution, strength = CARD_ARMS[card][arm]
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, strength)
        policy.beta = 0.0
        policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,)
        policy.selection_count = None
        policy.selection_top_p = None
        policy.selection_budget_schedule = None
        policy.selection_budget_scale = 1.0
        policy.selection_budget_label = arm
        policy.selection_budget_source = "matched"
        policy.counterfactual_constructor = constructor
        policy.guidance_execution = execution
        policy.intervention_location = CARD_LOCATION.get((card, arm), "projector")
        policy.save_prompt_attention = False

        # Capture the actual post-combination action logits for the scan arrays.
        # This is logging-only; the original scores and return value are passed
        # through unchanged to the canonical decoder.
        original_combine = policy._combine_action_scores
        def capture_combined_scores(clean_scores, negative_scores, *, _policy=policy, _combine=original_combine):
            final_scores, meta = _combine(clean_scores, negative_scores)
            _policy._lambda_grid_final_scores = final_scores.detach()
            return final_scores, meta
        policy._combine_action_scores = capture_combined_scores
        original_step = policy.step
        def step_and_record_guided_logits(*step_args, _policy=policy, _step=original_step, **step_kwargs):
            _policy._lambda_grid_final_scores = None
            result = _step(*step_args, **step_kwargs)
            final_scores = _policy._lambda_grid_final_scores
            if final_scores is None:
                raise RuntimeError("contrastive guided scores were not captured")
            final_scores = final_scores.clone()
            eos_id = int(_policy.vla.generation_config.eos_token_id)
            invalid = ~torch.isfinite(final_scores)
            if invalid.any():
                expected = torch.zeros_like(invalid, dtype=torch.bool)
                expected[:, eos_id] = True
                if (invalid & ~expected).any():
                    raise FloatingPointError("non-finite guided lambda-grid logits")
                final_scores[:, eos_id] = torch.finfo(final_scores.dtype).min
            _policy._episode_logits[-1]["guided"] = _action_logits(_policy, final_scores)
            _policy._lambda_grid_final_scores = None
            return result
        policy.step = step_and_record_guided_logits
        policies[arm] = policy
    return policies


def audit(trace: list[dict], arm: str, card: str) -> dict:
    constructor, execution, strength = CARD_ARMS[card][arm]
    checks = {
        "nonempty_trace": bool(trace),
        "matched_count_exact": bool(trace) and all(
            int(x.get("actual_selected_count", -1)) == int(x.get("m_t", -2)) for x in trace
        ),
        "clean_features_preserved": bool(trace) and all(x.get("feature_equal") is True for x in trace),
        "unselected_features_unchanged": bool(trace) and all(
            x.get("non_target_bit_identical") is True for x in trace
        ),
        "finite_reconstruction": bool(trace) and all(x.get("reconstruction_finite") is True for x in trace),
        "lambda_exact": bool(trace) and all(abs(float(x.get("lambda", -1)) - strength) < 1e-12 for x in trace),
        "layer_exact": bool(trace) and all(x.get("attention_layers") == [11] for x in trace),
        "constructor_exact": bool(trace) and all(
            x.get("counterfactual_constructor") == constructor for x in trace
        ),
        "execution_exact": bool(trace) and all(x.get("guidance_execution") == execution for x in trace),
        "intervention_location_exact": bool(trace) and all(
            x.get("intervention_location") == CARD_LOCATION.get((card, arm), "projector")
            for x in trace
        ),
        "seven_action_tokens": bool(trace) and all(
            len(x.get("positive_token_ids", [])) == 7
            and len(x.get("negative_token_ids", [])) == 7
            and len(x.get("final_token_ids", [])) == 7 for x in trace
        ),
    }
    if execution != "contrast":
        checks["direct_negative_decode"] = bool(trace) and all(
            x.get("final_token_ids") == x.get("negative_token_ids") for x in trace
        )
    if card == "2" and arm == "hard_intervention_only":
        checks["zero_constructor"] = bool(trace) and all(
            x.get("counterfactual_constructor") == "zero" for x in trace
        )
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"seven-card card {card} audit failed for {arm}: {checks}")
    return checks


def load_scan_reference(snapshot_root: Path, task: str, seed: int) -> dict:
    """Use the renderer-matched baseline for all depth-0 scan tasks."""
    if task not in RT0_TASKS:
        return shared.load_task_reference(snapshot_root, task, seed)
    path = shared.RT0_REFERENCE / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing RT-depth0 reference: {path}")
    reference = json.loads(path.read_text())
    if "depth0" not in str(reference.get("renderer", "")).lower():
        raise RuntimeError(f"RT-depth0 reference has wrong renderer: {path}")
    # Keep the canonical seed's instruction, which may be randomized by task.
    canonical = shared.load_reference(snapshot_root, task, seed)
    reference["instruction"] = canonical["instruction"]
    return reference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", choices=tuple(CARD_ARMS), required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, default=Path("artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--protocol-id", default=None,
                    help="Optional protocol label for an extension artifact.")
    ap.add_argument("--lock-seeds", default=None,
                    help="Full seed range recorded in CONFIG_LOCK, shared by all workers.")
    args = ap.parse_args()
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not arms or any(x not in CARD_ARMS[args.card] for x in arms):
        raise ValueError(f"invalid card {args.card} arms: {arms}")
    seeds = parse_lambda_grid_seeds(args.seeds)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    protocol_id = args.protocol_id or "OPENVLA_GLOBAL_LAMBDA_GRID_0_199_V1"
    locked_seeds = parse_lambda_grid_seeds(args.lock_seeds) if args.lock_seeds else list(range(0, 200))
    lock = {
        "protocol_id": protocol_id,
        "model": "OpenVLA Fractal SIMPLER checkpoint",
        "tasks": list(TASKS),
        "seeds": locked_seeds,
        "arms": list(arms),
        "rendering": {t: {"shader_dir": "rt", "rt_max_path_depth": RT_DEPTH[t], "rt_samples_per_pixel": 32, "rt_use_denoiser": True} for t in TASKS},
        "snapshot_artifact": str(source), "paired_reference": "task-matched RT baseline; RT0 reference for Pick/Move",
        "attention_layer_zero_based": 11, "lambda": {a: CARD_ARMS[args.card][a][2] for a in arms},
        "constructor": {a: CARD_ARMS[args.card][a][0] for a in arms},
        "execution": {a: CARD_ARMS[args.card][a][1] for a in arms},
        "intervention_location": {a: CARD_LOCATION.get((args.card, a), "projector") for a in arms},
    }
    lock_path = artifact / "CONFIG_LOCK.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise RuntimeError(f"CONFIG_LOCK mismatch: {lock_path}")
    atomic_json(lock_path, lock)

    env, environment_id = shared.make_environment(args.task)
    model_config = get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), args.task, {}, False)
    policies = build_policies(OpenVLAInference(**model_config), args.task, args.card, arms)
    for seed in seeds:
        snapshot = pickle.loads((source / "snapshots" / args.task / f"seed_{seed:03d}.pkl").read_bytes())
        reference = load_scan_reference(source, args.task, seed)
        expected = (reference["canonical_snapshot_sha256"], reference["initial_state_sha256"], reference["initial_rgb_sha256"])
        if snapshot_sha(snapshot) != expected[0]:
            raise RuntimeError(f"snapshot hash mismatch: {args.task}:{seed}")
        pair_hashes = []
        for arm, policy in policies.items():
            out = artifact / "episodes" / args.task / arm
            summary_path, arrays_path = out / f"episode_{seed:03d}_summary.json", out / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                old = json.loads(summary_path.read_text())
                if old.get("protocol_id") not in {lock["protocol_id"], "SEVEN_CARD_C6_OPENVLA_V1"}:
                    raise RuntimeError(f"incompatible existing output: {summary_path}")
                pair_hashes.append((old["initial_state_sha256"], old["initial_rgb_sha256"]))
                continue
            obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
            if (state_hash, rgb_hash) != expected[1:]:
                raise RuntimeError(f"paired snapshot mismatch {args.task}:{seed}: state={state_hash == expected[1]} rgb={rgb_hash == expected[2]}")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError(f"instruction mismatch: {args.task}:{seed}")
            policy.reset(instruction, seed=seed)
            policy._episode_trace, policy._episode_logits = [], []
            started = time.monotonic()
            result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs)
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, arm, args.card)
            out.mkdir(parents=True, exist_ok=True)
            write_lambda_grid_arrays(arrays_path, policy._episode_logits, actions, CARD_ARMS[args.card][arm][2])
            payload = {
                "protocol_id": lock["protocol_id"], "task": args.task, "seed": seed, "episode_id": seed,
                "arm": arm, "success": bool(result["success"]), "result": jsonable(result),
                "failure_reason": reason, "steps": steps, "runtime_seconds": time.monotonic() - started,
                "worker_id": args.worker_id, "gpu_id": args.gpu, "environment_id": environment_id,
                "canonical_snapshot_sha256": expected[0], "initial_state_sha256": state_hash,
                "initial_rgb_sha256": rgb_hash, "instruction": instruction,
                "lambda": CARD_ARMS[args.card][arm][2], "attention_layers": [11],
                "rendering": {"shader_dir": "rt", "rt_max_path_depth": RT_DEPTH[args.task], "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
                "selector": "L11 prompt attention + canonical dynamic Matched budget",
                "counterfactual_constructor": CARD_ARMS[args.card][arm][0],
                "guidance_execution": CARD_ARMS[args.card][arm][1],
                "arrays_file": arrays_path.name, "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(x.get("actual_selected_count") for x in trace),
                "selector_trace": trace, **checks,
            }
            atomic_json(summary_path, payload)
            pair_hashes.append((state_hash, rgb_hash))
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": payload["success"], "steps": steps, "runtime_seconds": round(payload["runtime_seconds"], 2)}, sort_keys=True), flush=True)
        if len(set(pair_hashes)) != 1:
            raise RuntimeError(f"arm pair identity mismatch: {args.task}:{seed}: {pair_hashes}")


if __name__ == "__main__":
    main()
