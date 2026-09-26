#!/usr/bin/env python3
"""Audit historical canonical L11-Matched against its saved first-step records.

This deliberately calls PromptAttentionSHRInference.step() directly.  It does
not construct or import the task-conditioned policy and does not implement the
Matched budget or mask selector in this file.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.task_conditioned_contrast_protocol import (
    ATTENTION_LAYERS,
    CANONICAL,
    LAMBDA0,
    PCD_SOURCE,
    TASKS,
)

HISTORICAL = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/prompt_attn_l11_token_count_v1")
SEEDS = tuple(range(100, 105))
TOL = 1e-5


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def build_historical_policy(base, task: str):
    # Use the historical PromptAttention implementation's own step() path.
    from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common

    policy = copy.copy(base)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA0)
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = ATTENTION_LAYERS
    policy.selection_count = None
    policy.selection_top_p = None
    policy.selection_budget_schedule = None
    policy.selection_budget_scale = 1.0
    policy.selection_budget_label = None
    policy.selection_budget_source = "matched"
    policy.selection_budget_entities = None
    policy.save_prompt_attention = False
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=None,
                        help="Baseline root containing episodes/<task>/{vanilla,l11_matched}; defaults to historical artifacts.")
    parser.add_argument("--shader-dir", choices=("ibl", "rt"), default=None,
                        help="Optional diagnostic renderer override; None preserves each historical task default.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    repo = Path("/home/leju-suzhou/zjt_ws/token-cd")
    for path in (repo / "task1/shim_site", repo, PCD_SOURCE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import restore_snapshot, snapshot_sha
    from research.semantic_token_cd.spatial_grid_rollout import make_environment
    import research.semantic_token_cd.prompt_attn_shr_policy as historical_policy_module

    captured_harmonic = []
    original_harmonic = historical_policy_module.harmonic_reconstruct

    def capture_harmonic(features, region, *a, **kw):
        result = original_harmonic(features, region, *a, **kw)
        captured_harmonic.append({
            "input_sha256": array_sha256(np.asarray(features)),
            "region": [int(x) for x in region],
            "output_sha256": array_sha256(np.asarray(result)),
            "delta_norm": float(np.linalg.norm(np.asarray(result) - np.asarray(features)[np.asarray(region)])),
        })
        return result

    historical_policy_module.harmonic_reconstruct = capture_harmonic
    env, _env_id = make_environment(args.task, shader_dir=args.shader_dir)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_historical_policy(OpenVLAInference(**config), args.task)
    rows = []

    for seed in SEEDS:
        reference_root = args.reference_root.resolve() if args.reference_root else HISTORICAL
        hist_summary_path = reference_root / "episodes" / args.task / "l11_matched" / f"episode_{seed:03d}_summary.json"
        hist_arrays_path = reference_root / "episodes" / args.task / "l11_matched" / f"episode_{seed:03d}_arrays.npz"
        hist = json.loads(hist_summary_path.read_text())
        hist_first = hist["selector_trace"][0]
        hist_arrays = np.load(hist_arrays_path)
        vanilla_root = reference_root if args.reference_root else CANONICAL
        vanilla_summary_path = vanilla_root / "episodes" / args.task / "vanilla" / f"episode_{seed:03d}_summary.json"
        vanilla_arrays_path = vanilla_root / "episodes" / args.task / "vanilla" / f"episode_{seed:03d}_arrays.npz"
        vanilla_summary = json.loads(vanilla_summary_path.read_text())
        vanilla_arrays = np.load(vanilla_arrays_path)
        snapshot_path = CANONICAL / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
        with snapshot_path.open("rb") as f:
            snapshot = pickle.load(f)
        observed_snapshot_sha = snapshot_sha(snapshot)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        image = get_image_from_maniskill2_obs_dict(env, obs)

        captured_harmonic.clear()
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        _raw, _actions, meta = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        logits = policy._episode_logits[0]
        harmonic = captured_harmonic[-1] if captured_harmonic else None

        saved_selected = [int(x) for x in hist_first["selected_token_ids"]]
        saved_mask_selected = np.flatnonzero(np.asarray(hist_arrays["selected_mask"][0]) > 0).astype(int).tolist()
        live_selected = [int(x) for x in meta["selected_token_ids"]]
        clean_diff = max(max_abs(np.asarray(logits["positive"]), hist_arrays["positive"][0]),
                         max_abs(np.asarray(logits["positive"]), vanilla_arrays["positive"][0]))
        negative_diff = max_abs(np.asarray(logits["negative"]), hist_arrays["negative"][0])
        live_delta_norm = float(meta["feature_perturbation_norm"])
        stored_delta_norm = float(hist_first["feature_perturbation_norm"])
        # PromptAttentionSHRInference flattens selector metadata into `meta`.
        ranking_sha = meta["attention_sha256"]
        checks = {
            "task_key": hist["task"] == args.task,
            "canonical_snapshot": hist["canonical_snapshot_sha256"] == observed_snapshot_sha,
            "initial_state": hist["initial_state_sha256"] == state_sha == vanilla_summary["initial_state_sha256"],
            "initial_rgb": hist["initial_rgb_sha256"] == rgb_sha == vanilla_summary["initial_rgb_sha256"],
            "clean_logits": clean_diff < TOL,
            "l11_ranking_sha": ranking_sha == hist_first["attention_sha256"],
            "m_t": int(meta["m_t"]) == int(hist_first["m_t"]),
            "selected_token_ids": live_selected == saved_selected == saved_mask_selected,
            "harmonic": harmonic is not None and abs(live_delta_norm - stored_delta_norm) < TOL,
            "negative_logits": negative_diff < TOL,
            "final_action_tokens": [int(x) for x in meta["final_token_ids"]] == [int(x) for x in hist_first["final_token_ids"]],
        }
        rows.append({
            "task": args.task,
            "seed": seed,
            "checks": checks,
            "pass": all(checks.values()),
            "snapshot_sha256": observed_snapshot_sha,
            "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha,
            "clean_logits_max_abs_diff": clean_diff,
            "l11_ranking_sha256": ranking_sha,
            "historical_m_t": int(hist_first["m_t"]),
            "live_m_t": int(meta["m_t"]),
            "historical_selected_token_ids": saved_selected,
            "historical_selected_mask_ids": saved_mask_selected,
            "live_selected_token_ids": live_selected,
            "harmonic_output_sha256": harmonic["output_sha256"] if harmonic else None,
            "historical_feature_perturbation_norm": stored_delta_norm,
            "live_feature_perturbation_norm": live_delta_norm,
            "negative_logits_max_abs_diff": negative_diff,
            "historical_final_action_tokens": [int(x) for x in hist_first["final_token_ids"]],
            "live_final_action_tokens": [int(x) for x in meta["final_token_ids"]],
        })
        print(json.dumps({"task": args.task, "seed": seed, "pass": rows[-1]["pass"], "checks": checks,
                          "m_t": f"{meta['m_t']}/{hist_first['m_t']}",
                          "selected_ids_match": checks["selected_token_ids"],
                          "clean_max_abs_diff": clean_diff, "negative_max_abs_diff": negative_diff}), flush=True)

    out = args.output / "analysis" / f"CANONICAL_PREFLIGHT_{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"protocol_id": "L11_MATCHED_CANONICAL_PREFLIGHT_V1", "task": args.task,
               "gpu": args.gpu, "shader_dir_override": args.shader_dir,
               "reference_root": str(reference_root),
               "seeds": list(SEEDS), "n": len(rows),
               "n_pass": sum(bool(r["pass"]) for r in rows), "results": rows}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "n": len(rows), "n_pass": payload["n_pass"], "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
