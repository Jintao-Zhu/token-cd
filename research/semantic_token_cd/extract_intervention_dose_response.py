"""Offline intervention-dose response over the 800 matched calibration states.

For every frozen state we vary only m and measure:
  D_feat    relative Frobenius visual-feature change
  D_action  mean JS divergence of the first six action distributions
  D_support log-probability drop of the clean greedy action under the negative branch
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, get_image_from_maniskill2_obs_dict, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy_floor import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import LAMBDA, atomic_json, load_reference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
TRAJ = {
    "google_robot_open_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_open_drawer/prompt_single",
    "google_robot_pick_coke_can": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_pick_coke_can/prompt_single",
    "google_robot_move_near": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_move_near/prompt_single",
    "google_robot_close_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes/google_robot_close_drawer/prompt_single",
}
PROGRESS = (0.10, 0.35, 0.60, 0.85)
RELATIVE_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), 1e-12, None); p /= p.sum()
    q = np.clip(q.astype(np.float64), 1e-12, None); q /= q.sum()
    mid = 0.5 * (p + q)
    def kl(a, b): return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)


def top_p_count(scores: np.ndarray, tau: float) -> int:
    p = np.asarray(scores, dtype=np.float64); p /= (p.sum() + 1e-12)
    order = np.lexsort((np.arange(len(p)), -p))
    return int(np.searchsorted(np.cumsum(p[order]), float(tau) - 1e-12) + 1)


def build_policy(base, task: str):
    p = copy.copy(base)
    p.__class__ = PromptAttentionSHRInference
    _init_common(p, LAMBDA)
    p.beta = 0.0
    p.selector_mode = "prompt_attention"
    p.task_index = TASK_INDEX[task]
    p.attention_layers = (11,)
    p.selection_top_p = None
    p.selection_budget_schedule = None
    p.selection_budget_scale = 1.0
    p.selection_budget_source = "matched"
    p.selection_budget_entities = None
    p.selection_budget_min_count = None
    p.selection_budget_max_count = None
    p.selection_budget_small_threshold = None
    p.selection_budget_small_scale = 1.0
    p.selection_extra_uniform_count = None
    p.selection_spatial_mode = None
    p.selection_spatial_zero_ids = None
    p.save_prompt_attention = False
    return p


def run_m(policy, image, instruction, seed, step, m):
    policy.selection_count = int(m)
    policy.reset(instruction, seed=seed)
    policy._selector_step = int(step)
    policy._episode_trace = []; policy._episode_logits = []
    policy.step(image, None, instruction)
    meta = policy._episode_trace[-1]; rec = policy._episode_logits[-1]
    pos = np.asarray(rec["positive"], dtype=np.float64)[:6]
    neg = np.asarray(rec["negative"], dtype=np.float64)[:6]
    ppos = softmax(pos); pneg = softmax(neg)
    action = pos.argmax(axis=1)
    support = np.log(ppos[np.arange(6), action] + 1e-12) - np.log(pneg[np.arange(6), action] + 1e-12)
    return {
        "m": int(m),
        "actual_selected_count": int(meta["actual_selected_count"]),
        "D_feat": float(meta["feature_perturbation_relative"]),
        "D_action": float(np.mean([js_divergence(ppos[q], pneg[q]) for q in range(6)])),
        "D_support": float(np.mean(support)),
        "selected_token_ids": [int(x) for x in meta["selected_token_ids"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--calibration-results", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    repo = Path(__file__).resolve().parents[2]
    artifact = args.artifact.resolve(); canonical = args.canonical.resolve()
    calibration = args.calibration.resolve(); cal_results = json.loads(args.calibration_results.read_text())
    tau_entity = float(cal_results["tau"]["entity"])
    task = args.task
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); seeds.extend(range(lo, hi + 1))
        elif part.strip(): seeds.append(int(part))
    seeds = sorted(set(seeds))
    out_dir = artifact / "dose" / task; out_dir.mkdir(parents=True, exist_ok=True)
    env, environment_id = make_environment(task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    policy = build_policy(OpenVLAInference(**config), task)
    traj_dir = repo / TRAJ[task]

    for seed in seeds:
        dst = out_dir / f"seed_{seed:03d}.json"
        if dst.exists():
            print(json.dumps({"skip": True, "task": task, "seed": seed}), flush=True); continue
        actions = np.load(traj_dir / f"episode_{seed:03d}_arrays.npz")["executed_actions"]
        ref = load_reference(canonical, task, seed)
        with (canonical / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as fh: snap = pickle.load(fh)
        if snapshot_sha(snap) != ref["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snap)
        if (state_sha, rgb_sha) != (ref["initial_state_sha256"], ref["initial_rgb_sha256"]):
            raise RuntimeError("restored snapshot mismatch")
        instruction = env.unwrapped.get_language_instruction()
        L = len(actions); targets = sorted({min(L - 1, int(round((L - 1) * f))) for f in PROGRESS})
        states = []
        for t in range(L):
            if t not in targets:
                obs = env.step(np.asarray(actions[t]))[0]; continue
            started = time.monotonic()
            image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
            policy.selection_count = None
            policy.reset(instruction, seed=seed); policy._selector_step = t; policy._episode_trace = []; policy._episode_logits = []
            policy.step(image, None, instruction)
            m0 = int(policy._episode_trace[-1]["actual_selected_count"])
            cal_npz = calibration / "states" / task / f"seed_{seed:03d}_step_{t:03d}.npz"
            if not cal_npz.exists():
                raise FileNotFoundError(cal_npz)
            m_entity = top_p_count(np.load(cal_npz)["a_entity"], tau_entity)
            values = {}
            for factor in RELATIVE_FACTORS:
                m = max(1, min(256, int(round(factor * m0))))
                values.setdefault(m, []).append(f"relative_{factor:g}")
            values.setdefault(max(1, min(256, 34)), []).append("fixed34")
            values.setdefault(max(1, min(256, m_entity)), []).append("entity")
            curve = []
            for m in sorted(values):
                rec = run_m(policy, image, instruction, seed, t, m)
                rec["sources"] = values[m]
                rec["m_ratio"] = float(m / max(1, m0))
                curve.append(rec)
            states.append({
                "step": int(t), "progress": float(t / max(1, L - 1)), "matched_m": m0,
                "entity_m": int(m_entity), "curve": curve,
                "attention_sha256": hashlib.sha256(np.ascontiguousarray(policy._episode_logits[-1]["prompt_attention"] if "prompt_attention" in policy._episode_logits[-1] else np.zeros(256)).tobytes()).hexdigest(),
                "runtime_seconds": time.monotonic() - started,
            })
            print(json.dumps({"task":task,"seed":int(seed),"step":int(t),"m0":m0,
                              "m_entity":int(m_entity),"n_m":len(curve),"sec":round(states[-1]["runtime_seconds"],1)}), flush=True)
            obs = env.step(np.asarray(actions[t]))[0]
        atomic_json(dst, {"task":task,"seed":int(seed),"environment_id":environment_id,
                          "instruction":instruction,"canonical_snapshot_sha256":ref["canonical_snapshot_sha256"],
                          "initial_state_sha256":state_sha,"initial_rgb_sha256":rgb_sha,
                          "relative_factors":list(RELATIVE_FACTORS),"states":states})


if __name__ == "__main__":
    main()
