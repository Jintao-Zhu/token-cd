"""Phase A: frozen-state counterfactual response curve over the L11 matched budget.

For each sampled state we take the state's OWN L11 ranking and only slide the
cutoff m.  Everything else is the frozen L11-Matched configuration.  No
environment.step() is ever driven by the evaluated policy: the trajectory is
replayed with actions saved by the completed matched rollout.

Primary metric per m:  D(m) = mean_q JS( softmax(z+_q) || softmax(z-_q,m) ), q=1..6
i.e. how far the negative branch's action distribution has moved away from clean.
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

TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
TRAJ = {
    "google_robot_open_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_open_drawer/prompt_single",
    "google_robot_pick_coke_can": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_pick_coke_can/prompt_single",
    "google_robot_move_near": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_move_near/prompt_single",
    "google_robot_close_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes/google_robot_close_drawer/prompt_single",
}
PROGRESS = (0.10, 0.35, 0.60, 0.85)
COARSE_M = (4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p.astype(np.float64), eps, None); p = p / p.sum()
    q = np.clip(q.astype(np.float64), eps, None); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


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
    p.log_current_matched_m = True
    p.save_prompt_attention = True
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 0-49")
    ap.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    repo = Path(__file__).resolve().parents[2]
    artifact = args.artifact.resolve()
    canonical = args.canonical.resolve()
    task = args.task
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); seeds.extend(range(lo, hi + 1))
        elif part.strip():
            seeds.append(int(part))
    seeds = sorted(set(seeds))

    env, environment_id = make_environment(task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    policy = build_policy(OpenVLAInference(**config), task)

    out_dir = artifact / "phase_a" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = repo / TRAJ[task]

    for seed in seeds:
        dst = out_dir / f"seed_{seed:03d}.json"
        if dst.exists():
            print(json.dumps({"skip": True, "task": task, "seed": seed}), flush=True)
            continue
        traj_npz = traj_dir / f"episode_{seed:03d}_arrays.npz"
        traj_json = traj_dir / f"episode_{seed:03d}_summary.json"
        if not traj_npz.exists() or not traj_json.exists():
            print(json.dumps({"miss": True, "task": task, "seed": seed}), flush=True)
            continue
        actions = np.load(traj_npz)["executed_actions"]
        ref = load_reference(canonical, task, seed)
        with (canonical / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as fh:
            snapshot = pickle.load(fh)
        if snapshot_sha(snapshot) != ref["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if (state_sha, rgb_sha) != (ref["initial_state_sha256"], ref["initial_rgb_sha256"]):
            raise RuntimeError("restored snapshot mismatch")
        instruction = env.unwrapped.get_language_instruction()

        L = len(actions)
        targets = sorted({min(L - 1, int(round((L - 1) * f))) for f in PROGRESS})
        states = []
        for t in range(L):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            if t in targets:
                started = time.monotonic()
                # current-state matched budget (policy decides with its own m)
                policy.selection_count = None
                policy.reset(instruction, seed=seed)
                policy._selector_step = t
                policy._episode_trace = []; policy._episode_logits = []
                policy.step(image, None, instruction)
                meta = policy._episode_trace[-1]
                matched_m = int(meta["actual_selected_count"])
                ranking = np.asarray(policy._episode_logits[-1]["prompt_attention"], dtype=np.float64)
                # reference budgets at the SAME state (for the method comparison table)
                ref_m = {}
                for tag, thr in (("topp80", 0.80), ("topp85", 0.85)):
                    policy.selection_count = None
                    policy.selection_top_p = float(thr)
                    policy.reset(instruction, seed=seed)
                    policy._selector_step = t
                    policy._episode_trace = []; policy._episode_logits = []
                    policy.step(image, None, instruction)
                    ref_m[tag] = int(policy._episode_logits[-1]["selected_mask"].sum())
                policy.selection_top_p = None
                m_list = sorted(set(list(COARSE_M) + [matched_m]))
                curve = []
                for m in m_list:
                    policy.selection_count = int(m)
                    policy.reset(instruction, seed=seed)
                    policy._selector_step = t
                    policy._episode_trace = []; policy._episode_logits = []
                    policy.step(image, None, instruction)
                    rec = policy._episode_logits[-1]
                    pos = np.asarray(rec["positive"], dtype=np.float64)[:6]
                    neg = np.asarray(rec["negative"], dtype=np.float64)[:6]
                    D = float(np.mean([js_divergence(p, q) for p, q in zip(pos, neg)]))
                    resid = pos - neg
                    curve.append({
                        "m": int(m),
                        "D": D,
                        "residual_l2": float(np.linalg.norm(resid)),
                        "residual_centered_l2": float(np.linalg.norm(resid - resid.mean(axis=1, keepdims=True))),
                        "actual_selected_count": int(rec["selected_mask"].sum()),
                    })
                # residual-direction cosine between adjacent coarse budgets, centred
                by_m = {c["m"]: c for c in curve}
                cos_pairs = []
                for a, b in zip(COARSE_M[:-1], COARSE_M[1:]):
                    if a in by_m and b in by_m:
                        cos_pairs.append({"m_from": a, "m_to": b, "cos": None})
                states.append({
                    "step": int(t),
                    "topp80_m": int(ref_m.get("topp80", -1)),
                    "topp85_m": int(ref_m.get("topp85", -1)),
                    "progress": float(t / max(1, L - 1)),
                    "matched_m": matched_m,
                    "m_list": m_list,
                    "curve": curve,
                    "attention_sha256": meta.get("attention_sha256"),
                    "ranking_sha256": hashlib.sha256(np.ascontiguousarray(ranking).tobytes()).hexdigest(),
                    "image_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                    "runtime_seconds": time.monotonic() - started,
                })
                print(json.dumps({"task": task, "seed": seed, "step": t,
                                  "matched_m": matched_m, "n_m": len(m_list),
                                  "sec": round(states[-1]["runtime_seconds"], 1)}), flush=True)
            step_res = env.step(np.asarray(actions[t]))
            obs = step_res[0]
        payload = {
            "task": task, "seed": seed, "environment_id": environment_id,
            "instruction": instruction, "trajectory_length": int(L),
            "trajectory_source": str(traj_dir.relative_to(repo)),
            "progress_points": list(PROGRESS),
            "coarse_m": list(COARSE_M),
            "canonical_snapshot_sha256": ref["canonical_snapshot_sha256"],
            "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
            "states": states,
        }
        atomic_json(dst, payload)
        print(json.dumps({"done": True, "task": task, "seed": seed, "n_states": len(states)}), flush=True)


if __name__ == "__main__":
    main()
