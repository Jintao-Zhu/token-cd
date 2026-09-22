"""Extract, for each frozen state, the three candidate budgets.

  m_matched : KMeans(entity-matched cluster) size          [existing estimator]
  m_relative: #{ i : p_i > gamma/256 } over the L11 attention (calibrated gamma)
  m_spectral: min k s.t. top-k singular values explain >= tau of ||V||_F^2

One projector forward per state yields BOTH the visual feature matrix V[256,d]
and the layer-11 prompt attention, so nothing is recomputed twice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, get_image_from_maniskill2_obs_dict, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy_floor import (
    PromptAttentionSHRInference, prompt_query_layout, N_VISUAL,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import LAMBDA, atomic_json, load_reference
from research.semantic_token_cd.rollout_policy import extract_entities
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from sklearn.cluster import KMeans

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
TRAJ = {
    "google_robot_open_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_open_drawer/prompt_single",
    "google_robot_pick_coke_can": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_pick_coke_can/prompt_single",
    "google_robot_move_near": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_move_near/prompt_single",
    "google_robot_close_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes/google_robot_close_drawer/prompt_single",
}
PROGRESS = (0.10, 0.35, 0.60, 0.85)
GAMMA_GRID = [round(x, 3) for x in np.arange(0.50, 3.001, 0.05)]
TAU_GRID = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.998, 0.999]
KMEANS_K, KMEANS_SEED = 8, 0


def build_policy(base, task: str):
    p = base
    p.__class__ = PromptAttentionSHRInference
    _init_common(p, LAMBDA)
    p.beta = 0.0
    p.selector_mode = "prompt_attention"
    p.task_index = TASK_INDEX[task]
    p.attention_layers = (11,)
    p.kmeans_K, p.kmeans_seed = KMEANS_K, KMEANS_SEED
    p.save_prompt_attention = False
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    repo = Path(__file__).resolve().parents[2]
    root = args.artifact.resolve(); canonical = args.canonical.resolve()
    task = args.task
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); seeds.extend(range(lo, hi + 1))
        elif part.strip():
            seeds.append(int(part))
    seeds = sorted(set(seeds))
    out_dir = root / "estimators" / task
    out_dir.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    policy = build_policy(OpenVLAInference(**config), task)
    tok = policy.processor.tokenizer
    special = set(int(v) for v in tok.all_special_ids)
    traj_dir = repo / TRAJ[task]

    for si, seed in enumerate(seeds):
        if args.shard_count > 1 and si % args.shard_count != args.shard_index:
            continue
        dst = out_dir / f"seed_{seed:03d}.json"
        if dst.exists():
            print(json.dumps({"skip": True, "task": task, "seed": seed}), flush=True); continue
        actions = np.load(traj_dir / f"episode_{seed:03d}_arrays.npz")["executed_actions"]
        ref = load_reference(canonical, task, seed)
        with (canonical / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as fh:
            snap = pickle.load(fh)
        if snapshot_sha(snap) != ref["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snap)
        if (state_sha, rgb_sha) != (ref["initial_state_sha256"], ref["initial_rgb_sha256"]):
            raise RuntimeError("restored snapshot mismatch")
        instruction = env.unwrapped.get_language_instruction()
        L = len(actions)
        targets = sorted({min(L - 1, int(round((L - 1) * f))) for f in PROGRESS})
        states = []
        for t in range(L):
            if t in targets:
                t0 = time.monotonic()
                image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
                inputs = policy.process_inputs(image, task_description=instruction)
                with projector_intervention(policy.vla) as trace:
                    out = policy.vla(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"], use_cache=False,
                        output_attentions=True, return_dict=True,
                    )
                V = trace.before[0].detach().float().cpu().numpy()          # [256, d]
                text_idx, qpos, qtok = prompt_query_layout(inputs["input_ids"], special)
                a = out.attentions[11][0, :, qpos, 1:1 + N_VISUAL].detach().float().cpu().mean(dim=(0, 1)).numpy()
                # --- matched: KMeans + entity match ---
                hd = V.astype(np.float64)
                labels = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_SEED, n_init=10).fit(hd).labels_
                gv = np.stack([hd[labels == g].mean(axis=0) if (labels == g).any() else np.zeros(hd.shape[1])
                               for g in range(KMEANS_K)])
                gv = gv / (np.linalg.norm(gv, axis=1, keepdims=True) + 1e-8)
                groups = []
                for e in extract_entities(instruction):
                    ids = tok(e, add_special_tokens=False)["input_ids"]
                    if not ids:
                        ids = tok(e, add_special_tokens=True)["input_ids"]
                    et = torch.tensor([ids], dtype=torch.long, device=policy.vla.device)
                    emb = policy.vla.language_model.model.embed_tokens(et)[0].mean(dim=0).detach().float().cpu().numpy()
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    g = int(np.argmax(gv @ emb))
                    if g not in groups: groups.append(g)
                ref_idx = sorted(set(int(i) for g in groups for i in np.flatnonzero(labels == g)))
                m_matched = len(ref_idx)
                # --- relative: attention mass above gamma/256 ---
                p = a.astype(np.float64)
                p = p / (p.sum() + 1e-12)
                rel = {f"{g:.3f}": int((p > g / N_VISUAL).sum()) for g in GAMMA_GRID}
                # --- spectral: components for tau energy ---
                sv = np.linalg.svd(V.astype(np.float64), compute_uv=False)
                e = sv ** 2
                cum = np.cumsum(e) / (e.sum() + 1e-12)
                spec = {}
                for tau in TAU_GRID:
                    k = int(np.searchsorted(cum, tau) + 1)
                    spec[f"{tau:.3f}"] = min(k, N_VISUAL)
                states.append({
                    "step": int(t), "progress": float(t / max(1, L - 1)),
                    "m_matched": int(m_matched),
                    "attention_sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
                    "V_sha256": hashlib.sha256(np.ascontiguousarray(V).tobytes()).hexdigest(),
                    "m_relative_by_gamma": rel,
                    "m_spectral_by_tau": spec,
                    "sigma_first10": [float(x) for x in sv[:10]],
                    "sigma_sum": float(e.sum()),
                    "runtime_seconds": time.monotonic() - t0,
                })
                print(json.dumps({"task": task, "seed": seed, "step": t,
                                  "m_matched": int(m_matched),
                                  "sec": round(states[-1]["runtime_seconds"], 1)}), flush=True)
            obs = env.step(np.asarray(actions[t]))[0]
        atomic_json(dst, {"task": task, "seed": seed, "environment_id": environment_id,
                          "instruction": instruction, "trajectory_length": int(L),
                          "progress_points": list(PROGRESS),
                          "gamma_grid": GAMMA_GRID, "tau_grid": TAU_GRID,
                          "canonical_snapshot_sha256": ref["canonical_snapshot_sha256"],
                          "states": states})
        print(json.dumps({"done": True, "task": task, "seed": seed}), flush=True)


if __name__ == "__main__":
    main()
