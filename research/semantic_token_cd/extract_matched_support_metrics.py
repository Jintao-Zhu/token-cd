"""Extract feature-space support metrics for the 800 matched calibration states.

This script only replays frozen states and runs one projector forward per state.
It does not run environment continuation or use success labels.
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
from sklearn.cluster import KMeans

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, get_image_from_maniskill2_obs_dict, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy_floor import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import LAMBDA, atomic_json, load_reference
from research.semantic_token_cd.rollout_policy import extract_entities
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
TRAJ = {
    "google_robot_open_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_open_drawer/prompt_single",
    "google_robot_pick_coke_can": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_pick_coke_can/prompt_single",
    "google_robot_move_near": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_move_near/prompt_single",
    "google_robot_close_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes/google_robot_close_drawer/prompt_single",
}
KMEANS_K, KMEANS_SEED = 8, 0
GRID = 16


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


def spatial_metrics(tokens: np.ndarray) -> dict:
    tokens = np.asarray(sorted(set(int(x) for x in tokens)), dtype=np.int64)
    if not tokens.size:
        return {"bbox_area": 0, "mean_spatial_distance": 0.0, "components": 0, "density": 0.0}
    rows, cols = tokens // GRID, tokens % GRID
    bbox = int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
    center = np.asarray([rows.mean(), cols.mean()], dtype=np.float64)
    mean_dist = float(np.mean(np.linalg.norm(np.stack([rows, cols], axis=1) - center, axis=1)))
    remaining = set(int(x) for x in tokens)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            cur = stack.pop()
            r, c = divmod(cur, GRID)
            for nb in (cur - GRID if r > 0 else None, cur + GRID if r < GRID - 1 else None,
                       cur - 1 if c > 0 else None, cur + 1 if c < GRID - 1 else None):
                if nb is not None and nb in remaining:
                    remaining.remove(nb); stack.append(nb)
    return {"bbox_area": bbox, "mean_spatial_distance": mean_dist,
            "components": components, "density": float(len(tokens) / max(1, bbox))}


def cluster_metrics(features: np.ndarray, groups: list[int], labels: np.ndarray,
                    entity_groups: list[int], entity_cos: list[float], entity_margins: list[float]) -> dict:
    f = np.asarray(features, dtype=np.float64)
    fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    sizes = np.bincount(labels, minlength=KMEANS_K).astype(np.float64)
    p = sizes / sizes.sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    centroids = np.stack([f[labels == g].mean(axis=0) if np.any(labels == g) else np.zeros(f.shape[1])
                          for g in range(KMEANS_K)])
    cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    intra = []
    for g in groups:
        idx = np.flatnonzero(labels == g)
        if len(idx):
            intra.append(float(np.mean(1.0 - fn[idx] @ (centroids[g] / (np.linalg.norm(centroids[g]) + 1e-8)))))
    separation = []
    for g in groups:
        d = 1.0 - cn @ cn[g]
        other = np.sort(np.delete(d, g))
        if len(other) >= 2:
            separation.append(float(other[1] - other[0]))
    ranks = []
    order = np.argsort(-sizes)
    rank_map = {int(g): i + 1 for i, g in enumerate(order)}
    ranks = [rank_map[int(g)] for g in groups]
    return {
        "cluster_intra_cosine_distance": float(np.mean(intra)) if intra else None,
        "cluster_separation_margin": float(np.mean(separation)) if separation else None,
        "semantic_centroid_cosine": float(np.mean(entity_cos)) if entity_cos else None,
        "semantic_margin": float(np.mean(entity_margins)) if entity_margins else None,
        "cluster_size_entropy": entropy,
        "matched_cluster_size_rank_mean": float(np.mean(ranks)) if ranks else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", type=Path, required=True)
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
    cal = args.calibration.resolve(); root = args.artifact.resolve(); canonical = args.canonical.resolve()
    task = args.task
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); seeds.extend(range(lo, hi + 1))
        elif part.strip(): seeds.append(int(part))
    seeds = sorted(set(seeds))
    out_dir = root / "metrics" / task; out_dir.mkdir(parents=True, exist_ok=True)
    env, environment_id = make_environment(task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    policy = build_policy(OpenVLAInference(**config), task)
    tok = policy.processor.tokenizer
    traj_dir = repo / TRAJ[task]

    for si, seed in enumerate(seeds):
        if args.shard_count > 1 and si % args.shard_count != args.shard_index:
            continue
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
        target_steps = sorted({min(L - 1, int(round((L - 1) * f))) for f in (0.10, 0.35, 0.60, 0.85)})
        for t in range(L):
            if t not in target_steps:
                obs = env.step(np.asarray(actions[t]))[0]
                continue
            dst = out_dir / f"seed_{seed:03d}_step_{t:03d}.json"
            if dst.exists():
                obs = env.step(np.asarray(actions[t]))[0]; continue
            started = time.monotonic()
            image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
            inputs = policy.process_inputs(image, task_description=instruction)
            with projector_intervention(policy.vla) as trace:
                policy.vla(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                           pixel_values=inputs["pixel_values"], use_cache=False, return_dict=True)
            h = trace.before[0].detach().float().cpu().numpy().astype(np.float64)
            labels = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_SEED, n_init=10).fit(h).labels_
            centroids = np.stack([h[labels == g].mean(axis=0) if np.any(labels == g) else np.zeros(h.shape[1])
                                  for g in range(KMEANS_K)])
            cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
            entities = extract_entities(instruction); groups = []; cosines = []; margins = []
            for entity in entities:
                ids = tok(entity, add_special_tokens=False)["input_ids"] or tok(entity, add_special_tokens=True)["input_ids"]
                emb = policy.vla.language_model.model.embed_tokens(torch.tensor([ids], dtype=torch.long, device=policy.vla.device))[0].mean(0).detach().float().cpu().numpy()
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                sims = cn @ emb; g = int(np.argmax(sims))
                if g not in groups: groups.append(g)
                cosines.append(float(sims[g])); margins.append(float(sims[g] - np.max(np.delete(sims, g))))
            G = np.flatnonzero(np.isin(labels, groups)).astype(np.int64)
            sm = spatial_metrics(G)
            cm = cluster_metrics(h, groups, labels, groups, cosines, margins)
            payload = {"task": task, "seed": int(seed), "step": int(t), "progress": float(t / max(1, L - 1)),
                       "instruction": instruction, "entities": entities, "selected_group_ids": groups,
                       "m_matched": int(len(G)), "G_token_ids": G.tolist(), "labels": labels.astype(np.int16).tolist(),
                       "group_sizes": np.bincount(labels, minlength=KMEANS_K).astype(int).tolist(),
                       "environment_id": environment_id,
                       "runtime_seconds": time.monotonic() - started, **sm, **cm}
            atomic_json(dst, payload)
            np.savez_compressed(dst.with_suffix(".npz"), G=G, labels=labels.astype(np.int16),
                                group_sizes=np.bincount(labels, minlength=KMEANS_K).astype(np.int16))
            print(json.dumps({"task":task,"seed":int(seed),"step":int(t),"m":int(len(G)),
                              "sec":round(payload["runtime_seconds"],2)}), flush=True)
            obs = env.step(np.asarray(actions[t]))[0]


if __name__ == "__main__":
    main()
