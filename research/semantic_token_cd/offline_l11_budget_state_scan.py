#!/usr/bin/env python3
"""Same-state L11 Matched budget multiplier response scan on saved states."""
from __future__ import annotations

import argparse
import hashlib
import glob
import json
import os
from pathlib import Path

import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=9)
    ap.add_argument("--states-file", type=Path, default=None,
                    help="optional newline-separated metadata paths for a small preflight subset")
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd import l11_table2_openvla_rollout as table2
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    library = args.library.resolve()
    if args.states_file is not None:
        paths = [Path(line.strip()) for line in args.states_file.read_text().splitlines() if line.strip()]
    else:
        all_paths = sorted(glob.glob(str(library / "states" / "*" / "*.json")))
        paths = [Path(p) for i, p in enumerate(all_paths) if i % args.shards == args.shard]
    if not paths:
        raise RuntimeError(f"shard {args.shard} has no states")
    # Normalize the first six continuous action dimensions by the empirical
    # clean rollout scale for each task; gripper is excluded from action dose.
    action_scales = {}
    for task in table2.FIG4_TASKS:
        chunks = []
        for p in (table2.ROOT / "artifacts/l11_table2_main_ablation_v1/formal/full/episodes" / task / "full").glob("*_arrays.npz"):
            with np.load(p, allow_pickle=False) as a:
                if "executed_actions" in a and a["executed_actions"].ndim == 2:
                    chunks.append(a["executed_actions"][:, :6])
        if chunks:
            scale = np.std(np.concatenate(chunks, axis=0), axis=0)
        else:
            scale = np.ones(6, dtype=np.float32)
        action_scales[task] = np.maximum(scale, 1e-3)

    base = None
    policies = {}
    result_path = args.output / f"shard_{args.shard:02d}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a") as out:
        for index, meta_path in enumerate(paths, 1):
            meta = json.loads(meta_path.read_text())
            task = meta["task"]
            if base is None:
                cfg = get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), task, {}, False)
                base = OpenVLAInference(**cfg)
            if task not in policies:
                policies[task] = table2.build_policies(base, task, "6", ("full",))["full"]
            policy = policies[task]
            state_path = meta_path.with_suffix(".npz")
            with np.load(state_path, allow_pickle=False) as arr:
                rgb = arr["rgb"]
                proprio = arr["eef_pos"]
            for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5):
                policy.selection_budget_scale = multiplier
                policy.selection_budget_label = f"dose_{multiplier:g}"
                policy.reset(meta["instruction"], seed=int(meta["seed"]))
                policy._episode_trace, policy._episode_logits = [], []
                _raw, actions, trace = policy.step(
                    rgb, None, meta["instruction"], proprio=proprio
                )
                if not isinstance(actions, list):
                    actions = [actions]
                if len(policy._episode_trace) != 1 or len(policy._episode_logits) != 1:
                    raise RuntimeError("single-state inference should emit exactly one trace/logit record")
                tr = policy._episode_trace[0]
                logits = policy._episode_logits[0]
                pos = np.asarray(logits["positive"], dtype=np.float64)[:6]
                neg = np.asarray(logits["negative"], dtype=np.float64)[:6]
                pp, pn = softmax(pos), softmax(neg)
                mid = (pp + pn) / 2
                js = 0.5 * np.sum(pp * np.log(np.maximum(pp / mid, 1e-12)), axis=-1)
                js += 0.5 * np.sum(pn * np.log(np.maximum(pn / mid, 1e-12)), axis=-1)
                clean_action = np.asarray(tr["clean_action"], dtype=np.float64).reshape(-1)
                guided_action = np.asarray(tr["guided_action"], dtype=np.float64).reshape(-1)
                delta = guided_action[:6] - clean_action[:6]
                row = {
                    "task": task, "seed": int(meta["seed"]), "step": int(meta["step"]),
                    "phase_quartile": int(meta["phase_quartile"]),
                    "episode_success": bool(meta["episode_success"]),
                    "multiplier": multiplier, "m_matched": int(tr["unscaled_matched_m_t"]),
                    "m_used": int(tr["m_t"]), "D_feat": float(tr["D_feat"]),
                    "D_feat_relative": float(tr["feature_perturbation_relative"]),
                    "D_action_raw": float(np.linalg.norm(delta)),
                    "D_action_scaled_l2": float(np.linalg.norm(delta / action_scales[task])),
                    "per_dim_action_delta": delta.tolist(),
                    "per_dim_action_delta_scaled": (delta / action_scales[task]).tolist(),
                    "mean_action_logit_js": float(np.mean(js)),
                    "action_token_flip_rate": float(tr["guided_change_ratio"]),
                    "guided_changed_dims": int(tr["guided_changed_dims"]),
                    "selected_token_ids": list(tr["selected_token_ids"]),
                    "l11_score_sha256": tr["l11_score_sha256"],
                    "positive_action_logits_sha256": hashlib.sha256(np.ascontiguousarray(pos).tobytes()).hexdigest(),
                    "initial_rgb_sha256": meta["initial_rgb_sha256"],
                    "state_rgb_sha256": meta["state_rgb_sha256"],
                }
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
            print(json.dumps({"event": "state_complete", "shard": args.shard,
                              "index": index, "total": len(paths), "task": task,
                              "seed": meta["seed"], "step": meta["step"]}), flush=True)


if __name__ == "__main__":
    main()
