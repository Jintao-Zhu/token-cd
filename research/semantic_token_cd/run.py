#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-0 compute loop.

Per Confirmation state: extract projector features h_i + attention graph A, group
the 256 visual tokens (A: kmeans K=8/16/32; B: cosine-agglo K=16; C: attention-
spectral K=16), mask each group + matched baselines (attention/random/full-object),
dump per-state npz. Reuses cat_cd_npz clean_ids / r_pixel / attention / object_ids.

Run (smoke):
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/run.py \
      --artifact artifacts/semantic_token_cd_phase0_v1 --split confirmation --limit 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from research.cw_lpcd.core import PCD_ROOT, CHECKPOINT, load_model, action_token_slice
from research.cw_lpcd.engine import load_position_mean, random_seed_for
from research.ar_cat_cd.core import forward_masked
from research.semantic_token_cd.core import extract_features
from research.semantic_token_cd.grouping import group_indices, group_overlap

STATES_LOCK = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl")
MEAN_PATH = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/position_conditioned_visual_mean.pt")
CAT_CD_NPZ = Path("artifacts/cat_cd_phase0_v1/cat_cd_npz")

# (method, K) combos. Method A (kmeans) at all K is the primary + K-ablation; B/C at K=16.
METHOD_KS = [("kmeans", 8), ("kmeans", 16), ("kmeans", 32), ("agglomerative", 16), ("spectral", 16)]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pcd-root", type=Path, default=PCD_ROOT)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    p.add_argument("--limit", type=int)
    p.add_argument("--state-id", type=str)
    p.add_argument("--mean-path", type=Path, default=MEAN_PATH)
    a = p.parse_args()

    pcd_root = a.pcd_root.resolve()
    art = a.artifact.resolve()
    npz_dir = art / "semantic_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads((Path("artifacts/cat_cd_phase0_v1/FROZEN_SPLIT.json")).read_text())
    (art / "FROZEN_SPLIT.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    if a.state_id:
        state_ids = [a.state_id]
    else:
        key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
        state_ids = split[key][: a.limit] if a.limit else split[key]

    states = {r["state_id"]: r for r in read_jsonl(STATES_LOCK)}

    torch.manual_seed(20260823)
    torch.cuda.manual_seed_all(20260823)
    model, processor = load_model(CHECKPOINT)
    mean = load_position_mean(a.mean_path.resolve()).to(model.device, dtype=torch.bfloat16)
    token_slice = action_token_slice(model)
    print(json.dumps({"model_loaded": True, "n_states": len(state_ids), "split": a.split}), flush=True)

    n_fwd = 0
    t0 = time.time()
    for ordinal, sid in enumerate(state_ids):
        out_npz = npz_dir / f"{sid}.npz"
        if out_npz.exists():
            print(json.dumps({"skip": sid, "done": ordinal + 1}), flush=True)
            continue
        st0 = time.time()
        cat = np.load(CAT_CD_NPZ / f"{sid}.npz")
        clean_ids = torch.tensor(cat["clean_ids"], dtype=torch.long, device=model.device).unsqueeze(0)  # [1,7]
        clean_action = cat["clean_action"]            # [7,256]
        r_pixel = cat["r_pixel"].astype(np.float32)   # [7,256]
        attention = cat["attention"]                  # [256]
        object_ids = cat["object_ids"].tolist()
        row = states[sid]
        instruction = row["instruction"]

        clean_img = cv2.cvtColor(cv2.imread(str(pcd_root / row["clean_path"])), cv2.COLOR_BGR2RGB)

        h_i, A = extract_features(model, processor, clean_img, instruction, clean_ids)
        n_fwd += 1

        out = {
            "state_id": sid, "task": str(cat["task"]), "seed": int(cat["seed"]),
            "clean_action": clean_action, "r_pixel": r_pixel, "attention": attention,
            "object_ids": np.asarray(sorted(object_ids), dtype=np.int64),
            "clean_ids": cat["clean_ids"],
        }

        for method, K in METHOD_KS:
            groups = group_indices(method, h_i.numpy(), A.numpy() if A is not None else None, K)
            label_arr = np.zeros(256, dtype=np.int64)
            for k, grp in enumerate(groups):
                label_arr[grp] = k
            # per-group masked action logits
            masked = np.zeros((K, 7, 256), dtype=np.float32)
            overlaps = np.zeros((K, 5), dtype=np.float64)  # prec, recall, iou, chance_prec, chance_iou
            for k, grp in enumerate(groups):
                ma, _ = forward_masked(model, processor, clean_img, instruction, clean_ids, grp, mean)
                masked[k] = ma[:, token_slice].numpy()
                ov = group_overlap(grp, object_ids)
                overlaps[k] = [ov["precision"], ov["recall"], ov["iou"], ov["chance_precision"], ov["chance_iou"]]
                n_fwd += 1
            # object group = argmax IoU
            g_obj_idx = int(np.argmax(overlaps[:, 2]))
            g_obj = groups[g_obj_idx]
            s = len(g_obj)
            # matched baselines: attention top-s, random s
            attn_order = np.argsort(-attention, kind="stable")
            attn_ids = attn_order[:s].astype(np.int64).tolist()
            rnd_ids = np.random.default_rng(random_seed_for(sid) + 1000 + K).choice(256, size=s, replace=False).astype(np.int64).tolist()
            attn_ma, _ = forward_masked(model, processor, clean_img, instruction, clean_ids, attn_ids, mean)
            rnd_ma, _ = forward_masked(model, processor, clean_img, instruction, clean_ids, rnd_ids, mean)
            n_fwd += 2

            out[f"labels_{method}_{K}"] = label_arr
            out[f"masked_{method}_{K}"] = masked
            out[f"overlap_{method}_{K}"] = overlaps
            out[f"g_obj_idx_{method}_{K}"] = np.asarray([g_obj_idx], dtype=np.int64)
            out[f"attn_ids_{method}_{K}"] = np.asarray(attn_ids, dtype=np.int64)
            out[f"attn_masked_{method}_{K}"] = attn_ma[:, token_slice].numpy().astype(np.float32)
            out[f"rnd_ids_{method}_{K}"] = np.asarray(rnd_ids, dtype=np.int64)
            out[f"rnd_masked_{method}_{K}"] = rnd_ma[:, token_slice].numpy().astype(np.float32)

        # full-object mask (persistent Token-PCD ceiling)
        full_ma, _ = forward_masked(model, processor, clean_img, instruction, clean_ids, object_ids, mean)
        out["fullobj_masked"] = full_ma[:, token_slice].numpy().astype(np.float32)
        n_fwd += 1

        np.savez_compressed(out_npz, **out)
        dt = time.time() - st0
        print(json.dumps({"state_id": sid, "done": ordinal + 1, "of": len(state_ids),
                          "n_object": len(object_ids), "sec": round(dt, 1),
                          "fwd": n_fwd, "rate": round(n_fwd / (time.time() - t0), 1)}), flush=True)

    print(json.dumps({"DONE": len(state_ids), "total_fwd": n_fwd, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
