#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-1 selector feature extraction (compute, GPU).

Per Confirmation state: re-extract projector features h_i [256,4096] (1 forward),
build per-group features for K-means@8 (visual mean v_i, spatial centroid, size),
compute the language embedding l (mean-pooled LLM token embeddings of the
instruction) and the group removal impact D_i = ||r_i||_F (offline from the
Phase-0 stored masked branches). Dumps per-state group_features.npz.

No rollout, no guidance, no training. Reuses semantic_npz (labels / masked /
object_ids / clean_action / r_pixel / g_obj_idx).

Run (smoke):
  cd /home/leju-suzhou/zjt_ws/token-cd
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/selector.py \
      --artifact artifacts/semantic_token_cd_phase1_selector_v1 --split confirmation --limit 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from research.cw_lpcd.core import PCD_ROOT, CHECKPOINT, load_model
from research.semantic_token_cd.core import extract_features

STATES_LOCK = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl")
SEMANTIC_NPZ = Path("artifacts/semantic_token_cd_phase0_v1/semantic_npz")
SPLIT = Path("artifacts/cat_cd_phase0_v1/FROZEN_SPLIT.json")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def log_softmax_residual(clean: np.ndarray, branch: np.ndarray) -> np.ndarray:
    ca = torch.tensor(clean, dtype=torch.float64)
    ba = torch.tensor(branch, dtype=torch.float64)
    return (torch.log_softmax(ca, -1) - torch.log_softmax(ba, -1)).numpy()


def language_embedding(model, tokenizer, instruction: str) -> torch.Tensor:
    """l = mean-pooled LLM token embedding of the instruction, in the same 4096-dim
    joint space as the projector output h_i."""
    ids = tokenizer(instruction, add_special_tokens=True)["input_ids"]
    ids_t = torch.tensor([ids], dtype=torch.long, device=model.device)
    embed = model.language_model.model.embed_tokens(ids_t)  # [1, L, 4096]
    return embed[0].mean(dim=0).detach().float().cpu()  # [4096]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pcd-root", type=Path, default=PCD_ROOT)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    p.add_argument("--limit", type=int)
    p.add_argument("--state-id", type=str)
    a = p.parse_args()

    pcd_root = a.pcd_root.resolve()
    art = a.artifact.resolve()
    feat_dir = art / "group_features_npz"
    feat_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT.read_text())
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
    print(json.dumps({"model_loaded": True, "n_states": len(state_ids), "split": a.split}), flush=True)

    t0 = time.time()
    for ordinal, sid in enumerate(state_ids):
        out_npz = feat_dir / f"{sid}.npz"
        if out_npz.exists():
            print(json.dumps({"skip": sid, "done": ordinal + 1}), flush=True)
            continue
        sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
        row = states[sid]
        instruction = row["instruction"]
        clean_img = cv2.cvtColor(cv2.imread(str(pcd_root / row["clean_path"])), cv2.COLOR_BGR2RGB)

        clean_ids = torch.tensor(sem["clean_ids"], dtype=torch.long, device=model.device).unsqueeze(0)
        h_i, _A = extract_features(model, processor, clean_img, instruction, clean_ids)  # [256,4096]
        h = h_i.numpy().astype(np.float64)
        l = language_embedding(model, processor.tokenizer, instruction).numpy().astype(np.float64)

        labels = sem["labels_kmeans_8"].astype(np.int64)
        masked = sem["masked_kmeans_8"]   # [8,7,256]
        clean_action = sem["clean_action"]
        r_pixel = sem["r_pixel"]

        v = np.zeros((8, h.shape[1]), dtype=np.float64)
        centroid = np.zeros((8, 2), dtype=np.float64)
        size = np.zeros(8, dtype=np.int64)
        D = np.zeros(8, dtype=np.float64)
        for k in range(8):
            idx = np.flatnonzero(labels == k)
            size[k] = len(idx)
            v[k] = h[idx].mean(axis=0)
            centroid[k] = np.stack([idx % 16, idx // 16]).mean(axis=1)
            D[k] = float(np.linalg.norm(log_softmax_residual(clean_action, masked[k])))

        np.savez_compressed(
            out_npz,
            state_id=sid, task=str(sem["task"]), seed=int(sem["seed"]),
            v=v.astype(np.float32), l=l.astype(np.float32),
            centroid=centroid.astype(np.float32), size=size,
            D=D.astype(np.float32), labels=labels, g_obj_idx=sem["g_obj_idx_kmeans_8"],
            object_ids=sem["object_ids"], clean_action=clean_action, r_pixel=r_pixel, masked=masked,
        )
        print(json.dumps({"state_id": sid, "done": ordinal + 1, "of": len(state_ids),
                          "sec": round(time.time() - t0, 1)}), flush=True)

    print(json.dumps({"DONE": len(state_ids), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
