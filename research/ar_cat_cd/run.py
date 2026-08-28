#!/usr/bin/env python3
"""CAT-CD Phase-0 compute loop: per-state clean / pixel / 256-token attribution /
9 negative branches, dumped to per-state npz.

Run (smoke):
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/ar_cat_cd/run.py \
      --artifact artifacts/cat_cd_phase0_v1 --split confirmation --limit 1

Run (full, background):
  ... --artifact artifacts/cat_cd_phase0_v1 --split confirmation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from research.cw_lpcd.core import (
    PCD_ROOT, CHECKPOINT, load_model, action_token_slice, generate_clean_ids,
    ensure_empty_action_token,
)
from research.cw_lpcd.engine import load_position_mean, random_seed_for
from research.ar_cat_cd.core import (
    forward_clean, forward_masked, residual, residual_norm,
)

STATES_LOCK = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl")
MEAN_PATH = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/position_conditioned_visual_mean.pt")
KS = (4, 8, 16)


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
    npz_dir = art / "cat_cd_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads((art / "FROZEN_SPLIT.json").read_text()) if (art / "FROZEN_SPLIT.json").exists() else None
    if split is None:
        # fall back to the canonical cw_lpcd split (frozen, identical protocol)
        split = json.loads((Path("artifacts/cw_lpcd_v1/FROZEN_SPLIT.json")).read_text())
        (art / "FROZEN_SPLIT.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")

    if a.state_id:
        state_ids = [a.state_id]
    else:
        key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
        state_ids = split[key][: a.limit] if a.limit else split[key]

    states = {r["state_id"]: r for r in read_jsonl(STATES_LOCK)}
    manifest = {r["state_id"]: r for r in read_jsonl(Path("artifacts/cw_lpcd_v1/object_token_manifest.jsonl"))}

    torch.manual_seed(20260823)
    torch.cuda.manual_seed_all(20260823)
    model, processor = load_model(CHECKPOINT)
    mean = load_position_mean(a.mean_path.resolve()).to(model.device, dtype=torch.bfloat16)
    token_slice = action_token_slice(model)
    print(json.dumps({"model_loaded": True, "vocab_size": int(model.vocab_size),
                      "n_states": len(state_ids), "split": a.split}), flush=True)

    n_fwd = 0
    t0 = time.time()
    for ordinal, sid in enumerate(state_ids):
        out_npz = npz_dir / f"{sid}.npz"
        if out_npz.exists():
            print(json.dumps({"skip": sid, "done": ordinal + 1, "of": len(state_ids)}), flush=True)
            continue
        row = states[sid]
        obj_ids = manifest[sid]["object_token_ids"]
        st0 = time.time()

        clean_img = cv2.cvtColor(cv2.imread(str(pcd_root / row["clean_path"])), cv2.COLOR_BGR2RGB)
        pixel_img = cv2.cvtColor(cv2.imread(str(pcd_root / row["pixel_path"])), cv2.COLOR_BGR2RGB)
        instruction = row["instruction"]

        clean_inputs = None  # (rebuilt inside forward_* for hook cleanliness)
        # generate shared vanilla prefix once
        from research.cw_lpcd.core import inputs_for
        ci = inputs_for(processor, model, clean_img, instruction)
        clean_ids = generate_clean_ids(model, ci)
        n_fwd += 1

        clean_a, attn = forward_clean(model, processor, clean_img, instruction, clean_ids, record_attention=True)
        n_fwd += 1
        pixel_a, _ = forward_clean(model, processor, pixel_img, instruction, clean_ids, record_attention=False)
        n_fwd += 1
        r_pixel = residual(clean_a, pixel_a, token_slice)  # [7,256] float64

        # ---- 256-token leave-one-out attribution ----
        D = np.zeros(256, dtype=np.float32)
        for v in range(256):
            masked_a, _ = forward_masked(model, processor, clean_img, instruction, clean_ids, [v], mean)
            D[v] = residual_norm(clean_a, masked_a, token_slice)
        n_fwd += 256

        cat_order = np.argsort(-D, kind="stable")            # highest D first
        attn_order = np.argsort(-attn.numpy(), kind="stable")

        out = {
            "state_id": sid, "task": row["task"], "seed": int(row["seed"]),
            "clean_action": clean_a[:, token_slice].numpy().astype(np.float32),
            "pixel_action": pixel_a[:, token_slice].numpy().astype(np.float32),
            "r_pixel": r_pixel.numpy().astype(np.float32),
            "attention": attn.numpy().astype(np.float32),
            "object_ids": np.asarray(sorted(obj_ids), dtype=np.int64),
            "D": D,
            "clean_ids": clean_ids[0].cpu().numpy().astype(np.int64),
        }
        # ---- negative branches ----
        for k in KS:
            cat_ids = cat_order[:k].astype(np.int64)
            attn_ids = attn_order[:k].astype(np.int64)
            rnd_ids = np.random.default_rng(random_seed_for(sid) + k).choice(256, size=k, replace=False).astype(np.int64)
            cat_a, _ = forward_masked(model, processor, clean_img, instruction, clean_ids, cat_ids.tolist(), mean)
            attn_a, _ = forward_masked(model, processor, clean_img, instruction, clean_ids, attn_ids.tolist(), mean)
            rnd_a, _ = forward_masked(model, processor, clean_img, instruction, clean_ids, rnd_ids.tolist(), mean)
            n_fwd += 3
            out[f"cat_ids_{k}"] = cat_ids
            out[f"attn_ids_{k}"] = attn_ids
            out[f"rnd_ids_{k}"] = rnd_ids
            out[f"cat_action_{k}"] = cat_a[:, token_slice].numpy().astype(np.float32)
            out[f"attn_action_{k}"] = attn_a[:, token_slice].numpy().astype(np.float32)
            out[f"rnd_action_{k}"] = rnd_a[:, token_slice].numpy().astype(np.float32)

        np.savez_compressed(out_npz, **out)
        dt = time.time() - st0
        el = time.time() - t0
        print(json.dumps({"state_id": sid, "done": ordinal + 1, "of": len(state_ids),
                          "n_object": len(obj_ids), "sec": round(dt, 1),
                          "fwd": n_fwd, "rate": round(n_fwd / el, 1) if el else 0.0}), flush=True)

    print(json.dumps({"DONE": len(state_ids), "total_fwd": n_fwd, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
