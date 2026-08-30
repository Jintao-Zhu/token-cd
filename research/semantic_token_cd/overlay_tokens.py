"""Overlay the KMeans-selected semantic tokens on the overhead camera image.

Answers Q1: what do the selected tokens (the "drawer" entity group) actually cover
in the image — the handle, the drawer front panel, its edges, or background?

Mapping: OpenVLA feeds a 224x224 SigLIP image (cv.resize INTER_AREA from the 512x640
overhead frame, direct stretch, no letterbox). The projector emits 256 tokens on a
16x16 grid, row-major (token idx = r*16 + c). Each token therefore covers
[32 x 40] px of the 512x640 frame: y in [r*32,(r+1)*32), x in [c*40,(c+1)*40).

Usage:
  <venv>/bin/python research/semantic_token_cd/overlay_tokens.py \
    --gpu 1 --seeds 205,200 --out artifacts/attn_semantic_merge_k8_v1/token_overlay
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
for _p in (str(REPO_ROOT), str(PCD_SOURCE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK = "google_robot_close_drawer"
ARTIFACT = REPO_ROOT / "artifacts/attn_semantic_merge_k8_v1"
MERGE_ARM = "semantic_merge_k8_eta100"


def load_selected(seed):
    p = ARTIFACT / "episodes" / TASK / MERGE_ARM / f"episode_{seed:03d}_summary.json"
    s = json.loads(p.read_text())
    t0 = s["selector_trace"][0]
    return (t0.get("selected_token_ids", []),
            t0.get("selected_entities", []),
            t0.get("selected_group_sizes", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seeds", default="205")
    ap.add_argument("--out", type=Path,
                    default=ARTIFACT / "token_overlay")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import simpler_env
    from PIL import Image, ImageDraw

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",") if x]

    env = simpler_env.make(TASK)
    for seed in seeds:
        env.reset(seed=seed)
        obs = env.unwrapped.get_obs()
        cam = obs["image"]["overhead_camera"]
        raw = cam.get("rgb")
        if raw is None:
            raw = cam["Color"][..., :3]  # (H,W,3) float32 in [0,1]
            raw = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
        img = np.asarray(raw).copy()
        H, W = img.shape[:2]
        toks, ents, sizes = load_selected(seed)
        sel = set(toks)
        im = Image.fromarray(img).convert("RGB")
        dr = ImageDraw.Draw(im, "RGBA")
        for r in range(16):
            for c in range(16):
                idx = r * 16 + c
                x0, y0 = c * 40, r * 32
                x1, y1 = x0 + 40, y0 + 32
                if idx in sel:
                    dr.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 90),
                                 outline=(255, 0, 0, 200), width=2)
                else:
                    dr.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 40), width=1)
        im.save(out / f"seed_{seed:03d}_overlay.png")
        print(json.dumps({"seed": seed, "n_tokens": len(toks),
                          "entities": ents, "group_sizes": sizes,
                          "rows": sorted({t // 16 for t in toks}),
                          "cols": sorted({t % 16 for t in toks})}), flush=True)
    env.close()
    print(json.dumps({"DONE": True, "n": len(seeds), "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
