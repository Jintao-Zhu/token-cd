"""Dump one guided-decode configuration for later A/B comparison.

Batch=2 changes XLA kernel selection, so the only acceptable evidence is token-level and
action-level equivalence on identical states. Because the decode loop is module_jit'd, the
flag cannot be flipped inside one process (the graph is cached), so this script records one
configuration and the comparison is done across two runs.

Usage: PI0FAST_CD_BATCH=<0|1> python pi0_fast_batch_validate.py <out_tag>
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
sys.path.insert(0, str(ROOT / "openpi/src"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import research.semantic_token_cd.pi0_fast_matched_cd as CD  # noqa: E402
from research.semantic_token_cd.pi0_fast_matched_cd import Pi0FastMatchedCDPolicy  # noqa: E402

POOL = REPO / "artifacts/spatialvla_layer_selection_v1b/states"
OUT = REPO / "artifacts/pi0_fast_batch_ab_v1"
LAM = 0.5
# one state per task keeps the A/B short while still spanning all nine tasks
TASKS = [
    "google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can",
    "google_robot_move_near", "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel",
    "widowx_stack_cube",
]
SEED = 100


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    OUT.mkdir(parents=True, exist_ok=True)
    phys = int(os.environ.get("BENCH_PHYS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "2")))
    pol = Pi0FastMatchedCDPolicy(gpu=phys)
    print(f"[ab] tag={tag} batch={CD.BATCH_BRANCHES} kmeans={CD.KMEANS_BACKEND_REQUESTED} "
          f"tap={pol.tap_mode} decode={pol.decode_mode}", flush=True)

    rows = []
    times = []
    for task in TASKS:
        d = POOL / task / f"seed_{SEED:03d}"
        if not (d / "input.npz").exists():
            print(f"[ab] skip {task}", flush=True)
            continue
        meta = json.loads((d / "meta.json").read_text())
        npz = np.load(d / "input.npz")
        image = np.asarray(npz["image"])
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        raw = {
            "observation/primary_image": image,
            "observation/wrist_image": np.zeros_like(image),
            "observation/state": np.zeros((8,), dtype=np.float32),
            "prompt": meta["instruction"],
        }
        CD.LAMBDA = LAM
        pol.lambda_ = LAM
        t0 = time.perf_counter()
        out = pol.infer(dict(raw), seed=SEED)
        dt = time.perf_counter() - t0
        tr = out["pi0_fast_matched_cd_trace"]
        act = np.asarray(out["actions"], dtype=np.float32)
        rows.append({
            "task": task,
            "instruction": meta["instruction"],
            "seconds": dt,
            "tokens": [int(x) for x in tr["raw_output_token_ids_full"]],
            "payload_chars": int(tr["decoded_fast_payload_chars"]),
            "stream_valid": bool(tr["fast_action_stream_valid"]),
            "decoder_agrees": bool(tr["openpi_official_decoder_agrees"]),
            "matched_m_t": int(tr["matched_m_t"]),
            "selected_token_ids": [int(x) for x in tr["selected_token_ids"]],
            "first_clean_top_token": int(tr["first_clean_top_token"]),
            "first_negative_top_token": int(tr["first_negative_top_token"]),
            "first_guided_top_token": int(tr["first_guided_top_token"]),
            "logit_residual_norm": float(tr["first_logit_residual_norm"]),
            "decode_target_delta": int(tr.get("decode_target_delta", 0)),
            "actions": act.tolist(),
            "action_absmax": float(np.max(np.abs(act))),
        })
        times.append(dt)
        print(f"[ab] {task.replace('google_robot_','').replace('widowx_','')}: {dt:.2f}s "
              f"m_t={tr['matched_m_t']} payload={tr['decoded_fast_payload_chars']} "
              f"valid={tr['fast_action_stream_valid']}", flush=True)

    summary = {
        "protocol": "PI0_FAST_BATCH_AB_V1",
        "tag": tag,
        "batch_branches": bool(CD.BATCH_BRANCHES),
        "kmeans_backend": CD.KMEANS_BACKEND_REQUESTED,
        "tap_mode": pol.tap_mode,
        "decode_mode": pol.decode_mode,
        "lambda": LAM,
        "states": len(rows),
        "mean_seconds": statistics.fmean(times) if times else None,
        "rows": rows,
    }
    (OUT / f"RUN_{tag}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[ab] wrote {OUT/('RUN_'+tag+'.json')}  mean={summary['mean_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
