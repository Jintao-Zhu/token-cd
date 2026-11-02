"""Official pi0-FAST vanilla policy cost per replan, measured the same way as the L6-Matched
replan benchmark, so the speed ratio between the two is apples-to-apples.

Episode records cannot give this: they include environment stepping and were collected with
three processes per GPU, so both arms carry contention. Here both policies are timed on the
same fixed state in the same process layout.
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

STATE = REPO / "artifacts/spatialvla_layer_selection_v1b/states/google_robot_open_drawer/seed_100"
OUT = REPO / "artifacts/pi0_fast_vanilla_timing_v1"
N = 8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phys = int(os.environ.get("BENCH_PHYS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "2")))
    pol = Pi0FastMatchedCDPolicy(gpu=phys)
    print(f"[van] kmeans={CD.KMEANS_BACKEND_REQUESTED} tap={pol.tap_mode} decode={pol.decode_mode}",
          flush=True)

    meta = json.loads((STATE / "meta.json").read_text())
    npz = np.load(STATE / "input.npz")
    image = np.asarray(npz["image"])
    raw = {
        "observation/primary_image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": meta["instruction"],
    }

    official = pol.policy
    van, mat = [], []
    for i in range(N):
        t0 = time.perf_counter()
        out_v = official.infer(dict(raw))
        van.append(time.perf_counter() - t0)
        assert np.isfinite(np.asarray(out_v["actions"])).all()

        CD.LAMBDA = 0.5
        pol.lambda_ = 0.5
        t0 = time.perf_counter()
        out_m = pol.infer(dict(raw), seed=200 + i)
        mat.append(time.perf_counter() - t0)
        tr = out_m["pi0_fast_matched_cd_trace"]
        assert tr["fast_action_stream_valid"], "matched stream invalid"

    vs, ms = van[1:], mat[1:]
    summary = {
        "protocol": "PI0_FAST_VANILLA_VS_MATCHED_TIMING_V1",
        "kmeans_backend": CD.KMEANS_BACKEND_REQUESTED,
        "tap_mode": pol.tap_mode,
        "decode_mode": pol.decode_mode,
        "vanilla_first_s": van[0],
        "vanilla_steady_s": vs,
        "vanilla_mean_s": statistics.fmean(vs),
        "matched_first_s": mat[0],
        "matched_steady_s": ms,
        "matched_mean_s": statistics.fmean(ms),
        "ratio_matched_over_vanilla": statistics.fmean(ms) / statistics.fmean(vs),
        "replans_per_episode": 29,
    }
    (OUT / "TIMING.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[van] vanilla steady = {statistics.fmean(vs):.2f}s  (first {van[0]:.2f}s)", flush=True)
    print(f"[van] matched steady = {statistics.fmean(ms):.2f}s  (first {mat[0]:.2f}s)", flush=True)
    print(f"[van] RATIO matched/vanilla = {summary['ratio_matched_over_vanilla']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
