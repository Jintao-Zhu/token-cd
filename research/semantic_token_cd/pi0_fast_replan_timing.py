"""Steady-state cost of one guided replan, as the closed loop actually experiences it.

The Gate 2 regression mixes tasks and instructions, so every new prefix length recompiles the
JIT functions and the measured seconds-per-stream are dominated by compilation. Inside a real
rollout the instruction is fixed for the episode, so compilation happens once and the
recurring cost is just the per-replan work.

This measures that recurring cost: one fixed state, infer() called repeatedly, reporting the
first call (cold, includes compilation) separately from the steady-state calls.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
sys.path.insert(0, str(ROOT / "openpi/src"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from research.semantic_token_cd.pi0_fast_matched_cd import Pi0FastMatchedCDPolicy  # noqa: E402

STATE = REPO / "artifacts/spatialvla_layer_selection_v1b/states/google_robot_open_drawer/seed_100"
OUT = REPO / "artifacts/pi0_fast_replan_timing_v1"
N_REPEAT = 8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phys = int(os.environ.get("BENCH_PHYS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "1")))
    pol = Pi0FastMatchedCDPolicy(gpu=phys)
    print(f"[bench] tap_mode={pol.tap_mode} decode_mode={pol.decode_mode} gpu={phys}", flush=True)

    meta = json.loads((STATE / "meta.json").read_text())
    npz = np.load(STATE / "input.npz")
    image = np.asarray(npz["image"])
    raw = {
        "observation/primary_image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": meta["instruction"],
    }

    times = []
    deltas = []
    for i in range(N_REPEAT):
        t0 = time.perf_counter()
        out = pol.infer(dict(raw), seed=100 + i)
        dt = time.perf_counter() - t0
        tr = out["pi0_fast_matched_cd_trace"]
        times.append(dt)
        deltas.append(int(tr["decode_target_delta"]))
        print(f"[bench] call {i+1}/{N_REPEAT}: {dt:7.2f}s  valid={tr['fast_action_stream_valid']} "
              f"agrees={tr['openpi_official_decoder_agrees']} delta={tr['decode_target_delta']} "
              f"m_t={tr['matched_m_t']}", flush=True)

    steady = times[1:]
    summary = {
        "protocol": "PI0_FAST_REPLAN_TIMING_V1",
        "tap_mode": pol.tap_mode,
        "decode_mode": pol.decode_mode,
        "first_call_s": times[0],
        "steady_calls_s": steady,
        "steady_mean_s": statistics.fmean(steady) if steady else None,
        "steady_median_s": statistics.median(steady) if steady else None,
        "decode_target_deltas": deltas,
        "exec_horizon": 4,
        "implied_decode_s_per_episode": (statistics.fmean(steady) * 24.25) if steady else None,
    }
    (OUT / "TIMING.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if steady:
        per_ep = statistics.fmean(steady) * 24.25
        print(f"[bench] steady per replan = {statistics.fmean(steady):.2f}s "
              f"(median {statistics.median(steady):.2f}s)", flush=True)
        print(f"[bench] x24.25 replans/episode = {per_ep:.0f}s decode per episode", flush=True)
        print(f"[bench] 500 episodes/worker -> {per_ep*500/3600:.1f}h per worker", flush=True)


if __name__ == "__main__":
    main()
