"""Where do the 2.69 s (matched budget) and 4.94 s (FAST decode) actually go?

Both numbers are suspicious for their problem size: K=8 clustering of 256 tokens should not
take seconds, and the FAST decode's arithmetic is small relative to the 7B forwards. The
hypothesis is that a large share is JAX dispatch/synchronisation and repeated calls rather
than the algorithm itself.

This instruments the two methods at sub-step granularity on a warm process, so the numbers
are steady-state (compilation excluded).
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

import research.semantic_token_cd.pi0_fast_matched_cd as CD  # noqa: E402
from research.semantic_token_cd.pi0_fast_matched_cd import Pi0FastMatchedCDPolicy  # noqa: E402

STATE = REPO / "artifacts/spatialvla_layer_selection_v1b/states/google_robot_open_drawer/seed_100"
OUT = REPO / "artifacts/pi0_fast_microprofile_v1"
N_REPEAT = 5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phys = int(os.environ.get("BENCH_PHYS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "1")))
    pol = Pi0FastMatchedCDPolicy(gpu=phys)
    print(f"[micro] tap={pol.tap_mode} decode={pol.decode_mode}", flush=True)

    import jax

    acc: dict[str, list[float]] = {}

    def rec(key, dt):
        acc.setdefault(key, []).append(dt)

    # ---- instrument _matched_budget internals ----
    orig_km = CD._kmeans_labels
    orig_ent = CD._extract_entities
    orig_emb = Pi0FastMatchedCDPolicy._entity_embedding

    def km(features):
        t0 = time.perf_counter()
        out = orig_km(features)
        rec("budget.kmeans_labels", time.perf_counter() - t0)
        return out

    def ent(instr):
        t0 = time.perf_counter()
        out = orig_ent(instr)
        rec("budget.extract_entities", time.perf_counter() - t0)
        return out

    def emb(self, phrase):
        t0 = time.perf_counter()
        out = orig_emb(self, phrase)
        rec("budget.entity_embedding", time.perf_counter() - t0)
        return out

    CD._kmeans_labels = km
    CD._extract_entities = ent
    Pi0FastMatchedCDPolicy._entity_embedding = emb

    # ---- instrument the decode ----
    orig_compiled = pol._compiled_decode
    orig_payload = pol._payload_of

    def compiled(*a, **k):
        t0 = time.perf_counter()
        out = orig_compiled(*a, **k)
        rec("decode.compiled_dispatch", time.perf_counter() - t0)
        return out

    def payload(tokens):
        t0 = time.perf_counter()
        out = orig_payload(tokens)
        rec("decode.payload_extract", time.perf_counter() - t0)
        return out

    pol._compiled_decode = compiled
    pol._payload_of = payload

    meta = json.loads((STATE / "meta.json").read_text())
    npz = np.load(STATE / "input.npz")
    image = np.asarray(npz["image"])
    raw = {
        "observation/primary_image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": meta["instruction"],
    }

    totals = []
    for i in range(N_REPEAT):
        t0 = time.perf_counter()
        out = pol.infer(dict(raw), seed=100 + i)
        totals.append(time.perf_counter() - t0)
    steady = totals[1:]

    print(f"[micro] infer() steady mean = {statistics.fmean(steady):.2f}s "
          f"(first call {totals[0]:.2f}s)", flush=True)
    print("[micro] sub-step breakdown (mean per infer call):", flush=True)
    rows = {}
    for k in sorted(acc):
        vals = acc[k][1:] if len(acc[k]) > 1 else acc[k]   # drop the cold call
        m = statistics.fmean(vals) if vals else 0.0
        rows[k] = {"mean_s": m, "n": len(vals)}
        print(f"[micro]   {k:32s} {m:7.3f}s  n={len(vals)}", flush=True)

    # what remains of the budget stage and the decode stage?
    b = rows.get("budget.kmeans_labels", {}).get("mean_s", 0) + \
        rows.get("budget.extract_entities", {}).get("mean_s", 0) + \
        rows.get("budget.entity_embedding", {}).get("mean_s", 0)
    d = rows.get("decode.compiled_dispatch", {}).get("mean_s", 0) + \
        rows.get("decode.payload_extract", {}).get("mean_s", 0)
    summary = {
        "protocol": "PI0_FAST_MICROPROFILE_V1",
        "infer_steady_mean_s": statistics.fmean(steady),
        "infer_first_call_s": totals[0],
        "substeps": rows,
        "budget_measured_s": b,
        "decode_measured_s": d,
        "note": "budget_total was 2.69s and decode_total 4.94s in the coarse breakdown; the "
                "difference is unmeasured work inside those stages (JAX sync, numpy, "
                "cache/attention plumbing).",
    }
    (OUT / "MICROPROFILE.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[micro] budget substeps sum = {b:.2f}s (coarse budget stage was ~2.69s)", flush=True)
    print(f"[micro] decode substeps sum = {d:.2f}s (coarse decode stage was ~4.94s)", flush=True)


if __name__ == "__main__":
    main()
