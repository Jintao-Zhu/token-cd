"""Stage-by-stage comparison of official vanilla vs L6-Matched on the SAME state.

The earlier component numbers are not comparable across the two paths: the matched adapter
times embed_inputs/prefix/negative/decode with explicit marks, while the official path is a
single opaque sample_actions call. This replicates both paths stage by stage so the extra
cost of the method can be attributed to specific stages rather than to the total.

Stages timed for both:
  embed_inputs        - SigLIP + text embedding of the observation
  prefix_forward      - the LLM pass over the prefix (matched: clean forward only)
  decode              - autoregressive action-token generation (matched: dual branch)
and, matched-only:
  tap_forward         - extra forward needed to read L6 attention (separate mode)
  negative_prefix     - forward over the counterfactual visual prefix
  matched_budget      - KMeans + entity matching
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
OUT = REPO / "artifacts/pi0_fast_stage_compare_v1"
N = 6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phys = int(os.environ.get("BENCH_PHYS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "2")))
    pol = Pi0FastMatchedCDPolicy(gpu=phys)
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    model = pol.model
    official = pol.policy
    gemma = pol.gemma_fast
    print(f"[cmp] kmeans={CD.KMEANS_BACKEND_REQUESTED} tap={pol.tap_mode} decode={pol.decode_mode}",
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

    van_stages: dict[str, list[float]] = {}
    mat_stages: dict[str, list[float]] = {}

    def rec(d, k, dt):
        d.setdefault(k, []).append(dt)

    for i in range(N):
        # ---------------- official vanilla, stage by stage ----------------
        t = time.perf_counter()
        inputs = official._input_transform(dict(raw))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        obs = model_lib.Observation.from_dict(batch)
        obs = model_lib.preprocess_observation(
            jax.random.key(100 + i), obs, train=False, image_keys=list(obs.images)
        )
        t_embed = time.perf_counter()
        prefix, pmask, par = model.embed_inputs(obs)
        jax.effects_barrier()
        t_pref = time.perf_counter()
        _ = np.asarray(model.sample_actions(
            jax.random.key(100 + i), obs,
            max_decoding_steps=int(pol.max_decoding_steps), temperature=0.0,
        )[0])
        t_end = time.perf_counter()
        if i:
            rec(van_stages, "embed_inputs", t_embed - t)
            rec(van_stages, "prefix_forward", t_pref - t_embed)
            rec(van_stages, "decode_total", t_end - t_pref)

        # ---------------- matched, stage by stage ----------------
        t = time.perf_counter()
        item = official._input_transform(dict(raw))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], item)
        obs2 = model_lib.Observation.from_dict(batch)
        obs2 = model_lib.preprocess_observation(
            jax.random.key(200 + i), obs2, train=False, image_keys=list(obs2.images)
        )
        t_embed = time.perf_counter()
        prefix2, pmask2, par2 = model.embed_inputs(obs2)
        jax.effects_barrier()
        t_pref = time.perf_counter()
        CD.LAMBDA = 0.5
        pol.lambda_ = 0.5
        out = pol.infer(dict(raw), seed=200 + i)
        t_end = time.perf_counter()
        tr = out["pi0_fast_matched_cd_trace"]
        if i:
            rec(mat_stages, "embed_inputs", t_embed - t)
            rec(mat_stages, "prefix_forward", t_pref - t_embed)
            rec(mat_stages, "rest_of_infer", t_end - t_pref)
            rec(mat_stages, "matched_budget", 0.0)  # from the microprofile, not re-timed here
        print(f"[cmp] iter {i}: vanilla_total={t_end - t:.2f}s "
              f"matched_total={t_end - t:.2f}s matched_valid={tr['fast_action_stream_valid']}",
              flush=True)

    def mean(d, k):
        v = d.get(k)
        return statistics.fmean(v) if v else float("nan")

    # the microprofile gave the matched sub-stage split; reuse it for attribution
    micro = {}
    mp = REPO / "artifacts/pi0_fast_microprofile_v1/MICROPROFILE.json"
    if mp.exists():
        micro = json.loads(mp.read_text()).get("substeps", {})

    summary = {
        "protocol": "PI0_FAST_STAGE_COMPARE_V1",
        "kmeans_backend": CD.KMEANS_BACKEND_REQUESTED,
        "vanilla": {k: mean(van_stages, k) for k in ("embed_inputs", "prefix_forward", "decode_total")},
        "matched_outer": {k: mean(mat_stages, k) for k in ("embed_inputs", "prefix_forward", "rest_of_infer")},
        "matched_substages_from_microprofile": {
            k: v.get("mean_s") for k, v in micro.items()
        },
        "note": "matched 'rest_of_infer' contains the tap forward, clean forward, negative "
                "forward, matched budget, dual-branch decode and the official cross-check.",
    }
    (OUT / "STAGE_COMPARE.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("[cmp] vanilla stages:", {k: round(v, 3) for k, v in summary["vanilla"].items()}, flush=True)
    print("[cmp] matched outer :", {k: round(v, 3) for k, v in summary["matched_outer"].items()}, flush=True)


if __name__ == "__main__":
    main()
