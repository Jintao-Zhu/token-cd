"""Is the matched path's embed_inputs slow because it runs eagerly?

Vanilla costs 1.58 s in total, yet the matched path spends 2.393 s on embed_inputs alone.
A 3-view SigLIP encode is a few milliseconds of arithmetic, so that number is almost entirely
dispatch/launch overhead: the official sample_actions runs the whole pipeline inside one
compiled program, while the matched adapter calls embed_inputs as a standalone eager call.

This measures eager vs module_jit embed_inputs and checks the outputs are numerically
identical - if they are, compiling it is a move TOWARD the official execution, not away from
it, so the equivalence risk is different in kind from the batch=2 experiment.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
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

from research.semantic_token_cd.pi0_fast_matched_cd import Pi0FastMatchedCDPolicy  # noqa: E402

STATE = REPO / "artifacts/spatialvla_layer_selection_v1b/states/google_robot_open_drawer/seed_100"
OUT = REPO / "artifacts/pi0_fast_embed_jit_v1"
N = 6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phys = int(os.environ.get("BENCH_PHYS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "3")))
    pol = Pi0FastMatchedCDPolicy(gpu=phys)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as model_lib

    model = pol.model
    from openpi.shared import nnx_utils
    print(f"[jit] using nnx_utils from {nnx_utils.__name__}", flush=True)

    meta = json.loads((STATE / "meta.json").read_text())
    npz = np.load(STATE / "input.npz")
    image = np.asarray(npz["image"])
    raw = {
        "observation/primary_image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": meta["instruction"],
    }
    item = pol.policy._input_transform(dict(raw))
    batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], item)
    obs = model_lib.Observation.from_dict(batch)
    obs = model_lib.preprocess_observation(
        jax.random.key(100), obs, train=False, image_keys=list(obs.images)
    )

    eager_t, jit_t = [], []
    ref = None
    compiled = nnx_utils.module_jit(model.embed_inputs)
    for i in range(N):
        t0 = time.perf_counter()
        p1 = model.embed_inputs(obs)
        jax.effects_barrier()
        eager_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        p2 = compiled(obs)
        jax.effects_barrier()
        jit_t.append(time.perf_counter() - t0)

        if ref is None:
            ref = p1
        d = float(np.max(np.abs(np.asarray(p1[0]) - np.asarray(p2[0]))))
        print(f"[jit] iter {i}: eager={eager_t[-1]:.3f}s jit={jit_t[-1]:.3f}s "
              f"prefix_maxdiff={d:.3e}", flush=True)

    d_last = float(np.max(np.abs(np.asarray(p1[0]) - np.asarray(p2[0]))))
    d_ref = float(np.max(np.abs(np.asarray(ref[0]) - np.asarray(p1[0]))))
    summary = {
        "protocol": "PI0_FAST_EMBED_JIT_V1",
        "eager_mean_s": statistics.fmean(eager_t[1:]),
        "jit_mean_s": statistics.fmean(jit_t[1:]),
        "eager_first_s": eager_t[0], "jit_first_s": jit_t[0],
        "speedup": statistics.fmean(eager_t[1:]) / statistics.fmean(jit_t[1:]),
        "prefix_maxdiff_eager_vs_jit": d_last,
        "prefix_maxdiff_run_to_run_eager": d_ref,
        "identical": d_last == 0.0,
    }
    (OUT / "EMBED_JIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[jit] eager={summary['eager_mean_s']:.3f}s  jit={summary['jit_mean_s']:.3f}s  "
          f"speedup={summary['speedup']:.1f}x", flush=True)
    print(f"[jit] prefix identical: {summary['identical']} (maxdiff {d_last:.3e}; "
          f"run-to-run eager variation {d_ref:.3e})", flush=True)


if __name__ == "__main__":
    main()
