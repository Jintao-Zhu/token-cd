"""Introspect the pi0-FAST KV cache pytree so branch batching can be done correctly.

The batch=2 attempt failed because dynamic_update_slice saw a batch-1 k_cache while the
incoming k had batch 2, i.e. the cache concatenation did not take effect the way
jax.tree.map suggested. This prints the real structure and leaf shapes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
sys.path.insert(0, str(ROOT / "openpi/src"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from research.semantic_token_cd.pi0_fast_matched_cd import Pi0FastMatchedCDPolicy  # noqa: E402


def main() -> None:
    pol = Pi0FastMatchedCDPolicy(gpu=int(os.environ.get("CUDA_VISIBLE_DEVICES", "3")))
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    model = pol.model
    state = REPO / "artifacts/spatialvla_layer_selection_v1b/states/google_robot_open_drawer/seed_100"
    import json

    meta = json.loads((state / "meta.json").read_text())
    npz = np.load(state / "input.npz")
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
    prefix, pmask, par = model.embed_inputs(obs)
    amask = pol.pi0_fast_lib.make_attn_mask(pmask, par)
    prefix, pmask, amask = pol.pi0_fast_lib.left_to_right_align(prefix, pmask, amask)
    padded = jnp.pad(amask, ((0, 0), (0, 0), (0, pol.max_decoding_steps)))
    pos = jnp.cumsum(pmask, axis=-1) - 1

    logits, cache, _ = model.PaliGemma.llm(
        embedded_prefix=prefix, mask=padded, positions=pos, decode=True
    )
    jax.effects_barrier()

    print("[cache] type:", type(cache), flush=True)
    print("[cache] pytree def:", jax.tree.structure(cache), flush=True)
    leaves = jax.tree.leaves(cache)
    print(f"[cache] n_leaves={len(leaves)}", flush=True)
    for i, lf in enumerate(leaves[:8]):
        print(f"   leaf[{i}] shape={getattr(lf, 'shape', None)} dtype={getattr(lf, 'dtype', None)}",
              flush=True)
    if len(leaves) > 8:
        print(f"   ... ({len(leaves)} leaves total; pattern repeats per layer)", flush=True)

    # try the concatenation the batched path uses and re-inspect
    try:
        cat = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), cache, cache)
        cl = jax.tree.leaves(cat)
        print("[cache] after tree.map concat, first leaves:", flush=True)
        for i, lf in enumerate(cl[:6]):
            print(f"   leaf[{i}] shape={getattr(lf, 'shape', None)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[cache] tree.map concat failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
