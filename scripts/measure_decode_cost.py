"""Per-token decode cost on both sides of the comparison.

OpenVLA also generates its action tokens autoregressively (7 tokens via HF generate), so the
earlier claim that it "does 7 tokens in one forward" was wrong. What actually differs is the
TOKEN COUNT and the PER-TOKEN cost. This measures both on pi0-FAST: the vanilla
sample_actions call and a single prefix forward, so the vanilla decode can be isolated and
compared against the matched path's measured 4.276 s dual-branch decode.
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
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
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
OUT = REPO / "artifacts/pi0_fast_decode_cost_v1"
N = 6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pol = Pi0FastMatchedCDPolicy(gpu=int(os.environ.get("CUDA_VISIBLE_DEVICES", "3")))
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    model, official = pol.model, pol.policy
    meta = json.loads((STATE / "meta.json").read_text())
    npz = np.load(STATE / "input.npz")
    image = np.asarray(npz["image"])
    raw = {
        "observation/primary_image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": meta["instruction"],
    }

    van, prefix, mat = [], [], []
    for i in range(N):
        # 1) vanilla end to end
        t0 = time.perf_counter()
        toks = np.asarray(model.sample_actions(
            jax.random.key(100 + i), model_lib.preprocess_observation(
                jax.random.key(100 + i),
                model_lib.Observation.from_dict(jax.tree.map(
                    lambda x: jnp.asarray(x)[None, ...], official._input_transform(dict(raw)))),
                train=False,
                image_keys=list(official._input_transform(dict(raw))["image"].keys())),
            max_decoding_steps=int(pol.max_decoding_steps), temperature=0.0)[0], dtype=np.int32)
        jax.effects_barrier()
        van.append(time.perf_counter() - t0)
        n_tok = int(np.count_nonzero(toks))

        # 2) one prefix forward only (embed + llm prefill), isolated with a barrier
        item = official._input_transform(dict(raw))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], item)
        obs = model_lib.Observation.from_dict(batch)
        obs = model_lib.preprocess_observation(
            jax.random.key(200 + i), obs, train=False, image_keys=list(obs.images))
        t0 = time.perf_counter()
        prefix_emb, pmask, par = model.embed_inputs(obs)
        amask = pol.pi0_fast_lib.make_attn_mask(pmask, par)
        prefix_emb, pmask, amask = pol.pi0_fast_lib.left_to_right_align(prefix_emb, pmask, amask)
        padded = jnp.pad(amask, ((0, 0), (0, 0), (0, pol.max_decoding_steps)))
        pos = jnp.cumsum(pmask, axis=-1) - 1
        _ = model.PaliGemma.llm(embedded_prefix=prefix_emb, mask=padded, positions=pos, decode=True)
        jax.effects_barrier()
        prefix.append(time.perf_counter() - t0)

        # 3) matched end to end
        CD.LAMBDA = 0.5
        pol.lambda_ = 0.5
        t0 = time.perf_counter()
        pol.infer(dict(raw), seed=200 + i)
        mat.append(time.perf_counter() - t0)

    vs, ps, ms = van[1:], prefix[1:], mat[1:]
    v = statistics.fmean(vs); p = statistics.fmean(ps); m = statistics.fmean(ms)
    dec_v = v - p
    summary = {
        "protocol": "PI0_FAST_DECODE_COST_V1",
        "vanilla_total_s": v, "prefix_forward_s": p,
        "vanilla_decode_implied_s": dec_v,
        "matched_total_s": m,
        "matched_decode_s_measured_earlier": 4.276,
        "vanilla_action_tokens": n_tok,
        "note": "OpenVLA also decodes autoregressively (7 tokens); the comparison is token COUNT "
                "and PER-TOKEN cost, not 'one forward vs many'.",
    }
    (OUT / "DECODE_COST.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[dec] vanilla total      = {v:.3f} s")
    print(f"[dec] embed+prefix       = {p:.3f} s")
    print(f"[dec] => vanilla decode  = {dec_v:.3f} s  for {n_tok} FAST tokens "
          f"({1000*dec_v/max(n_tok,1):.1f} ms/token, single branch)")
    print(f"[dec] matched total      = {m:.3f} s")
    print(f"[dec] matched decode     = 4.276 s (measured earlier, dual branch, ~30 tokens)")
    print(f"[dec] => matched per token per branch = {1000*4.276/2/30:.1f} ms")


if __name__ == "__main__":
    main()
