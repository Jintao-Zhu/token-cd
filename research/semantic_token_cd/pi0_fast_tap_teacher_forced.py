"""Metric (3) of the tap isolation test: does the cache difference reach the logits?

The isolation test showed B (tap separate) vs A (official) now agrees 12/12 while B vs C
(tap inline) differs in exactly the 5 states that used to diverge - i.e. the tap is the cause.
This script closes the mechanism: it builds the two caches, then teacher-forces the SAME
official token prefix through both and compares the resulting logits step by step.

If the logits differ under an identical token prefix, the cache difference is proven to
propagate into decoding rather than being an inert numerical artifact.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PI0FAST_CD_LAMBDA", "0.0")
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

STATES = REPO / "artifacts/spatialvla_layer_selection_v1b/states"
OUT = REPO / "artifacts/pi0_fast_tap_isolation_v1"
# The five states where the tap actually changed the action, plus one control.
CASES = [
    ("google_robot_pick_coke_can", 100),
    ("google_robot_move_near", 110),
    ("google_robot_open_drawer", 110),
    ("google_robot_pick_coke_can", 110),
    ("google_robot_move_near", 120),
    ("google_robot_open_drawer", 100),  # control: tap had no behavioural effect
]
N_TEACHER_STEPS = 12


def main() -> None:
    pol = Pi0FastMatchedCDPolicy(gpu=int(os.environ.get("CUDA_VISIBLE_DEVICES", "1")))
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    model, gemma_fast = pol.model, pol.gemma_fast
    official = pol.policy
    rows = []
    for task, seed in CASES:
        d = STATES / task / f"seed_{seed:03d}"
        meta = json.loads((d / "meta.json").read_text())
        npz = np.load(d / "input.npz")
        image = np.asarray(npz["image"])
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        prompt = meta["instruction"]
        raw = {
            "observation/primary_image": image,
            "observation/wrist_image": np.zeros_like(image),
            "observation/state": np.zeros((8,), dtype=np.float32),
            "prompt": prompt,
        }
        inputs = official._input_transform(dict(raw))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        obs = model_lib.Observation.from_dict(batch)
        obs = model_lib.preprocess_observation(
            jax.random.key(int(seed)), obs, train=False, image_keys=list(obs.images)
        )
        # the official greedy stream supplies the shared teacher-forced prefix
        off = np.asarray(
            model.sample_actions(
                jax.random.key(int(seed)), obs,
                max_decoding_steps=int(pol.max_decoding_steps), temperature=0.0,
            )[0],
            dtype=np.int32,
        )

        prefix, pmask, par = model.embed_inputs(obs)
        amask = pol.pi0_fast_lib.make_attn_mask(pmask, par)
        prefix, pmask, amask = pol.pi0_fast_lib.left_to_right_align(prefix, pmask, amask)
        psize = prefix.shape[1]
        plen = jnp.sum(pmask, axis=-1)
        pstart = psize - plen
        qm = pol._task_query_mask(inputs, prompt, psize, len(obs.images) * 256)
        qm = np.roll(qm, -int(np.asarray(plen)[0]))[None, :]
        padded = jnp.pad(amask, ((0, 0), (0, 0), (0, pol.max_decoding_steps)))
        pos = jnp.cumsum(pmask, axis=-1) - 1

        def prefix_forward(tapped: bool):
            if tapped:
                gemma_fast._PI0FAST_ATTN_CAPTURE = {
                    "query_mask": qm, "key_start": int(np.asarray(pstart)[0]),
                    "callback": lambda x: None,
                }
            else:
                gemma_fast._PI0FAST_ATTN_CAPTURE = None
            try:
                lg, cache, _ = model.PaliGemma.llm(
                    embedded_prefix=prefix, mask=padded, positions=pos, decode=True
                )
                jax.effects_barrier()
                return lg, cache
            finally:
                gemma_fast._PI0FAST_ATTN_CAPTURE = None

        _, cache_C = prefix_forward(True)
        _, cache_B = prefix_forward(False)

        # teacher-force the SAME official tokens through both caches
        logits_B, logits_C = [], []
        cb, cc = cache_B, cache_C
        prefill_size = int(psize)
        columns = jnp.arange(prefill_size + pol.max_decoding_steps)[None, None, :]
        for step in range(N_TEACHER_STEPS):
            tok = jnp.asarray([[int(off[step])]], dtype=jnp.int32)
            emb = model.PaliGemma.llm(tok, embed_only=True)
            positions = plen[:, None] + step + 1
            mask = jnp.logical_and(
                columns >= pstart[:, None, None],
                columns < jnp.broadcast_to(prefill_size + step + 1, (pstart.shape[0], 1, 1)),
            )
            lb, cb, _ = model.PaliGemma.llm(
                embedded_prefix=emb, mask=mask, positions=positions, decode=True, kv_cache=cb
            )
            lc, cc, _ = model.PaliGemma.llm(
                embedded_prefix=emb, mask=mask, positions=positions, decode=True, kv_cache=cc
            )
            jax.effects_barrier()
            logits_B.append(np.asarray(lb[0, -1], dtype=np.float32))
            logits_C.append(np.asarray(lc[0, -1], dtype=np.float32))

        per_step = []
        flips = 0
        for i, (a, b) in enumerate(zip(logits_B, logits_C)):
            am, bm = int(np.argmax(a)), int(np.argmax(b))
            per_step.append({
                "step": i,
                "teacher_token": int(off[i]),
                "max_abs_logit_diff": float(np.abs(a - b).max()),
                "argmax_B": am, "argmax_C": bm, "argmax_flipped": bool(am != bm),
            })
            flips += int(am != bm)
        row = {
            "task": task, "seed": int(seed),
            "teacher_steps": N_TEACHER_STEPS,
            "max_logit_diff_overall": max(x["max_abs_logit_diff"] for x in per_step),
            "argmax_flips": flips,
            "first_flip_step": next((x["step"] for x in per_step if x["argmax_flipped"]), -1),
            "per_step": per_step,
        }
        rows.append(row)
        print(
            f"[tf] {task.replace('google_robot_','')} seed{seed}: "
            f"max_logit_diff={row['max_logit_diff_overall']:.4f} "
            f"argmax_flips={flips}/{N_TEACHER_STEPS} first_flip={row['first_flip_step']}",
            flush=True,
        )

    summary = {
        "protocol": "PI0_FAST_TAP_TEACHER_FORCED_LOGITS_V1",
        "cases": len(rows),
        "cases_with_logit_change": sum(1 for r in rows if r["max_logit_diff_overall"] > 0),
        "cases_with_argmax_flip": sum(1 for r in rows if r["argmax_flips"] > 0),
        "rows": rows,
    }
    (OUT / "TEACHER_FORCED_LOGITS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[tf] logits changed in {summary['cases_with_logit_change']}/{len(rows)} cases, "
        f"argmax flipped in {summary['cases_with_argmax_flip']}/{len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
