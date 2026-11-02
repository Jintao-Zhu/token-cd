"""Does the L6 attention tap change the numerics of the positive prefix forward?

The custom adapter installs a diagnostic tap inside the attention softmax of the instrumented
OpenPI overlay:

    selected = jnp.einsum("bkgts,bt->bks", probs[..., key_start:key_start+256], query_mask)
    jax.debug.callback(_PI0FAST_ATTN_CAPTURE["callback"], selected, ordered=True)

That inserts extra ops into the attention graph, which can change XLA fusion and therefore
floating-point rounding. The lambda=0 equivalence gate found the custom decoder picking a
different FIRST payload token than official vanilla on 2/12 states, and the official side of
that comparison ran WITHOUT the tap. So the tap is the prime suspect.

This script runs the identical prefix forward twice - once with the tap active, once without -
and reports whether the logits move, whether the argmax flips, and by how much.
"""
from __future__ import annotations

import json
import os
import sys
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

STATES = REPO / "artifacts/spatialvla_layer_selection_v1b/states"
OUT = REPO / "artifacts/pi0_fast_tap_numerics_v1"
# Mode-A states (divergence at the first payload token) plus controls that agreed.
CASES = [
    ("google_robot_pick_coke_can", 100),   # mode A: official[3]=256715 vs custom 256709
    ("google_robot_move_near", 110),       # mode A
    ("google_robot_open_drawer", 100),     # control: actions agreed
    ("google_robot_close_drawer", 100),    # control: actions agreed
    ("google_robot_open_drawer", 110),     # mode B: agreed on first token, drifted later
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pol = Pi0FastMatchedCDPolicy(gpu=int(os.environ.get("CUDA_VISIBLE_DEVICES", "1")))
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    model = pol.model
    gemma_fast = pol.gemma_fast

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

        item = pol.policy._input_transform(dict(raw))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], item)
        obs = model_lib.Observation.from_dict(batch)
        obs = model_lib.preprocess_observation(
            jax.random.key(int(seed)), obs, train=False, image_keys=list(obs.images)
        )
        prefix, prefix_mask, prefix_ar = model.embed_inputs(obs)
        attn_mask = pol.pi0_fast_lib.make_attn_mask(prefix_mask, prefix_ar)
        prefix, prefix_mask, attn_mask = pol.pi0_fast_lib.left_to_right_align(
            prefix, prefix_mask, attn_mask
        )
        prefix_size = prefix.shape[1]
        prefix_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefix_size - prefix_len
        image_prefix_len = len(obs.images) * 256
        qmask = pol._task_query_mask(item, prompt, prefix_size, image_prefix_len)
        qmask = np.roll(qmask, -int(np.asarray(prefix_len)[0]))[None, :]
        padded = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, pol.max_decoding_steps)))
        positions = jnp.cumsum(prefix_mask, axis=-1) - 1

        def forward(with_tap: bool):
            if with_tap:
                gemma_fast._PI0FAST_ATTN_CAPTURE = {
                    "query_mask": qmask,
                    "key_start": int(np.asarray(prefix_start)[0]),
                    "callback": lambda x: None,
                }
            else:
                gemma_fast._PI0FAST_ATTN_CAPTURE = None
            try:
                lg, _, _ = model.PaliGemma.llm(
                    embedded_prefix=prefix, mask=padded, positions=positions, decode=True
                )
                jax.effects_barrier()
                return np.asarray(lg[0, -1], dtype=np.float32)
            finally:
                gemma_fast._PI0FAST_ATTN_CAPTURE = None

        a = forward(True)
        b = forward(False)
        diff = np.abs(a - b)
        top_a, top_b = int(np.argmax(a)), int(np.argmax(b))
        # margin between the best and second best of the tap-free run, to judge near-ties
        order = np.argsort(-b)
        margin = float(b[order[0]] - b[order[1]])
        row = {
            "task": task,
            "seed": int(seed),
            "max_abs_logit_diff": float(diff.max()),
            "n_changed_bits": int(np.count_nonzero(a != b)),
            "argmax_with_tap": top_a,
            "argmax_without_tap": top_b,
            "argmax_flipped": bool(top_a != top_b),
            "top1_minus_top2_without_tap": margin,
            "tap_shift_at_argmax": float(a[top_b] - b[top_b]),
        }
        rows.append(row)
        print(
            f"[tap] {task.replace('google_robot_','')} seed{seed}: "
            f"maxdiff={row['max_abs_logit_diff']:.6f} changed={row['n_changed_bits']} "
            f"argmax {top_a} -> {top_b} flipped={row['argmax_flipped']} "
            f"margin={margin:.6f} shift_at_argmax={row['tap_shift_at_argmax']:.6f}",
            flush=True,
        )

    n_flip = sum(r["argmax_flipped"] for r in rows)
    n_changed = sum(r["n_changed_bits"] > 0 for r in rows)
    summary = {
        "protocol": "PI0_FAST_TAP_NUMERICS_V1",
        "cases": len(rows),
        "cases_with_any_logit_change": n_changed,
        "cases_with_argmax_flip": n_flip,
        "verdict": (
            "TAP PERTURBS NUMERICS"
            if n_changed or n_flip
            else "tap is numerically inert"
        ),
        "rows": rows,
    }
    (OUT / "TAP_NUMERICS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[tap] {summary['verdict']}: logits changed in {n_changed}/{len(rows)}, "
        f"argmax flipped in {n_flip}/{len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
