"""Tap Isolation Test: is the attention tap the cause of the lambda=0 divergence?

Three groups on the same fixed states, no closed-loop episodes:

  A  official vanilla      : tap none,     official greedy decode
  B  custom, tap separate  : tap off in the cache-producing forward (the option-A fix)
  C  custom, tap inline    : tap on in the cache-producing forward (historic behaviour)

B and C differ ONLY in where the tap sits; both run the same infer() path, the same negative
branch, the same decode loop, and lambda=0 so no visual intervention acts on the result.

Reported, per state:
  * B vs A  : FAST token sequence, BPE-decoded payload string (character-exact), action chunk
  * B vs C  : final actions, per-layer KV-cache numerical difference, and teacher-forced
              logits under the SAME official token prefix (does a cache difference propagate?)
  * validity of every decode
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
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SEEDS = (100, 110, 120)


def flatten_cache(cache):
    import jax
    leaves = jax.tree.leaves(cache)
    return [np.asarray(x, dtype=np.float32) for x in leaves if hasattr(x, "shape")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pol = Pi0FastMatchedCDPolicy(gpu=int(os.environ.get("CUDA_VISIBLE_DEVICES", "1")))
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    print(f"[iso] lambda_={pol.lambda_} default_tap_mode={pol.tap_mode}", flush=True)
    if pol.lambda_ != 0.0:
        raise SystemExit("lambda must be 0 for this isolation test")

    official = pol.policy
    model = pol.model
    gemma_fast = pol.gemma_fast
    pg = pol.tokenizer._paligemma_tokenizer

    def decoded_payload_string(tokens: np.ndarray) -> str:
        """The BPE payload text between the Action: header and the '|' delimiter."""
        txt = pg.decode([int(x) for x in tokens])
        if "Action: " not in txt:
            return ""
        return txt.split("Action: ")[1].split("|")[0]

    rows = []
    for task in TASKS:
        for seed in SEEDS:
            d = STATES / task / f"seed_{seed:03d}"
            if not d.exists():
                continue
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

            # ---------- A: official vanilla ----------
            inputs = official._input_transform(dict(raw))
            batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
            obs = model_lib.Observation.from_dict(batch)
            off_tokens = np.asarray(
                model.sample_actions(
                    jax.random.key(int(seed)), obs,
                    max_decoding_steps=int(pol.max_decoding_steps), temperature=0.0,
                )[0],
                dtype=np.int32,
            )
            off_actions = np.asarray(
                official._output_transform(
                    {"state": inputs["state"], "actions": off_tokens.copy()}
                )["actions"],
                dtype=np.float32,
            )

            # ---------- B and C: same infer() path, only tap placement differs ----------
            res = {}
            for tag, mode in (("B", "separate"), ("C", "inline")):
                pol.tap_mode = mode
                out = pol.infer(dict(raw), seed=int(seed))
                tr = out["pi0_fast_matched_cd_trace"]
                res[tag] = {
                    "tokens": np.asarray(tr["raw_output_token_ids_full"], dtype=np.int32),
                    "actions": np.asarray(out["actions"], dtype=np.float32),
                    "valid": bool(tr["fast_action_stream_valid"]),
                    "agrees": bool(tr["openpi_official_decoder_agrees"]),
                }
            pol.tap_mode = "inline"

            def cmp_tokens(x, y):
                n = min(len(x), len(y))
                dd = np.flatnonzero(x[:n] != y[:n])
                return (int(dd[0]) if dd.size else -1), bool(np.array_equal(x[:n], y[:n]))

            div_BA, same_BA = cmp_tokens(off_tokens, res["B"]["tokens"])
            div_BC, same_BC = cmp_tokens(res["B"]["tokens"], res["C"]["tokens"])
            str_A, str_B = decoded_payload_string(off_tokens), decoded_payload_string(res["B"]["tokens"])
            act_BA = float(np.max(np.abs(off_actions - res["B"]["actions"]))) if off_actions.shape == res["B"]["actions"].shape else float("nan")
            act_BC = float(np.max(np.abs(res["B"]["actions"] - res["C"]["actions"]))) if res["B"]["actions"].shape == res["C"]["actions"].shape else float("nan")

            # ---------- mechanism: per-layer KV cache, B vs C ----------
            # Rebuild the prefix and run the two tap placements directly so the caches can be
            # compared. This mirrors exactly what infer() does internally.
            obs2 = model_lib.preprocess_observation(
                jax.random.key(int(seed)), obs, train=False, image_keys=list(obs.images)
            )
            prefix, pmask, par = model.embed_inputs(obs2)
            amask = pol.pi0_fast_lib.make_attn_mask(pmask, par)
            prefix, pmask, amask = pol.pi0_fast_lib.left_to_right_align(prefix, pmask, amask)
            psize = prefix.shape[1]
            plen = jnp.sum(pmask, axis=-1)
            pstart = psize - plen
            qm = pol._task_query_mask(inputs, prompt, psize, len(obs2.images) * 256)
            qm = np.roll(qm, -int(np.asarray(plen)[0]))[None, :]
            padded = jnp.pad(amask, ((0, 0), (0, 0), (0, pol.max_decoding_steps)))
            pos = jnp.cumsum(pmask, axis=-1) - 1

            def run(tapped: bool):
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
                    return np.asarray(lg[0, -1], dtype=np.float32), cache
                finally:
                    gemma_fast._PI0FAST_ATTN_CAPTURE = None

            _, cache_C = run(True)
            _, cache_B = run(False)
            fc = flatten_cache(cache_C)
            fb = flatten_cache(cache_B)
            layer_diffs = []
            for i, (a, b) in enumerate(zip(fc, fb)):
                if a.shape != b.shape:
                    layer_diffs.append({"leaf": i, "shape_mismatch": True})
                    continue
                d_ = np.abs(a - b)
                layer_diffs.append({
                    "leaf": i,
                    "max_abs_diff": float(d_.max()),
                    "mean_abs_diff": float(d_.mean()),
                    "frac_changed": float(np.mean(a != b)),
                })
            n_leaves_changed = sum(1 for x in layer_diffs if x.get("max_abs_diff", 0) > 0)

            row = {
                "task": task, "seed": int(seed),
                "A_tokens": off_tokens[:40].tolist(),
                "B_tokens": res["B"]["tokens"][:40].tolist(),
                "C_tokens": res["C"]["tokens"][:40].tolist(),
                "A_payload_string": str_A[:80],
                "B_payload_string": str_B[:80],
                "payload_string_equal_BA": str_A == str_B,
                "payload_string_len_A": len(str_A),
                "payload_string_len_B": len(str_B),
                "tokens_same_BA": same_BA, "first_div_BA": div_BA,
                "tokens_same_BC": same_BC, "first_div_BC": div_BC,
                "actions_maxdiff_BA": act_BA,
                "actions_maxdiff_BC": act_BC,
                "actions_same_BA": bool(act_BA == act_BA and act_BA < 1e-5),
                "B_valid": res["B"]["valid"], "B_decoder_agrees": res["B"]["agrees"],
                "C_valid": res["C"]["valid"], "C_decoder_agrees": res["C"]["agrees"],
                "cache_leaves": len(fc),
                "cache_leaves_changed": n_leaves_changed,
                "cache_layer_diffs": layer_diffs,
            }
            rows.append(row)
            print(
                f"[iso] {task.replace('google_robot_','')} seed{seed}: "
                f"BvsA str_eq={row['payload_string_equal_BA']} tok_eq={same_BA} div={div_BA} "
                f"act={act_BA:.3e} | BvsC tok_eq={same_BC} div={div_BC} act={act_BC:.3e} "
                f"| cache_leaves_changed={n_leaves_changed}/{len(fc)}",
                flush=True,
            )

    n = len(rows)
    summary = {
        "protocol": "PI0_FAST_TAP_ISOLATION_V1",
        "lambda": 0.0,
        "states": n,
        "payload_string_equal_BA": sum(r["payload_string_equal_BA"] for r in rows),
        "actions_same_BA": sum(r["actions_same_BA"] for r in rows),
        "tokens_same_BA": sum(r["tokens_same_BA"] for r in rows),
        "actions_same_BC": sum(1 for r in rows if r["actions_maxdiff_BC"] < 1e-5),
        "states_with_cache_change": sum(1 for r in rows if r["cache_leaves_changed"] > 0),
        "all_decodes_valid": all(r["B_valid"] and r["C_valid"] for r in rows),
        "rows": rows,
    }
    (OUT / "ISOLATION.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[iso] states={n} | B vs A: payload_string_eq={summary['payload_string_equal_BA']}/{n} "
        f"actions_eq={summary['actions_same_BA']}/{n} | B vs C actions_eq={summary['actions_same_BC']}/{n} "
        f"| cache changed in {summary['states_with_cache_change']}/{n}",
        flush=True,
    )


if __name__ == "__main__":
    main()
