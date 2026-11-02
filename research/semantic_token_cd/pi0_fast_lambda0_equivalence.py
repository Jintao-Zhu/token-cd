"""P0 gate: does the custom pi0-FAST decoder reproduce official vanilla at lambda=0?

The custom L6-Matched adapter primes a fixed `Action: ` header, restricts the vocabulary to
FAST action-code tokens with a character-budget mask, and stops on the `|` delimiter, whereas
official OpenPI lets the model generate the header freely, applies no vocabulary restriction,
and stops on PALIGEMMA_EOS. Those are structural differences, so before any L6-Matched result
on pi0-FAST can be interpreted we must show that with lambda=0 (guidance is then a no-op:
guided = 1.0*positive - 0.0*negative) and therefore zero visual intervention, the custom path
reproduces the official one.

Reported per state, following the agreed comparison table:
  prefix tokens | first action logit | generated FAST token sequence (first divergence)
  | real BPE char count | official decode success | final action chunk

If this fails, every downstream conclusion about L6 / Matched / harmonic on pi0-FAST is
unfounded, so the script exits non-zero.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PI0FAST_CD_LAMBDA", "0.0")  # must be set before importing the module
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

from research.semantic_token_cd.pi0_fast_matched_cd import (  # noqa: E402
    CHECKPOINT,
    Pi0FastMatchedCDPolicy,
)

OUT = REPO / "artifacts/pi0_fast_lambda0_equivalence_v1"
# Results are tagged by the configuration that produced them, so a run under a different
# backend/mode can never overwrite the evidence for the frozen configuration.
OUT = OUT / os.environ.get("PI0FAST_GATE_TAG", "untagged")
STATES = ROOT / "token-cd/artifacts/spatialvla_layer_selection_v1b/states"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SEEDS = (100, 110, 120)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[gate] PI0FAST_CD_LAMBDA={os.environ['PI0FAST_CD_LAMBDA']}", flush=True)

    pol = Pi0FastMatchedCDPolicy(gpu=int(os.environ.get("CUDA_VISIBLE_DEVICES", "1")))
    import jax
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as model_lib

    print(f"[gate] custom lambda_={pol.lambda_}", flush=True)
    if pol.lambda_ != 0.0:
        raise SystemExit("lambda is not 0; this run would not be a no-guidance control")

    official = pol.policy  # the same trained policy object the adapter wraps
    model = pol.model

    # Reproduce the deployed SIMPLER adapter's view layout: overhead image + zero wrist view.
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "observation/image": "observation/primary_image",
                    "observation/wrist_image": "observation/wrist_image",
                    "observation/state": "observation/state",
                    "prompt": "prompt",
                }
            )
        ]
    )

    rows = []
    for task in TASKS:
        for seed in SEEDS:
            d = STATES / task / f"seed_{seed:03d}"
            if not d.exists():
                print(f"[gate] skip missing state {task} seed {seed}", flush=True)
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

            # ---- official path: capture the generated tokens, then the official actions ----
            inputs = official._input_transform(dict(raw))
            batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
            obs = model_lib.Observation.from_dict(batch)
            rng = jax.random.key(int(seed))
            off_tokens = model.sample_actions(
                rng, obs, max_decoding_steps=int(pol.max_decoding_steps), temperature=0.0
            )
            off_tokens = np.asarray(off_tokens[0], dtype=np.int32)
            off_out = official._output_transform(
                {"state": inputs["state"], "actions": off_tokens.copy()}
            )
            off_actions = np.asarray(off_out["actions"], dtype=np.float32)

            # ---- custom path at lambda = 0 ----
            cus_out = pol.infer(dict(raw), seed=int(seed))
            tr = cus_out["pi0_fast_matched_cd_trace"]
            cus_tokens = np.asarray(tr["raw_output_token_ids_full"], dtype=np.int32)
            cus_actions = np.asarray(cus_out["actions"], dtype=np.float32)

            # first divergence over the meaningful (non-padding) prefix
            n = min(len(off_tokens), len(cus_tokens))
            diff = np.flatnonzero(off_tokens[:n] != cus_tokens[:n])
            first_div = int(diff[0]) if diff.size else -1

            same_tokens = bool(np.array_equal(off_tokens[:n], cus_tokens[:n]))
            max_abs = float(np.max(np.abs(off_actions - cus_actions))) if off_actions.shape == cus_actions.shape else float("nan")
            same_actions = bool(off_actions.shape == cus_actions.shape and np.allclose(off_actions, cus_actions, rtol=1e-5, atol=1e-6))

            # Compare the DECODED PAYLOAD STRING, not raw token ids. FAST BPE is not
            # canonical: the same character string can be tokenised several ways, so token
            # ids legitimately differ while the decoded action is identical. The meaningful
            # equivalence is the payload string plus the action chunk.
            def payload_string(tokens_arr):
                txt = pol.tokenizer._paligemma_tokenizer.decode([int(x) for x in tokens_arr])
                if "Action: " not in txt:
                    return ""
                return txt.split("Action: ")[1].split("|")[0]

            off_str, cus_str = payload_string(off_tokens), payload_string(cus_tokens)
            same_str = off_str == cus_str

            row = {
                "task": task,
                "seed": int(seed),
                "official_token_count": int(np.count_nonzero(off_tokens) or len(off_tokens)),
                "custom_token_count": int(tr["decoded_token_steps"]),
                "same_token_sequence": same_tokens,
                "same_payload_string": same_str,
                "official_payload_len": len(off_str),
                "custom_payload_len": len(cus_str),
                "first_divergence_index": first_div,
                "official_tokens_prefix": off_tokens[:24].tolist(),
                "custom_tokens_prefix": cus_tokens[:24].tolist(),
                "custom_fast_code_token_count": int(tr["fast_code_token_count"]),
                "custom_payload_chars": int(tr["decoded_fast_payload_chars"]),
                "custom_payload_expected": int(tr["fast_payload_char_count"]),
                "custom_stream_valid": bool(tr["fast_action_stream_valid"]),
                "custom_official_decoder_agrees": bool(tr["openpi_official_decoder_agrees"]),
                "same_actions": same_actions,
                "action_max_abs_diff": max_abs,
                "official_action_finite": bool(np.isfinite(off_actions).all()),
                "official_action_absmax": float(np.max(np.abs(off_actions))),
                "custom_action_absmax": float(np.max(np.abs(cus_actions))),
                "matched_m_t": int(tr["matched_m_t"]),
                "first_clean_top_token": int(tr["first_clean_top_token"]),
                "first_guided_top_token": int(tr["first_guided_top_token"]),
            }
            rows.append(row)
            print(
                f"[gate] {task.replace('google_robot_','')} seed{seed}: "
                f"tokens_same={same_tokens} first_div={first_div} "
                f"actions_same={same_actions} maxdiff={max_abs:.3e} "
                f"stream_valid={row['custom_stream_valid']} "
                f"decoder_agrees={row['custom_official_decoder_agrees']}",
                flush=True,
            )

    n = len(rows)
    n_str = sum(r["same_payload_string"] for r in rows)
    n_act = sum(r["same_actions"] for r in rows)
    n_valid = sum(r["custom_stream_valid"] for r in rows)
    n_agree = sum(r["custom_official_decoder_agrees"] for r in rows)
    summary = {
        "protocol": "PI0_FAST_LAMBDA0_EQUIVALENCE_V1",
        "lambda": float(os.environ["PI0FAST_CD_LAMBDA"]),
        "checkpoint": str(CHECKPOINT),
        "states": n,
        "same_payload_string": n_str,
        "same_actions": n_act,
        "same_token_sequence_raw": sum(r["same_token_sequence"] for r in rows),
        "custom_stream_valid": n_valid,
        "custom_official_decoder_agrees": n_agree,
        "criterion": "payload string identical AND action chunk identical (raw token ids may "
                     "differ because FAST BPE re-tokenises the same string)",
        "verdict": (
            "PASS: custom lambda=0 reproduces official vanilla"
            if n and n_str == n and n_act == n
            else "FAIL: custom lambda=0 does NOT reproduce official vanilla"
        ),
        "rows": rows,
    }
    (OUT / "EQUIVALENCE.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[gate] {summary['verdict']}  states={n} payload_string_same={n_str}/{n} "
        f"actions_same={n_act}/{n} stream_valid={n_valid}/{n} decoder_agrees={n_agree}/{n}",
        flush=True,
    )
    if not (n and n_str == n and n_act == n):
        sys.exit(2)


if __name__ == "__main__":
    main()
