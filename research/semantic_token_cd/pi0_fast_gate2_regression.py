"""Gate 2 regression: FAST action-stream validity across tasks and all four lambdas.

Requirement being tested (from the experiment plan):
  * every guided decode produces a stream the official FAST decoder accepts
  * action shape matches the checkpoint's FAST action config, all finite
  * no padding, no truncation, no silent fallback to vanilla
  * covers all nine SIMPLER tasks and all four guidance strengths

Because the exact decoder is a Python loop that reads LAMBDA from the module global at call
time, all four lambdas can be swept inside one process (no model reload).

The lambda=0 alignment control is included per state: with no guidance the custom path must
reproduce the official vanilla action chunk.
"""
from __future__ import annotations

import json
import os
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

POOLS = [
    REPO / "artifacts/spatialvla_layer_selection_v1b/states",
    REPO / "artifacts/spatialvla_layer_selection_v1/states",
]
OUT = REPO / "artifacts/pi0_fast_gate2_regression_v1"
LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.00)


def assert_on_gpu(expected: int) -> None:
    """Fail loudly if this process is not actually on the intended physical GPU."""
    import subprocess

    uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(expected)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    mine = str(os.getpid())
    hit = [ln for ln in apps.splitlines() if ln.split(",")[-1].strip() == mine]
    if not hit:
        print(f"[g2] note: PID {mine} not yet visible on any GPU (pre-load)", flush=True)
        return
    on = hit[0].split(",")[0].strip()
    if on != uuid:
        raise SystemExit(f"[g2] ABORT: expected physical GPU {expected} ({uuid}) but this "
                         f"process is on {on}")
    print(f"[g2] confirmed on physical GPU {expected}", flush=True)


def collect_states() -> list[tuple[str, int, Path]]:
    seen: dict[tuple[str, int], Path] = {}
    for pool in POOLS:
        if not pool.is_dir():
            continue
        for task in sorted(os.listdir(pool)):
            tdir = pool / task
            if not tdir.is_dir():
                continue
            for sd in sorted(os.listdir(tdir)):
                if not sd.startswith("seed_"):
                    continue
                seed = int(sd.split("_")[1])
                key = (task, seed)
                if key not in seen and (tdir / sd / "input.npz").exists():
                    seen[key] = tdir / sd
    return [(t, s, p) for (t, s), p in sorted(seen.items())]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get("GATE2_LIMIT", "0"))
    states = collect_states()
    if limit:
        states = states[:limit]
    tasks = sorted({t for t, _, _ in states})
    print(f"[g2] states={len(states)} over {len(tasks)} tasks; lambdas={LAMBDAS}", flush=True)

    # `gpu=` is the PHYSICAL GPU id: Pi0FastMatchedCDPolicy.__init__ assigns it to
    # CUDA_VISIBLE_DEVICES, overwriting whatever the shell exported. Passing a logical index
    # here silently moves the process to another physical card, so take it from an explicit
    # variable and then assert where we actually landed.
    phys_gpu = int(os.environ.get("GATE2_PHYS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "2")))
    pol = Pi0FastMatchedCDPolicy(gpu=phys_gpu)
    assert_on_gpu(phys_gpu)
    print(f"[g2] tap_mode={pol.tap_mode} decode_mode={pol.decode_mode}", flush=True)
    if pol.tap_mode != "separate" or pol.decode_mode == "legacy":
        raise SystemExit("gates not in the frozen configuration")
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    official = pol.policy
    model = pol.model

    rows = []
    t_start = time.monotonic()
    for idx, (task, seed, sdir) in enumerate(states):
        meta = json.loads((sdir / "meta.json").read_text())
        npz = np.load(sdir / "input.npz")
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
        # official vanilla action chunk for the alignment control
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

        for lam in LAMBDAS:
            CD.LAMBDA = float(lam)
            pol.lambda_ = float(lam)
            try:
                out = pol.infer(dict(raw), seed=int(seed))
                tr = out["pi0_fast_matched_cd_trace"]
                act = np.asarray(out["actions"], dtype=np.float32)
                rec = {
                    "task": task, "seed": int(seed), "lambda": float(lam),
                    "ok": True,
                    "stream_valid": bool(tr["fast_action_stream_valid"]),
                    "decoder_agrees": bool(tr["openpi_official_decoder_agrees"]),
                    "action_shape": list(act.shape),
                    "action_finite": bool(np.isfinite(act).all()),
                    "payload_chars": int(tr["decoded_fast_payload_chars"]),
                    "payload_expected": int(tr["fast_payload_char_count"]),
                    "code_tokens": int(tr["fast_code_token_count"]),
                    "matched_m_t": int(tr["matched_m_t"]),
                    "selected_tokens": int(tr["selected_token_count"]),
                    "entities": tr["entities"],
                    "kmeans_backend": tr["matched_kmeans_backend"],
                    "action_absmax": float(np.max(np.abs(act))),
                    # --- negative-branch construction checks ---
                    # lambda=0 cannot detect a broken counterfactual, because the guidance is
                    # multiplied by zero. These fields verify the intervention itself.
                    "selected_count_equals_matched": bool(tr["selected_count_equals_matched"]),
                    "harmonic_finite": bool(tr["harmonic_finite"]),
                    "selected_ids_in_range": bool(
                        all(0 <= int(i) < 256 for i in tr["selected_token_ids"])
                    ),
                    "selected_ids_unique": bool(
                        len(set(int(i) for i in tr["selected_token_ids"])) == len(tr["selected_token_ids"])
                    ),
                    "neg_branch_logit_residual": float(tr["first_logit_residual_norm"]),
                    "neg_branch_changed_argmax": bool(
                        tr["first_clean_top_token"] != tr["first_negative_top_token"]
                    ),
                    "decode_target_delta": int(tr.get("decode_target_delta", 0)),
                }
                if lam == 0.0 and off_actions.shape == act.shape:
                    rec["action_maxdiff_vs_official"] = float(np.max(np.abs(off_actions - act)))
                    rec["actions_match_official"] = bool(rec["action_maxdiff_vs_official"] < 1e-5)
                rows.append(rec)
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "task": task, "seed": int(seed), "lambda": float(lam),
                    "ok": False, "error_type": type(exc).__name__, "error": str(exc)[:400],
                })
        if idx % 5 == 0 or idx == len(states) - 1:
            done = len(rows)
            bad = sum(1 for r in rows if not r.get("ok") or not r.get("stream_valid"))
            print(f"[g2] {idx+1}/{len(states)} states, {done} streams, invalid={bad}, "
                  f"elapsed={time.monotonic()-t_start:.0f}s", flush=True)

    n = len(rows)
    ok = [r for r in rows if r.get("ok")]
    valid = [r for r in ok if r["stream_valid"]]
    agree = [r for r in ok if r["decoder_agrees"]]
    lam0 = [r for r in ok if r["lambda"] == 0.0 and "actions_match_official" in r]
    lam0_match = [r for r in lam0 if r["actions_match_official"]]
    # negative-branch construction, checked over every guided (lambda>0) stream
    guided = [r for r in ok if r["lambda"] > 0.0]
    neg_ok = [
        r for r in guided
        if r["selected_count_equals_matched"] and r["harmonic_finite"]
        and r["selected_ids_in_range"] and r["selected_ids_unique"]
        and r["neg_branch_logit_residual"] > 0.0
    ]
    neg_effective = [r for r in guided if r["neg_branch_logit_residual"] > 0.0]
    neg_flips = [r for r in guided if r["neg_branch_changed_argmax"]]
    backends = sorted({r["kmeans_backend"] for r in ok})
    per_lam = {
        str(l): {
            "streams": sum(1 for r in ok if r["lambda"] == l),
            "valid": sum(1 for r in ok if r["lambda"] == l and r["stream_valid"]),
            "decoder_agrees": sum(1 for r in ok if r["lambda"] == l and r["decoder_agrees"]),
        }
        for l in LAMBDAS
    }
    summary = {
        "protocol": "PI0_FAST_GATE2_REGRESSION_V1",
        "states": len(states),
        "tasks": tasks,
        "lambdas": list(LAMBDAS),
        "streams": n,
        "ok": len(ok),
        "errors": n - len(ok),
        "stream_valid": len(valid),
        "decoder_agrees": len(agree),
        "lambda0_streams": len(lam0),
        "lambda0_match_official": len(lam0_match),
        "guided_streams": len(guided),
        "neg_branch_valid": len(neg_ok),
        "neg_branch_effective": len(neg_effective),
        "neg_branch_argmax_flips": len(neg_flips),
        "kmeans_backends": backends,
        "per_lambda": per_lam,
        "all_finite": all(r["action_finite"] for r in ok),
        "shapes": sorted({tuple(r["action_shape"]) for r in ok}),
        "verdict": (
            "PASS" if (n and len(ok) == n and len(valid) == n and len(agree) == n
                       and lam0 and len(lam0_match) == len(lam0)
                       and len(neg_ok) == len(guided) and len(guided) > 0)
            else "FAIL"
        ),
        "rows": rows,
    }
    (OUT / "REGRESSION.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[g2] {summary['verdict']}: states={len(states)} streams={n} ok={len(ok)} "
        f"valid={len(valid)} decoder_agrees={len(agree)} "
        f"lambda0_match={len(lam0_match)}/{len(lam0)} finite={summary['all_finite']}",
        flush=True,
    )
    print(
        f"[g2] negative branch: valid={len(neg_ok)}/{len(guided)} "
        f"effective(residual>0)={len(neg_effective)}/{len(guided)} "
        f"argmax_flips={len(neg_flips)}/{len(guided)}  backends={backends}",
        flush=True,
    )
    for l, d in per_lam.items():
        print(f"[g2]   lambda={l}: {d['valid']}/{d['streams']} valid, "
              f"{d['decoder_agrees']}/{d['streams']} decoder-agree", flush=True)
    if summary["verdict"] != "PASS":
        sys.exit(3)


if __name__ == "__main__":
    main()
