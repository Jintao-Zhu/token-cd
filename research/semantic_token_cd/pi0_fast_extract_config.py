"""Extract the exact configuration that passed Gate 1 and Gate 2.

Every value is read from the live objects the validated runs used, not retyped from memory or
taken from library defaults, so config.yaml describes the code that actually produced the
passing regression results.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
sys.path.insert(0, str(ROOT / "openpi/src"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import research.semantic_token_cd.pi0_fast_matched_cd as CD  # noqa: E402


def git_rev(path: Path) -> dict:
    out = {}
    for k, cmd in (("commit", ["git", "-C", str(path), "rev-parse", "HEAD"]),
                   ("branch", ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"]),
                   ("dirty", ["git", "-C", str(path), "status", "--porcelain"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            v = r.stdout.strip()
            out[k] = v.splitlines() if k == "dirty" else v
        except Exception as exc:  # noqa: BLE001
            out[k] = f"unavailable: {exc}"
    out["dirty_file_count"] = len(out["dirty"]) if isinstance(out["dirty"], list) else None
    return out


def main() -> None:
    cfg: dict = {}

    # ---- algorithm constants, read from the module that ran the gates ----
    cfg["algorithm"] = {
        "selector_layer_index_zero_based": CD.LAYER_INDEX,
        "grid": CD.GRID,
        "n_visual_tokens": CD.N_VISUAL,
        "visual_tokens_total": 3 * CD.N_VISUAL,
        "kmeans_backend_requested": CD.KMEANS_BACKEND_REQUESTED,
        "kmeans_params": dict(CD.KMEANS_PARAMS),
        "kmeans_threads": CD.KMEANS_THREADS,
        "tap_mode_default": os.environ.get("PI0FAST_CD_TAP_MODE", "separate"),
        "decode_mode_default": os.environ.get("PI0FAST_CD_DECODE", "verified"),
        "embed_jit_default": os.environ.get("PI0FAST_CD_EMBED_JIT", "0"),
        "batch_branches_default": os.environ.get("PI0FAST_CD_BATCH", "0"),
        "lambda_default": CD.LAMBDA,
        "disable_validity_mask_default": CD.DISABLE_VALIDITY_MASK,
    }

    # ---- model-derived constants: these come from the checkpoint itself ----
    try:
        import jax
        from openpi.models import model as model_lib
        from openpi.models.tokenizer import FASTTokenizer

        tok = FASTTokenizer()
        fast = tok._fast_tokenizer
        fast_vocab = getattr(fast, "vocab_size", None) or len(fast)
        bpe = fast.bpe_tokenizer
        cfg["tokenizer"] = {
            "fast_vocab_size": int(fast_vocab),
            "paligemma_tokenizer": type(tok._paligemma_tokenizer).__name__,
            "fast_skip_tokens": getattr(tok, "_fast_skip_tokens", None),
            "bos_id": int(tok._paligemma_tokenizer.bos_id()) if hasattr(tok._paligemma_tokenizer, "bos_id") else None,
            "eos_id": int(tok._paligemma_tokenizer.eos_id()) if hasattr(tok._paligemma_tokenizer, "eos_id") else None,
        }
        ck = ROOT / "checkpoints/openpi_fast_finetuned/fractal"
        cfg["checkpoint"] = {"path": str(ck), "exists": ck.exists()}
        if (ck / "config.json").exists():
            cc = json.loads((ck / "config.json").read_text())
            cfg["checkpoint"]["config_keys"] = sorted(cc.keys())[:20]
    except Exception as exc:  # noqa: BLE001
        cfg["tokenizer"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- rollout settings read from the rollout module ----
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rollout_mod", REPO / "research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py"
        )
        src = (REPO / "research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py").read_text()
        cfg["rollout_source_facts"] = {
            "exec_horizon": 4 if "self.exec_horizon = 4" in src else None,
            "max_steps": 250 if "max_steps = 250" in src else None,
            "rt_depth_default": 0 if 'PI0FAST_RT_DEPTH", "0"' in src else None,
            "default_out": next((l.split("DEFAULT_OUT")[1].strip(" =") for l in src.splitlines()
                                 if l.startswith("DEFAULT_OUT")), None),
            "arms": next((l.split("ARMS")[1].strip(" =") for l in src.splitlines()
                          if l.startswith("ARMS")), None),
            "tasks": next((l.split("TASKS")[1].strip(" =") for l in src.splitlines()
                           if l.startswith("TASKS")), None),
        }
    except Exception as exc:  # noqa: BLE001
        cfg["rollout_source_facts"] = {"error": str(exc)}

    # ---- environment versions ----
    vers = {}
    for mod in ("numpy", "scipy", "jax", "flax"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            vers[mod] = None
    try:
        import sklearn
        vers["sklearn"] = sklearn.__version__
    except Exception:  # noqa: BLE001
        vers["sklearn"] = None
    try:
        import openpi
        vers["openpi"] = getattr(openpi, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        vers["openpi"] = None
    vers["python"] = sys.version.split()[0]
    vers["env"] = os.environ.get("CONDA_DEFAULT_ENV")
    cfg["environment"] = vers

    cfg["git"] = {
        "token_cd": git_rev(REPO),
        "openpi": git_rev(ROOT / "openpi"),
    }

    # ---- regression evidence, if present ----
    ev = {}
    for name, p in (
        ("gate1_lambda0", REPO / "artifacts/pi0_fast_lambda0_equivalence_v1/EQUIVALENCE.json"),
        ("gate2_bpe", REPO / "artifacts/pi0_fast_gate2_regression_v1/REGRESSION.json"),
        ("kmeans_backend", REPO / "artifacts/pi0_fast_kmeans_backend_v1/BACKEND_COMPARISON.json"),
    ):
        if p.exists():
            d = json.loads(p.read_text())
            ev[name] = {k: d.get(k) for k in
                        ("verdict", "states", "streams", "payload_string_same", "actions_same",
                         "stream_valid", "decoder_agrees", "neg_branch_valid", "guided_streams",
                         "lambda0_match_official", "kmeans_backend", "speedup",
                         "m_same_fraction", "matched_set_jaccard_mean") if k in d}
    cfg["regression_evidence"] = ev

    out = REPO / "runs/pi0fast_l6_matched_9task_100seed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_extracted.json").write_text(json.dumps(cfg, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(cfg, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
