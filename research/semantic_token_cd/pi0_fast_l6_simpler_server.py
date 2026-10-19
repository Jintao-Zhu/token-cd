"""OpenPI websocket server exposing Vanilla and L6-Matched pi0-FAST inference."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PI0FAST_WORKSPACE_ROOT", Path.home() / "zjt_ws")).expanduser()
TOKEN_CD = Path(os.environ.get("PI0FAST_REPO_ROOT", Path(__file__).resolve().parents[2])).expanduser()


class ArmPolicy:
    def __init__(self, gpu: int, checkpoint: Path, norm_stats: Path | None = None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        sys.path.insert(0, str(TOKEN_CD))
        from research.semantic_token_cd import pi0_fast_matched_cd as cd_module
        from research.semantic_token_cd.pi0_fast_matched_cd import Pi0FastMatchedCDPolicy

        self.cd_module = cd_module
        # norm_stats=None keeps the historical default (Fractal stats).  Pass a path to serve a
        # checkpoint with its own statistics - e.g. the Bridge checkpoint must use
        # <ckpt>/assets/local/bridge_lerobot/norm_stats.json, not the Fractal ones.
        extra = {"stats_path": norm_stats} if norm_stats is not None else {}
        logging.info("Serving %s with norm stats: %s", checkpoint,
                     norm_stats if norm_stats is not None else cd_module.FRACTAL_STATS)
        self.adapter = Pi0FastMatchedCDPolicy(gpu=gpu, checkpoint=checkpoint, **extra)
        self.default_jit_header = self.adapter.jit_header
        self.jax = self.adapter.jax
        self.call_count = 0
        logging.info("JAX backend=%s devices=%s CPU=%s", self.jax.default_backend(), self.jax.devices(), self.jax.devices("cpu"))
        # Compile both official Vanilla and L6-CD paths before accepting client
        # traffic. This also verifies that the per-layer attention callback and
        # FAST decoding are live in the server's exact import environment.
        warm = {
            "observation/primary_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros((8,), dtype=np.float32),
            "prompt": "open the drawer",
        }
        self.adapter.policy._rng = self.jax.random.key(0)
        vanilla = self.adapter.policy.infer(dict(warm))
        if not np.isfinite(vanilla["actions"]).all():
            raise RuntimeError("OpenPI vanilla warmup produced non-finite actions")
        guided = self.adapter.infer(dict(warm), seed=0)
        trace = guided["pi0_fast_matched_cd_trace"]
        if not trace.get("fast_action_stream_valid") or not trace.get("openpi_official_decoder_agrees"):
            raise RuntimeError(f"L6-Matched server warmup/FAST decoder failed: {trace}")
        logging.info("Vanilla and L6-Matched warmups passed; trace=%s", {
            "selected_token_count": trace.get("selected_token_count"),
            "matched_m_t": trace.get("matched_m_t"),
            "selector_layer_label": trace.get("selector_layer_label"),
            "decoded_fast_payload_chars": trace.get("decoded_fast_payload_chars"),
        })

    def infer(self, observation: dict):
        request = dict(observation)
        arm = str(request.pop("cd_arm", "vanilla"))
        seed = int(request.pop("cd_seed", 0)) % (2**31 - 1)
        # Guidance strength travels with the request so one server can serve every lambda
        # arm. The exact decoder reads LAMBDA from the module global at call time, so setting
        # it here takes effect without reloading the model.
        lam = request.pop("cd_lambda", None)
        header_mode = request.pop("cd_jit_header", None)
        if arm == "l6_matched":
            if header_mode is not None and header_mode not in (0, 1, False, True):
                raise ValueError(f"cd_jit_header must be 0 or 1, got {header_mode!r}")
            self.adapter.jit_header = (
                self.default_jit_header if header_mode is None else bool(header_mode)
            )
            if lam is not None:
                self.cd_module.LAMBDA = float(lam)
                self.adapter.lambda_ = float(lam)
            result = self.adapter.infer(request, seed=seed)
        elif arm == "vanilla":
            # Deterministic paired replays, independent of socket scheduling or
            # server process history.
            self.adapter.policy._rng = self.jax.random.key(seed)
            result = self.adapter.policy.infer(request)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        self.call_count += 1
        result["pi0_fast_server_meta"] = {
            "arm": arm, "request_seed": seed, "request_index": self.call_count,
            "lambda": float(lam) if lam is not None else None,
            "kmeans_backend": self.cd_module.KMEANS_BACKEND_REQUESTED,
            "tap_mode": self.adapter.tap_mode,
            "decode_mode": self.adapter.decode_mode,
            "jit_header": self.adapter.jit_header,
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(os.environ.get(
            "PI0FAST_CHECKPOINT", ROOT / "checkpoints/openpi_fast_finetuned/fractal"
        )).expanduser(),
    )
    parser.add_argument("--norm-stats", default=None,
                        help="path to a norm_stats.json, or 'auto' to auto-discover "
                             "<checkpoint>/assets/local/*/norm_stats.json (one required)")
    args = parser.parse_args()

    norm_stats = None
    if args.norm_stats:
        if args.norm_stats == "auto":
            candidates = sorted((args.checkpoint / "assets/local").glob("*/norm_stats.json"))
            if len(candidates) != 1:
                raise SystemExit(
                    f"--norm-stats auto needs exactly one assets/local/*/norm_stats.json under "
                    f"{args.checkpoint}, found {candidates}")
            norm_stats = candidates[0]
        else:
            norm_stats = Path(args.norm_stats)
        if not Path(norm_stats).is_file():
            raise SystemExit(f"norm stats file not found: {norm_stats}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
    policy = ArmPolicy(args.gpu, args.checkpoint, norm_stats)
    # The adapter installs the temporary attention-instrumented OpenPI overlay.
    # Importing ``openpi.serving`` first would preload the uninstrumented package
    # root and make the L6 attention tap silently invisible.
    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={
            "model": "pi0-FAST",
            "checkpoint": str(args.checkpoint),
            "arms": ["vanilla", "l6_matched"],
            "selector": "L6 Prompt-Attention, dynamic entity-matched budget, harmonic reconstruction, lambda=0.5",
        },
    )
    logging.info("Loaded %s policy on physical GPU %s; serving port %s", args.checkpoint, args.gpu, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
