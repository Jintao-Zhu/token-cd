"""Serve the official OpenPI pi0-FAST base checkpoint for a SIMPLER smoke run."""
from __future__ import annotations

import logging
import json
import os
from pathlib import Path

import numpy as np
import tyro

from openpi import transforms
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared.normalize import NormStats
from openpi.training import config


ROOT = Path("/home/leju-suzhou/zjt_ws")
DEFAULT_CHECKPOINT = ROOT / "checkpoints/openpi-assets/openpi-assets/checkpoints/pi0_fast_base"


def main() -> None:
    import dataclasses

    @dataclasses.dataclass
    class ServerArgs:
        checkpoint: Path = DEFAULT_CHECKPOINT
        norm_stats: Path | None = None
        port: int = 8117

    args = tyro.cli(ServerArgs)
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Missing pi0-FAST params: {args.checkpoint / 'params'}")
    stats_path = args.norm_stats
    if stats_path is None:
        candidates = list((args.checkpoint / "assets/local").glob("*/norm_stats.json"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Pass --norm-stats explicitly; expected exactly one assets/local/*/norm_stats.json under {args.checkpoint}, found {candidates}"
            )
        stats_path = candidates[0]
    source_stats = json.loads(stats_path.read_text())["norm_stats"]
    stats = {
        "state": NormStats(
            mean=np.asarray(source_stats["state"]["mean"], dtype=np.float32),
            std=np.asarray(source_stats["state"]["std"], dtype=np.float32),
            q01=np.asarray(source_stats["state"]["q01"], dtype=np.float32),
            q99=np.asarray(source_stats["state"]["q99"], dtype=np.float32),
        ),
        "actions": NormStats(
            mean=np.asarray(source_stats["actions"]["mean"], dtype=np.float32),
            std=np.asarray(source_stats["actions"]["std"], dtype=np.float32),
            q01=np.asarray(source_stats["actions"]["q01"], dtype=np.float32),
            q99=np.asarray(source_stats["actions"]["q99"], dtype=np.float32),
        ),
    }

    cfg = config.get_config("pi0_fast_libero")
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
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint,
        repack_transforms=repack,
        norm_stats=stats,
    )
    logging.info("Loaded pi0-FAST policy from %s", args.checkpoint)
    logging.info("Using checkpoint normalization stats from %s", stats_path)

    # Compile the autoregressive FAST decode before the environment starts.
    sample = {
        "observation/primary_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "pick up the coke can",
    }
    warm = policy.infer(sample)
    actions = np.asarray(warm["actions"])
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise RuntimeError(f"FAST warmup produced invalid action chunk: {actions.shape}")
    logging.info("FAST decode warmup passed: actions=%s", actions.shape)

    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={"model": "pi0-FAST", "checkpoint": str(args.checkpoint), "action_horizon": 10, "action_dim": 7},
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
