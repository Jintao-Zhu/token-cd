"""Real-checkpoint equivalence and nonzero-guidance smoke for pi0.5-LIBERO."""
from __future__ import annotations

import argparse
import json

import numpy as np

from research.semantic_token_cd.pi05_matched_flow_cd import Pi05MatchedFlowCDPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()

    policy = Pi05MatchedFlowCDPolicy(gpu=args.gpu)
    rng = np.random.default_rng(args.seed)
    raw = {
        "observation/image": rng.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image": rng.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/state": rng.uniform(-0.5, 0.5, size=(8,)).astype(np.float32),
        "prompt": "pick up the black bowl and place it on the plate",
    }
    noise = rng.normal(size=(10, 32)).astype(np.float32)

    official = np.asarray(policy.policy.infer(raw, noise=noise)["actions"])
    lambda_zero = np.asarray(
        policy.infer(raw, seed=args.seed, guidance_scale=0.0, noise=noise)["actions"]
    )
    guided = np.asarray(
        policy.infer(raw, seed=args.seed, guidance_scale=0.5, noise=noise)["actions"]
    )

    report = {
        "protocol": "PI05_MATCHED_FLOW_CD_LAMBDA0_EQUIVALENCE_V1",
        "seed": args.seed,
        "official_shape": list(official.shape),
        "custom_shape": list(lambda_zero.shape),
        "lambda0_array_equal": bool(np.array_equal(official, lambda_zero)),
        "lambda0_max_abs_diff": float(np.max(np.abs(official - lambda_zero))),
        "guided_finite": bool(np.isfinite(guided).all()),
        "guided_differs_from_official": bool(not np.array_equal(guided, official)),
        "guided_max_abs_diff": float(np.max(np.abs(guided - official))),
        "trace": policy.last_trace,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["lambda0_array_equal"]:
        raise SystemExit("FAIL: lambda=0 does not reproduce official pi0.5")
    if not report["guided_finite"] or not report["guided_differs_from_official"]:
        raise SystemExit("FAIL: nonzero guidance is non-finite or degenerate")


if __name__ == "__main__":
    main()
