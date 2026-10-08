"""One-state technical smoke for the Pi0-FAST L6 matched-CD adapter."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/home/leju-suzhou/zjt_ws/token-cd")
from research.semantic_token_cd.pi0_fast_matched_cd import CHECKPOINT, Pi0FastMatchedCDPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = Path("/home/leju-suzhou/zjt_ws/token-cd")
    state_dir = root / "artifacts/spatialvla_layer_selection_v1b/states" / args.task / f"seed_{args.seed:03d}"
    meta = json.loads((state_dir / "meta.json").read_text())
    with np.load(state_dir / "input.npz") as data:
        image = np.asarray(data["image"], dtype=np.uint8)
    raw = {
        "observation/primary_image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": meta["instruction"],
    }

    started = time.monotonic()
    policy = Pi0FastMatchedCDPolicy(gpu=args.gpu, checkpoint=args.checkpoint)
    loaded = time.monotonic()
    import jax
    import jax.numpy as jnp

    vanilla_item = policy.policy._input_transform(dict(raw))
    vanilla_batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], vanilla_item)
    vanilla_obs = policy.model_lib.Observation.from_dict(vanilla_batch)
    vanilla_tokens = np.asarray(
        policy.policy._sample_actions(jax.random.key(args.seed), vanilla_obs)[0], dtype=np.int32
    )
    vanilla_stdout = io.StringIO()
    with contextlib.redirect_stdout(vanilla_stdout):
        vanilla_result = policy.policy._output_transform(
            {"state": vanilla_item["state"], "actions": vanilla_tokens}
        )
    vanilla_actions = np.asarray(vanilla_result["actions"], dtype=np.float32)
    result = policy.infer(raw, seed=args.seed)
    finished = time.monotonic()
    actions = np.asarray(result["actions"], dtype=np.float32)
    trace = result["pi0_fast_matched_cd_trace"]
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise RuntimeError(f"invalid decoded action chunk: shape={actions.shape}, finite={np.isfinite(actions).all()}")
    if trace["selector_layer_label"] != "L6" or not trace["selected_count_equals_matched"]:
        raise RuntimeError(f"L6/matched invariant failed: {trace}")

    vanilla_warning = vanilla_stdout.getvalue().strip()
    action_stream_valid = bool(
        trace["fast_action_stream_valid"] and trace["openpi_official_decoder_agrees"]
    )
    vanilla_stream_valid = not bool(vanilla_warning)
    report = {
        "status": "PASS" if action_stream_valid and vanilla_stream_valid else "PARTIAL",
        "technical_smoke_only": True,
        "attention_matched_cd_path_passed": True,
        "action_token_streams_valid": action_stream_valid and vanilla_stream_valid,
        "task": args.task,
        "seed": args.seed,
        "rgb_sha256": meta["rgb_sha256"],
        "checkpoint": str(args.checkpoint),
        "trace": trace,
        "first_action": actions[0].tolist(),
        "vanilla_control": {
            "output_action_shape": list(vanilla_actions.shape),
            "output_action_finite": bool(np.isfinite(vanilla_actions).all()),
            "first_action": vanilla_actions[0].tolist(),
            "action_stream_valid": vanilla_stream_valid,
            "decoder_warning": vanilla_warning[:300],
        },
        "load_seconds": round(loaded - started, 2),
        "inference_seconds": round(finished - loaded, 2),
    }
    output = args.out or (root / "artifacts/pi0_fast_l6_matched_cd_v1/smoke" / f"{args.task}_seed{args.seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"REPORT={output}")


if __name__ == "__main__":
    main()
