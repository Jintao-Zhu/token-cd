"""Paired SIMPLER rollouts comparing eager and compiled pi0-FAST CD headers."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.pi0_fast_simpler_l6_matched_rollout import (
    RolloutPolicy,
    ROOT,
    _environment,
    _episode,
)

TASKS = (
    "google_robot_open_drawer",
    "google_robot_pick_coke_can",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
)
ARMS = ("l6_eager_header", "l6_jit_header")


class PairedPolicy(RolloutPolicy):
    def __init__(self, host: str, port: int, guidance_lambda: float = 0.5):
        super().__init__(host, port)
        self.guidance_lambda = float(guidance_lambda)

    def chunk(self, arm: str, raw: dict, seed: int):
        if arm not in ARMS:
            raise ValueError(f"unknown pilot arm {arm!r}")
        jit_header = arm == "l6_jit_header"
        request = {
            **raw,
            "cd_arm": "l6_matched",
            "cd_seed": int(seed),
            "cd_lambda": self.guidance_lambda,
            "cd_jit_header": int(jit_header),
        }
        start = time.perf_counter()
        result = self.client.infer(request)
        elapsed = time.perf_counter() - start
        actions = np.asarray(result["actions"], dtype=np.float32)
        trace = dict(result["pi0_fast_matched_cd_trace"])
        meta = result.get("pi0_fast_server_meta", {})
        if meta.get("jit_header") is not jit_header or trace.get("jit_header") is not jit_header:
            raise RuntimeError(f"server header mode mismatch for {arm}: {meta}")
        if meta.get("lambda") != self.guidance_lambda or trace.get("lambda") != self.guidance_lambda:
            raise RuntimeError(
                f"server lambda mismatch for {arm}: expected={self.guidance_lambda}, meta={meta}"
            )
        if not trace.get("fast_action_stream_valid") or not trace.get("openpi_official_decoder_agrees"):
            raise RuntimeError(f"invalid guided action stream for {arm}: {trace}")
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid action chunk for {arm}: {actions.shape}")
        trace["policy_call_seconds"] = elapsed
        trace["action_chunk_sha256"] = hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest()
        return actions, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--rt-depth", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/openpi_fast_finetuned/fractal")
    parser.add_argument("--guidance-lambda", type=float, default=0.5)
    args = parser.parse_args()
    if args.gpu not in (1, 2, 3):
        raise ValueError(f"this pilot is authorized only on GPUs 1/2/3, got {args.gpu}")
    if any(seed < 0 or seed > 19 for seed in args.seeds):
        raise ValueError("this pilot is locked to seeds 0..19")
    if args.rt_depth != 8:
        raise ValueError("this pilot is locked to RT-full depth 8")
    if not 0.0 <= args.guidance_lambda <= 2.5:
        raise ValueError("guidance lambda is outside the pre-registered supported range [0, 2.5]")
    from research.semantic_token_cd import pi0_fast_simpler_l6_matched_rollout as rollout_module

    rollout_module._CHECKPOINT_FOR_LOG = str(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    policy = PairedPolicy("127.0.0.1", args.port, args.guidance_lambda)
    env = _environment(args.task, args.rt_depth)
    try:
        for seed in args.seeds:
            order = ARMS if seed % 2 == 0 else ARMS[::-1]
            records = {}
            for arm in order:
                path = args.out / "episodes" / args.task / f"{arm}_seed{seed:03d}.json"
                if path.is_file():
                    record = json.loads(path.read_text())
                    if record.get("status") == "complete":
                        records[arm] = record
                        continue
                records[arm] = _episode(
                    policy, env, args.task, seed, arm, args.out, args.rt_depth
                )
            eager, jit = records[ARMS[0]], records[ARMS[1]]
            if eager["initial_rgb_sha256"] != jit["initial_rgb_sha256"]:
                raise RuntimeError(
                    f"paired reset RGB mismatch for {args.task}/{seed}: "
                    f"{eager['initial_rgb_sha256']} != {jit['initial_rgb_sha256']}"
                )
            if eager["initial_agent_state_sha256"] != jit["initial_agent_state_sha256"]:
                raise RuntimeError(
                    f"paired reset state mismatch for {args.task}/{seed}: "
                    f"{eager['initial_agent_state_sha256']} != {jit['initial_agent_state_sha256']}"
                )
            eager_chunks = [x["trace"] for x in eager["chunks"]]
            jit_chunks = [x["trace"] for x in jit["chunks"]]
            pair = {
                "task": args.task,
                "seed": seed,
                "gpu": args.gpu,
                "guidance_lambda": args.guidance_lambda,
                "rt_depth": args.rt_depth,
                "initial_rgb_equal": eager["initial_rgb_sha256"] == jit["initial_rgb_sha256"],
                "initial_agent_state_equal": (
                    eager["initial_agent_state_sha256"] == jit["initial_agent_state_sha256"]
                ),
                "success_eager": eager["success"],
                "success_jit": jit["success"],
                "steps_eager": eager["steps"],
                "steps_jit": jit["steps"],
                "first_action_chunk_equal": bool(
                    eager_chunks and jit_chunks
                    and eager_chunks[0]["action_chunk_sha256"] == jit_chunks[0]["action_chunk_sha256"]
                ),
                "first_selected_tokens_equal": bool(
                    eager_chunks and jit_chunks
                    and eager_chunks[0]["selected_token_ids"] == jit_chunks[0]["selected_token_ids"]
                ),
                "replans_eager": len(eager_chunks),
                "replans_jit": len(jit_chunks),
                "all_action_chunks_equal": (
                    len(eager_chunks) == len(jit_chunks)
                    and all(
                        a["action_chunk_sha256"] == b["action_chunk_sha256"]
                        for a, b in zip(eager_chunks, jit_chunks)
                    )
                ),
                "all_selected_tokens_equal": (
                    len(eager_chunks) == len(jit_chunks)
                    and all(
                        a["selected_token_ids"] == b["selected_token_ids"]
                        for a, b in zip(eager_chunks, jit_chunks)
                    )
                ),
                "mean_call_seconds_eager": float(np.mean(
                    [x["policy_call_seconds"] for x in eager_chunks]
                )),
                "mean_call_seconds_jit": float(np.mean(
                    [x["policy_call_seconds"] for x in jit_chunks]
                )),
            }
            destination = args.out / "pairs" / args.task / f"seed{seed:03d}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(pair, indent=2) + "\n")
            print("PAIR " + json.dumps(pair), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
