#!/usr/bin/env python3
"""Compare eager and JIT header on one identical SIMPLER reset observation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.pi0_fast_jit_header_closed_loop import PairedPolicy, TASKS
from research.semantic_token_cd.pi0_fast_simpler_l6_matched_rollout import (
    _eef_pos, _environment, _raw_policy_input,
)


def digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    env = _environment(args.task, 8)
    try:
        obs, _ = env.reset(seed=args.seed)
        image = np.asarray(obs["image"]["overhead_camera"]["rgb"])
        state = _eef_pos(obs["agent"], env)
        raw = _raw_policy_input(image, str(env.unwrapped.get_language_instruction()), obs["agent"], env)
        policy = PairedPolicy("127.0.0.1", args.port)
        seed = args.seed * 1009
        eager_actions, eager = policy.chunk("l6_eager_header", raw, seed)
        jit_actions, jit = policy.chunk("l6_jit_header", raw, seed)
        report = {
            "task": args.task, "seed": args.seed,
            "initial_rgb_sha256": digest(image),
            "initial_agent_state_sha256": digest(state),
            "selected_equal": eager["selected_token_ids"] == jit["selected_token_ids"],
            "fast_tokens_equal": eager["raw_output_token_ids_full"] == jit["raw_output_token_ids_full"],
            "actions_equal": bool(np.array_equal(eager_actions, jit_actions)),
            "eager_first_residual_norm": eager["first_logit_residual_norm"],
            "jit_first_residual_norm": jit["first_logit_residual_norm"],
            "eager_tokens_prefix": eager["raw_output_token_ids_prefix"],
            "jit_tokens_prefix": jit["raw_output_token_ids_prefix"],
            "eager_seconds": eager["policy_call_seconds"],
            "jit_seconds": jit["policy_call_seconds"],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
