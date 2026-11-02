"""Optimized pi0-FAST CD rollouts paired to the existing RT-full baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.semantic_token_cd.pi0_fast_jit_header_closed_loop import (
    PairedPolicy,
    TASKS,
)
from research.semantic_token_cd.pi0_fast_simpler_l6_matched_rollout import (
    _atomic_json,
    _environment,
    _episode,
)


ARM = "l6_jit_header"
OLD_ARM = "l6_matched_050"
EXPECTED_CHECKPOINT = "/home/leju-suzhou/zjt_ws/checkpoints/openpi_fast_finetuned/fractal"


def reference_pair(reference: Path, task: str, seed: int, rt_depth: int) -> tuple[dict, dict]:
    directory = reference / "episodes" / task
    vanilla = json.loads((directory / f"vanilla_seed{seed:03d}.json").read_text())
    old = json.loads((directory / f"{OLD_ARM}_seed{seed:03d}.json").read_text())
    for expected_arm, record in (("vanilla", vanilla), (OLD_ARM, old)):
        if record.get("status") != "complete" or record.get("task") != task or record.get("seed") != seed:
            raise RuntimeError(f"incomplete/mismatched historical {task}/{seed}/{expected_arm}")
        if record.get("arm") != expected_arm or record.get("checkpoint") != EXPECTED_CHECKPOINT:
            raise RuntimeError(f"historical arm/checkpoint mismatch {task}/{seed}/{expected_arm}")
        renderer = record.get("renderer", {})
        if renderer != {"shader_dir": "rt", "rt_max_path_depth": rt_depth,
                        "rt_samples_per_pixel": 32, "rt_use_denoiser": True}:
            raise RuntimeError(f"historical renderer mismatch {task}/{seed}/{expected_arm}: {renderer}")
    return vanilla, old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--rt-depth", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed_start <= args.seed_end <= 99:
        raise ValueError("seed range must lie in 0..99")

    # Audit every historical record before loading a renderer or contacting the model.
    references = {seed: reference_pair(args.reference, args.task, seed, args.rt_depth)
                  for seed in range(args.seed_start, args.seed_end + 1)}
    policy = PairedPolicy("127.0.0.1", args.port)
    env = _environment(args.task, args.rt_depth)
    try:
        for seed, (vanilla, old) in references.items():
            episode_path = args.out / "episodes" / args.task / f"{ARM}_seed{seed:03d}.json"
            if episode_path.is_file():
                record = json.loads(episode_path.read_text())
                if record.get("status") == "complete":
                    print(json.dumps({"event": "skip_complete", "task": args.task, "seed": seed}), flush=True)
                else:
                    record = _episode(policy, env, args.task, seed, ARM, args.out, args.rt_depth)
            else:
                record = _episode(policy, env, args.task, seed, ARM, args.out, args.rt_depth)
            comparison = {
                "task": args.task, "seed": seed, "gpu": args.gpu,
                "reference": str(args.reference),
                "vanilla_success": vanilla["success"],
                "old_l6_success": old["success"],
                "jit_l6_success": record["success"],
                "vanilla_old_rgb_equal": vanilla["initial_rgb_sha256"] == old["initial_rgb_sha256"],
                "vanilla_jit_rgb_equal": vanilla["initial_rgb_sha256"] == record["initial_rgb_sha256"],
                "old_jit_rgb_equal": old["initial_rgb_sha256"] == record["initial_rgb_sha256"],
                "vanilla_old_state_equal": vanilla["initial_agent_state_sha256"] == old["initial_agent_state_sha256"],
                "vanilla_jit_state_equal": vanilla["initial_agent_state_sha256"] == record["initial_agent_state_sha256"],
                "old_jit_state_equal": old["initial_agent_state_sha256"] == record["initial_agent_state_sha256"],
                "old_jit_steps_equal": old["steps"] == record["steps"],
            }
            comparison["strict_pair"] = all(comparison[key] for key in (
                "vanilla_old_rgb_equal", "vanilla_jit_rgb_equal", "old_jit_rgb_equal",
                "vanilla_old_state_equal", "vanilla_jit_state_equal", "old_jit_state_equal",
            ))
            _atomic_json(args.out / "comparisons" / args.task / f"seed{seed:03d}.json", comparison)
            print(json.dumps({"event": "comparison", **comparison}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
