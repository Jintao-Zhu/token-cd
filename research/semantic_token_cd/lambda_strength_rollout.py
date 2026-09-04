"""Task-dependent guidance-strength analysis: lambda sweep on SHR.

Strictly paired rollout. For every (task, seed) the environment snapshot is
captured once and the three arms (lambda 0 / 0.25 / 0.5) are run inside the
same process by restoring that same snapshot, so all arms share the exact
initial state, RGB, and instruction. No recapture between arms.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)
from research.semantic_token_cd.spatial_harmonic_recon_rollout import (
    KMEANS_K,
    KMEANS_SEED,
    LAMBDA,
    TASK_INDEX,
    TASKS,
    build_policies,
)


PROTOCOL = "LAMBDA_STRENGTH_LOCAL_PAIRED_V1"
ARMS = {
    0.0: "lambda_0",
    0.25: "lambda_025",
    0.5: "lambda_05",
}


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def audit_shr_trace_light(trace: list[dict]) -> dict:
    if not trace:
        return {"technical_pass": False, "n_cd_steps": 0}
    finite = all(isinstance(s.get("residual_norm"), (int, float))
                 and s.get("residual_norm") == s.get("residual_norm") for s in trace)
    return {
        "technical_pass": bool(finite),
        "n_cd_steps": len(trace),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/lambda_strength_analysis_v1")
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", default="300-399")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--worker-id", default="manual")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = Path(args.artifact).resolve()
    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    log_dir = artifact / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**config)
    policies = build_policies(base, args.task)
    weak = copy.copy(policies["spatial_harmonic_recon"])
    weak.lambd = 0.25
    weak.alpha = 0.25
    arm_policies = {
        "lambda_0": policies["vanilla"],
        "lambda_025": weak,
        "lambda_05": policies["spatial_harmonic_recon"],
    }

    for seed in parse_seeds(args.seeds):
        complete = all(
            (task_root / arm / f"episode_{seed:03d}_summary.json").exists()
            and (task_root / arm / f"episode_{seed:03d}_arrays.npz").exists()
            for arm in ARMS.values()
        )
        if complete:
            continue

        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        hashes = []
        for arm in ARMS.values():
            arm_dir = task_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            hashes.append((arm, canonical, state_sha, rgb_sha))
            instruction = env.unwrapped.get_language_instruction()
            policy = arm_policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, action_jerk = run_episode(
                env, policy, instruction, obs
            )
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            audit = audit_shr_trace_light(trace) if arm != "lambda_0" else {
                "technical_pass": True, "n_cd_steps": steps}
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            write_arrays(arrays_path, policy._episode_logits, actions)
            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": args.task,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "lambda": float(ARMS_LAMBDA_BY_NAME[arm]),
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "instruction": instruction,
                "success": bool(result["success"]),
                "result": jsonable(result),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": action_jerk,
                "runtime_seconds": runtime,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                **audit,
            }
            tmp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, summary_path)
            print(json.dumps({
                "task": args.task, "seed": seed, "arm": arm,
                "success": summary["success"], "steps": steps,
                "lambda": summary["lambda"],
            }, sort_keys=True), flush=True)

        # Same-process pairing sanity: hashes must match across arms.
        assert len({h[1] for h in hashes}) == 1
        assert len({h[2] for h in hashes}) == 1
        assert len({h[3] for h in hashes}) == 1


ARMS_LAMBDA_BY_NAME = {name: lam for lam, name in ARMS.items()}


if __name__ == "__main__":
    main()
