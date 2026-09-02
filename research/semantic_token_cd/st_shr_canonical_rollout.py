"""Run one ST-SHR arm against an existing canonical paired-snapshot artifact."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.semantic_recon_rollout import TASKS
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)
from research.semantic_token_cd.st_shr_rollout import (
    KMEANS_K,
    KMEANS_SEED,
    LAMBDA,
    arm_name,
    audit_trace,
    build_policy,
    finite_mean,
    parse_seeds,
)


PROTOCOL = "ST_SHR_BETA100_CANONICAL_PAIRED_V1"
REFERENCE_ARMS = ("vanilla", "semantic_recon_k8_m10", "shr_harmonic")


def reference_summary(task_root: Path, seed: int) -> dict | None:
    for arm in REFERENCE_ARMS:
        path = task_root / arm / f"episode_{seed:03d}_summary.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--beta", type=float, default=1.0, choices=(1.0,))
    parser.add_argument("--seeds", default="0-299")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument("--wait-for-snapshots", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    snapshot_artifact = args.snapshot_artifact.resolve()
    source_task_root = snapshot_artifact / "episodes" / args.task
    snapshot_dir = snapshot_artifact / "snapshots" / args.task
    arm = arm_name(args.beta)
    arm_dir = artifact / "episodes" / args.task / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config), args.beta)
    pending = set(parse_seeds(args.seeds))

    while pending:
        made_progress = False
        waiting: set[int] = set()
        for seed in sorted(pending):
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                pending.remove(seed)
                made_progress = True
                continue

            snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
            reference = reference_summary(source_task_root, seed)
            if not snapshot_path.exists() or reference is None:
                waiting.add(seed)
                continue

            with snapshot_path.open("rb") as handle:
                snapshot = pickle.load(handle)
            canonical = snapshot_sha(snapshot)
            if reference.get("canonical_snapshot_sha256") != canonical:
                raise RuntimeError(f"canonical snapshot hash mismatch: {args.task} seed={seed}")

            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if reference.get("initial_state_sha256") != state_sha:
                raise RuntimeError(f"initial state hash mismatch: {args.task} seed={seed}")
            if reference.get("initial_rgb_sha256") != rgb_sha:
                raise RuntimeError(f"initial RGB hash mismatch: {args.task} seed={seed}")

            instruction = env.unwrapped.get_language_instruction()
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            started = time.monotonic()
            result, steps, reason, actions, action_jerk = run_episode(
                env, policy, instruction, obs
            )
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            audit = audit_trace(trace, args.beta)
            write_arrays(arrays_path, policy._episode_logits, actions)
            displacements = [
                entity.get("centroid_displacement")
                for step in trace
                for entity in step.get("entity_alignment", [])
            ]
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "evaluation_seed": seed,
                "episode_id": seed,
                "arm": arm,
                "beta": args.beta,
                "lambda": LAMBDA,
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "instruction": instruction,
                "success": bool(result["success"]),
                "result": jsonable(result),
                "failure_reason": reason,
                "trajectory_length": steps,
                "control_steps": steps,
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "reference_arm": reference["arm"],
                "snapshot_artifact": str(snapshot_artifact),
                "arrays_file": arrays_path.name,
                "temporal_prior_used_fraction": finite_mean(
                    step["temporal_prior_used_fraction"] for step in trace
                ),
                "fallback_fraction": finite_mean(step["fallback_fraction"] for step in trace),
                "semantic_centroid_displacement": finite_mean(displacements),
                "residual_temporal_cos": finite_mean(
                    step.get("residual_temporal_cos") for step in trace
                ),
                "residual_jerk": finite_mean(step.get("residual_jerk") for step in trace),
                "action_jerk": action_jerk,
                "action_jitter_index": action_jerk,
                "selector_trace": trace,
                **audit,
            }
            tmp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, summary_path)
            pending.remove(seed)
            made_progress = True
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": summary["success"],
                "steps": steps,
                "runtime_seconds": round(runtime, 2),
                "technical_pass": True,
            }), flush=True)

        if not pending:
            break
        if not args.wait_for_snapshots:
            print(json.dumps({"waiting_for_snapshot_or_reference": sorted(waiting)}), flush=True)
            break
        print(json.dumps({
            "task": args.task,
            "waiting_for_snapshot_or_reference": len(waiting),
            "remaining": len(pending),
        }), flush=True)
        time.sleep(args.poll_seconds if not made_progress else min(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
