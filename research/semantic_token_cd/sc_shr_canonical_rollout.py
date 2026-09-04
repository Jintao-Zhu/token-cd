"""SC-SHR canonical paired rollout (one arm against a released snapshot set).

The arm under test is SC-SHR (Spatially Coherent SHR). Each (task, seed) is
restored from the released canonical snapshot pickle and hash-verified against
the reference ``shr_harmonic`` summary before the episode runs. Vanilla and SHR
arms are NOT rerun: their released summaries are the comparison baseline.
"""

from __future__ import annotations

import argparse
import copy
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
from research.semantic_token_cd.sc_shr_policy import ScShrHarmonicReconCDInference
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
    audit_shr_trace,
)


PROTOCOL = "SC_SHR_CANONICAL_PAIRED_V1"
ARM = "sc_shr_harmonic"
REFERENCE_ARM = "shr_harmonic"


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


def build_policy(base, task: str):
    policy = copy.copy(base)
    policy.__class__ = ScShrHarmonicReconCDInference
    policy.alpha = LAMBDA
    policy.lambd = LAMBDA
    policy.selection_mode = "semantic"
    policy.recon_selection_mode = "semantic"
    policy.kmeans_K = KMEANS_K
    policy.kmeans_seed = KMEANS_SEED
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    policy._task_id = TASK_INDEX[task]
    return policy


def finite_mean(values) -> float:
    vals = [float(v) for v in values if v is not None]
    vals = [v for v in vals if v == v]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def reference_summary(source_task_root: Path, seed: int) -> dict | None:
    path = source_task_root / REFERENCE_ARM / f"episode_{seed:03d}_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/sc_shr_canonical_v1")
    ap.add_argument("--snapshot-artifact", default=(
        "artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", default="0-299")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--worker-id", default="manual")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = Path(args.artifact).resolve()
    snapshot_artifact = Path(args.snapshot_artifact).resolve()
    source_task_root = snapshot_artifact / "episodes" / args.task
    snapshot_dir = snapshot_artifact / "snapshots" / args.task
    arm_dir = artifact / "episodes" / args.task / ARM
    arm_dir.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config), args.task)

    for seed in parse_seeds(args.seeds):
        summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
        arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            continue
        snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
        reference = reference_summary(source_task_root, seed)
        if not snapshot_path.exists() or reference is None:
            raise RuntimeError(f"missing snapshot/reference: {args.task} seed={seed}")

        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        if reference["canonical_snapshot_sha256"] != canonical:
            raise RuntimeError(f"canonical hash mismatch: {args.task} seed={seed}")

        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if reference["initial_state_sha256"] != state_sha:
            raise RuntimeError(f"state hash mismatch: {args.task} seed={seed}")
        if reference["initial_rgb_sha256"] != rgb_sha:
            raise RuntimeError(f"RGB hash mismatch: {args.task} seed={seed}")

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
        audit = audit_shr_trace(trace)
        write_arrays(arrays_path, policy._episode_logits, actions)

        summary = {
            "protocol_id": PROTOCOL,
            "environment_id": environment_id,
            "task": args.task,
            "seed": seed,
            "episode_id": seed,
            "arm": ARM,
            "lambda": LAMBDA,
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
            "reference_arm": REFERENCE_ARM,
            "arrays_file": arrays_path.name,
            "selector_trace": trace,
            "sc_mean_before_tokens": finite_mean(
                s.get("sc_before_tokens") for s in trace),
            "sc_mean_after_tokens": finite_mean(
                s.get("sc_after_tokens") for s in trace),
            "sc_mean_dropped_tokens": finite_mean(
                (s.get("sc_before_tokens", 0) or 0)
                - (s.get("sc_after_tokens", 0) or 0)
                for s in trace),
            "sc_mean_n_components": finite_mean(
                s.get("sc_n_components") for s in trace),
            "sc_mean_n_kept": finite_mean(s.get("sc_n_kept") for s in trace),
            **audit,
        }
        tmp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, summary_path)
        print(json.dumps({
            "task": args.task,
            "seed": seed,
            "arm": ARM,
            "success": summary["success"],
            "steps": steps,
            "before": summary["sc_mean_before_tokens"],
            "after": summary["sc_mean_after_tokens"],
            "technical_pass": audit["technical_pass"],
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
