"""Isolated KMeans-granularity extension of the frozen budget-rule runner.
No shared policy/default edits. One arm and one historical seed block per process.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ARM_K = {f"matched_k{k:02d}_l100": k for k in (8, 16, 32)}

def parse_arm(arm):
    if arm not in ARM_K:
        raise ValueError(f"unsupported K sweep arm: {arm}")
    return "matched", 1.0, ARM_K[arm]

def validate_request(artifact, task, arm, seeds, gpu, protocol, snapshot):
    if gpu not in (1, 2, 3):
        raise ValueError("only GPU1/2/3 authorized")
    parse_arm(arm)
    lock_path = artifact / "EXPERIMENT_LOCK.json"
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if (lock["protocol_id"] != protocol or task not in lock["tasks"]
        or arm not in lock["arms"] or seeds not in lock["seed_blocks"]
        or not set(seeds).issubset(lock["seeds"])
        or lock["kmeans_K"][arm] != ARM_K[arm]
        or lock["lambda"] != 1.0 or lock["kmeans_seed"] != 0
        or Path(lock["snapshot_artifact"]).resolve() != snapshot.resolve()):
        raise ValueError("request differs from immutable experiment lock")
    out = artifact / "episodes" / task / arm
    for seed in seeds:
        for suffix in ("summary.json", "arrays.npz"):
            if (out / f"episode_{seed:03d}_{suffix}").exists():
                raise RuntimeError("existing output: partial-block skip/resume is forbidden")
    return lock, hashlib.sha256(lock_bytes).hexdigest()

def k_checks(trace, arm):
    k = ARM_K[arm]
    return {
        "kmeans_K_exact": bool(trace) and all(x.get("kmeans_K") == k for x in trace),
        "kmeans_seed_exact": bool(trace) and all(x.get("kmeans_seed") == 0 for x in trace),
        "matched_source_exact": bool(trace) and all(x.get("coverage_mode") == "own_state_matched_budget" for x in trace),
        "matched_union_count_exact": bool(trace) and all(
            len(set(x.get("reference_shr_token_ids", []))) == x.get("m_t") for x in trace),
        "selected_ids_exact": bool(trace) and all(
            len(set(x.get("selected_token_ids", []))) == x.get("m_t")
            and all(isinstance(i, int) and 0 <= i < 256 for i in x.get("selected_token_ids", [])) for x in trace),
        "matched_groups_exact": bool(trace) and all(
            x.get("selected_group_ids") == x.get("matched_budget_group_ids")
            and bool(x.get("selected_group_ids"))
            and all(isinstance(i, int) and 0 <= i < k for i in x.get("selected_group_ids", [])) for x in trace),
    }

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--snapshot-artifact", type=Path, default=Path("artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--lock-seeds", default="200-299")
    args = ap.parse_args()
    if args.gpu not in (1, 2, 3):
        raise ValueError("only GPU1/2/3 authorized")
    # Set visibility before importing torch or any model dependencies.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from research.semantic_token_cd import openvla_budget_rule_rollout as base
    seeds = base.parse_seeds(args.seeds)
    lock, lock_sha = validate_request(args.artifact.resolve(), args.task, args.arms,
                                     seeds, args.gpu, args.protocol_id, args.snapshot_artifact)
    if base.parse_seeds(args.lock_seeds) != lock["seeds"]:
        raise ValueError("lock-seeds differ")
    original_build, original_audit, original_atomic = base.build_policies, base.audit, base.atomic_json
    base.parse_arm = parse_arm

    def build(model, task, arms):
        policies = original_build(model, task, arms)
        for arm, policy in policies.items():
            policy.kmeans_K = ARM_K[arm]
            if policy.kmeans_seed != 0:
                raise RuntimeError("unexpected original KMeans seed")
        return policies

    def audit(trace, arm, task):
        checks = original_audit(trace, arm, task)
        checks.update(k_checks(trace, arm))
        checks["technical_pass"] = all(v is True for k, v in checks.items() if k != "technical_pass")
        if not checks["technical_pass"]:
            raise RuntimeError(f"K sweep audit failed: {checks}")
        return checks

    def atomic(path, payload):
        path = Path(path)
        if path.name == "CONFIG_LOCK.json":
            # Never concurrently merge shared configuration. Immutable EXPERIMENT_LOCK
            # is pre-created; retain this invocation's inherited config separately.
            path = args.artifact.resolve() / "invocations" / args.task / args.arms / f"seeds_{seeds[0]}_{seeds[-1]}_config.json"
            payload = {**payload, "kmeans_K": {args.arms: ARM_K[args.arms]},
                       "kmeans_seed": 0, "seed_block": seeds,
                       "experiment_lock_sha256": lock_sha,
                       "vary": "KMeans K only; same matched union-count and L11 ranking"}
        elif path.name.endswith("_summary.json"):
            if payload.get("technical_pass") is not True:
                raise RuntimeError("refusing failed summary")
            payload = {**payload, "kmeans_K": ARM_K[args.arms], "kmeans_seed": 0,
                       "process_seed_block": seeds, "experiment_lock_sha256": lock_sha}
        path.parent.mkdir(parents=True, exist_ok=True)
        original_atomic(path, payload)

    if (args.artifact / "CONFIG_LOCK.json").exists():
        raise RuntimeError("legacy mutable CONFIG_LOCK not permitted in isolated artifact")
    base.build_policies, base.audit, base.atomic_json = build, audit, atomic
    base.main()

if __name__ == "__main__":
    main()
