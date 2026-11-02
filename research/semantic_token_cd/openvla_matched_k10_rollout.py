"""Real closed-loop K=10 Matched arms at lambda 0.5 and 1.0.

Unlike the probe, this EXECUTES the K=10 budget: it is the same matched rule (own-state
KMeans entity-matched cluster union) with kmeans_K=10, and everything else identical to
`artifacts/openvla_lambda_budget_grid_200_299_v1` (same runner, same 4 Google tasks, same
seeds 200-299, same 11 reference process blocks, same snapshot artifact, same rendering),
so the result pairs one-to-one with that grid's `matched_l050` / `matched_l100`.

Arms: matched_k10_l050, matched_k10_l100.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

K10 = 10


def parse_arm(arm: str) -> tuple[str, float, int]:
    if not arm.startswith("matched_k10_l"):
        raise ValueError(f"unsupported arm for this study: {arm}")
    suffix = arm.rsplit("_l", 1)[1]
    if not suffix.isdigit():
        raise ValueError(f"cannot parse lambda from arm: {arm}")
    strength = int(suffix) / 100.0
    if strength not in (0.5, 1.0):
        raise ValueError(f"unsupported lambda in arm: {arm}")
    return "matched", strength, K10


def parse_seeds(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = map(int, part.split("-", 1))
            values.extend(range(low, high + 1))
        else:
            values.append(int(part))
    result = sorted(set(values))
    if not result:
        raise ValueError("no seeds")
    return result


def validate_request(artifact: Path, task: str, arms: list[str], seeds: list[int],
                     gpu: int, protocol: str, snapshot: Path) -> tuple[dict, str]:
    if gpu not in (1, 2, 3):
        raise ValueError("only GPU1/2/3 authorized")
    lock_path = artifact / "EXPERIMENT_LOCK.json"
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if lock["protocol_id"] != protocol:
        raise ValueError("protocol id differs from the lock")
    if task not in lock["tasks"] or seeds not in lock["seed_blocks"] or not set(seeds).issubset(lock["seeds"]):
        raise ValueError("task/seed block is not in the immutable lock")
    for arm in arms:
        rule, strength, k = parse_arm(arm)
        if arm not in lock["arms"]:
            raise ValueError(f"arm not in lock: {arm}")
        if lock["kmeans_K"][arm] != k or abs(lock["lambda"][arm] - strength) > 1e-12:
            raise ValueError(f"arm definition differs from the lock: {arm}")
    if Path(lock["snapshot_artifact"]).resolve() != snapshot.resolve():
        raise ValueError("snapshot artifact differs from the lock")
    for arm in arms:
        out = artifact / "episodes" / task / arm
        for seed in seeds:
            for suffix in ("summary.json", "arrays.npz"):
                if (out / f"episode_{seed:03d}_{suffix}").exists():
                    raise RuntimeError(f"existing output ({arm}): partial-block resume is forbidden")
    return lock, hashlib.sha256(lock_bytes).hexdigest()


def k_checks(trace: list, arm: str) -> dict:
    return {
        "kmeans_K_exact": bool(trace) and all(x.get("kmeans_K") == K10 for x in trace),
        "kmeans_seed_exact": bool(trace) and all(x.get("kmeans_seed") == 0 for x in trace),
        "matched_source_exact": bool(trace) and all(
            x.get("coverage_mode") == "own_state_matched_budget" for x in trace),
        "matched_union_count_exact": bool(trace) and all(
            len(set(x.get("reference_shr_token_ids", []))) == x.get("m_t") for x in trace),
        "selected_ids_exact": bool(trace) and all(
            len(set(x.get("selected_token_ids", []))) == x.get("m_t")
            and all(isinstance(i, int) and 0 <= i < 256 for i in x.get("selected_token_ids", []))
            for x in trace),
        "matched_groups_exact": bool(trace) and all(
            x.get("selected_group_ids") == x.get("matched_budget_group_ids")
            and bool(x.get("selected_group_ids"))
            and all(isinstance(i, int) and 0 <= i < K10 for i in x.get("selected_group_ids", []))
            for x in trace),
    }


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path,
                    default=Path("artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--lock-seeds", default="200-299")
    args = ap.parse_args()

    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    seeds = parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    lock, lock_sha = validate_request(artifact, args.task, list(arms), seeds, args.gpu,
                                     args.protocol_id, args.snapshot_artifact)
    if parse_seeds(args.lock_seeds) != lock["seeds"]:
        raise ValueError("lock-seeds differ from the lock")
    if (artifact / "CONFIG_LOCK.json").exists():
        raise RuntimeError("legacy mutable CONFIG_LOCK not permitted in this isolated artifact")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from research.semantic_token_cd import openvla_budget_rule_rollout as base

    original_build, original_audit, original_atomic = (base.build_policies, base.audit,
                                                       base.atomic_json)
    base.parse_arm = parse_arm

    def build(model, task, arm_names):
        policies = original_build(model, task, arm_names)
        for arm, policy in policies.items():
            if parse_arm(arm)[2] != K10:
                raise RuntimeError("this runner only executes K=10")
            policy.kmeans_K = K10
            if policy.kmeans_seed != 0:
                raise RuntimeError("unexpected KMeans seed")
        return policies

    def audit(trace, arm, task):
        checks = original_audit(trace, arm, task)
        checks.update(k_checks(trace, arm))
        checks["technical_pass"] = all(v is True for k, v in checks.items() if k != "technical_pass")
        if not checks["technical_pass"]:
            raise RuntimeError(f"K=10 audit failed for {arm}: {checks}")
        return checks

    def atomic(path, payload):
        path = Path(path)
        if path.name == "CONFIG_LOCK.json":
            path = (artifact / "invocations" / args.task / args.arms
                    / f"seeds_{seeds[0]}_{seeds[-1]}_config.json")
            payload = {**payload, "kmeans_K": {args.arms: K10}, "kmeans_seed": 0,
                       "seed_block": seeds, "experiment_lock_sha256": lock_sha}
        elif path.name.endswith("_summary.json"):
            if payload.get("technical_pass") is not True:
                raise RuntimeError("refusing failed summary")
            payload = {**payload, "kmeans_K": K10, "kmeans_seed": 0,
                       "process_seed_block": seeds, "experiment_lock_sha256": lock_sha}
        path.parent.mkdir(parents=True, exist_ok=True)
        original_atomic(path, payload)

    base.build_policies, base.audit, base.atomic_json = build, audit, atomic
    base.main()


if __name__ == "__main__":
    main()
