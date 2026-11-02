"""Matched-budget probe: run the standard K8 intervention, but additionally record what
the matched budget WOULD be at other cluster counts, on the very same states.

Why: changing K changes the cluster size (about 256/K tokens per cluster), so K16/K32 did
not test "finer granularity" - they tested a much smaller budget.  Before any further
closed-loop sweep we need the actual m_t(K) curve.  This runner keeps the intervention at
K=8 (so the trajectory is the standard one) and evaluates KMeans for a ladder of K values
on the same projector features h at every control step.

Output per episode: the normal summary (untouched schema) plus a side file
`episode_NNN_probe.json` with per-step budgets for every probe K.

Audit: reuses the frozen K8 audit; the probe only adds measurements.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REAL_K = 8
PROBE_KS = (8, 9, 10, 11, 12, 14, 16, 20)
ARM = "lambda_100"


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


def validate_request(artifact: Path, task: str, seeds: list[int], gpu: int, protocol: str) -> dict:
    if gpu not in (1, 2, 3):
        raise ValueError("only GPU1/2/3 authorized")
    lock = json.loads((artifact / "EXPERIMENT_LOCK.json").read_text())
    if (lock["protocol_id"] != protocol or task not in lock["tasks"]
            or seeds not in lock["seed_blocks"] or not set(seeds).issubset(lock["seeds"])):
        raise ValueError("request differs from the immutable probe lock")
    if tuple(lock["probe_Ks"]) != tuple(PROBE_KS) or lock["real_K"] != REAL_K:
        raise ValueError("probe ladder differs from the code")
    out = artifact / "episodes" / task / ARM
    for seed in seeds:
        if (out / f"episode_{seed:03d}_probe.json").exists():
            raise RuntimeError("existing probe output: partial-block resume is forbidden")
    return lock


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--arms", default=ARM)
    ap.add_argument("--snapshot-artifact", type=Path,
                    default=Path("artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--lock-seeds", default="200-299")
    args = ap.parse_args()
    if args.arms.replace(" ", "") not in (ARM, f"{ARM},{ARM}"):
        raise ValueError(f"probe runs the matched {ARM} arm only, got {args.arms!r}")
    artifact = args.artifact.resolve()
    seeds = parse_seeds(args.seeds)
    validate_request(artifact, args.task, seeds, args.gpu, args.protocol_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from research.semantic_token_cd import openvla_budget_rule_rollout as base

    original_build = base.build_policies
    original_atomic = base.atomic_json

    def build(model, task, arms):
        policies = original_build(model, task, arms)
        for arm, policy in policies.items():
            original_clusters = policy._semantic_clusters
            original_groups = policy._entity_groups

            def clusters(h, _policy=policy, _clusters=original_clusters, _groups=original_groups):
                labels, groups, scores = _clusters(h)
                entry = {}
                for k in PROBE_KS:
                    if k == REAL_K:
                        continue
                    _policy.kmeans_K = int(k)
                    try:
                        labels_k, _, _ = _clusters(h)
                        matched = []
                        for group in _groups(h, labels_k):
                            if int(group) not in matched:
                                matched.append(int(group))
                        tokens = set()
                        for group in matched:
                            tokens.update(int(i) for i in np.flatnonzero(labels_k == group))
                        entry[str(k)] = {"m": len(tokens), "groups": matched,
                                         "cluster_sizes": [int((labels_k == g).sum()) for g in matched]}
                    finally:
                        _policy.kmeans_K = REAL_K
                _policy._probe_log.append(entry)
                return labels, groups, scores

            def reset(instruction, seed=None, _policy=policy, _orig=policy.reset):
                _policy._probe_log = []
                return _orig(instruction, seed=seed)

            policy._probe_log = []
            policy._semantic_clusters = clusters
            policy.reset = reset
            if policy.kmeans_K != REAL_K:
                raise RuntimeError("probe requires the standard K8 matched policy")
            policy._probe_task = task
        return policies

    def atomic(path, payload):
        path = Path(path)
        if path.name.endswith("_summary.json") and "selector_trace" in payload:
            trace = payload["selector_trace"]
            probe_log = []
            for policy in base_policies.values():
                probe_log = getattr(policy, "_probe_log", [])
                break
            per_step = []
            if len(probe_log) == len(trace):
                for step, (entry, row) in enumerate(zip(probe_log, trace)):
                    record = {"step": step, "m_k8": int(row.get("m_t", -1))}
                    record.update({f"m_k{k}": v["m"] for k, v in entry.items()})
                    record.update({f"groups_k{k}": v["groups"] for k, v in entry.items()})
                    per_step.append(record)
            else:
                per_step = [{"error": f"probe/trace length mismatch: {len(probe_log)} vs {len(trace)}"}]
            summary = {}
            for k in PROBE_KS:
                key = f"m_k{k}"
                values = [row.get(key) for row in per_step if isinstance(row.get(key), int)]
                summary[key] = {"mean": float(np.mean(values)) if values else None,
                                "min": min(values) if values else None,
                                "max": max(values) if values else None,
                                "n_steps": len(values)}
            probe_payload = {
                "task": payload.get("task"), "seed": payload.get("seed"), "arm": payload.get("arm"),
                "protocol_id": payload.get("protocol_id"), "real_K": REAL_K,
                "probe_Ks": list(PROBE_KS), "steps": payload.get("steps"),
                "success": payload.get("success"),
                "matched_groups_k8_mean": float(np.mean([
                    len(row.get("matched_budget_group_ids") or []) for row in trace])) if trace else None,
                "per_step": per_step, "summary": summary,
                "note": "Intervention stays at K8; the other K columns are counterfactual budgets "
                        "measured on the same states and are not executed.",
            }
            probe_path = path.with_name(path.name.replace("_summary.json", "_probe.json"))
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            original_atomic(probe_path, probe_payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        original_atomic(path, payload)

    base_policies = {}
    inner_build = build

    def build_tracking(model, task, arms):
        policies = inner_build(model, task, arms)
        base_policies.clear()
        base_policies.update(policies)
        return policies

    base.build_policies = build_tracking
    base.atomic_json = atomic
    base.main()


if __name__ == "__main__":
    main()
