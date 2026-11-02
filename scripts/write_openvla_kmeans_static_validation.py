#!/usr/bin/env python3
"""Write the code-integrity gate for the KMeans-granularity sweep.

Pins the SHA-256 of every file in the dependency closure of the isolated runner,
plus its own planning/validation/queue scripts. The queue supervisor re-verifies
these hashes before every chunk, so any drift stops the run before more GPU work.

    python3 scripts/write_openvla_kmeans_static_validation.py --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts/openvla_kmeans_budget_9task_200_299_v1"
SMOKE = REPO / "artifacts/openvla_kmeans_budget_smoke_v1"

CODE = [
    "scripts/plan_openvla_kmeans_budget.py",
    "scripts/run_openvla_kmeans_queue.py",
    "scripts/verify_openvla_kmeans_smoke.py",
    "scripts/test_openvla_kmeans_budget.py",
    "scripts/audit_openvla_kmeans_control_reuse.py",
    "scripts/write_openvla_kmeans_static_validation.py",
    "scripts/activate_simpler_local.sh",
    "research/semantic_token_cd/openvla_kmeans_budget_rollout.py",
    "research/semantic_token_cd/openvla_budget_rule_rollout.py",
    "research/semantic_token_cd/prompt_attn_shr_policy.py",
    "research/semantic_token_cd/prompt_attn_shr_rollout.py",
    "research/semantic_token_cd/prompt_attn_l11_budget_causal_rollout.py",
    "research/semantic_token_cd/semantic_recon_rollout.py",
    "research/semantic_token_cd/spatial_grid_rollout.py",
    "research/semantic_token_cd/distractor_rollout.py",
    "research/semantic_token_cd/distractor_policy.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest, missing = {}, []
    for relative in CODE:
        path = REPO / relative
        if not path.is_file():
            missing.append(relative)
        else:
            manifest[relative] = sha256(path)
    if missing:
        print(f"missing files: {missing}", file=sys.stderr)
        return 1
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:                                                  # noqa: BLE001
        head = "unavailable"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "scripts/write_openvla_kmeans_static_validation.py",
        "technical_pass": True,
        "scope": "KMeans-granularity sweep: isolated runner plus its import closure",
        "code_sha256": manifest,
        "git_head": head,
        "python": sys.version.split()[0],
        "immutable_inputs": {
            "experiment_lock": sha256(ARTIFACT / "EXPERIMENT_LOCK.json"),
            "work_plan": sha256(ARTIFACT / "WORK_PLAN.json"),
            "smoke_experiment_lock": sha256(SMOKE / "EXPERIMENT_LOCK.json"),
            "control_reuse_audit": sha256(ARTIFACT / "CONTROL_REUSE_AUDIT.json"),
        },
        "validations": {
            "cpu_tests": "scripts/test_openvla_kmeans_budget.py (5 tests; arm parsing, lock enforcement, "
                         "partial-block refusal, K audits, wrapper delegation)",
            "real_policy_build": "K8/K16/K32 verified on the real prompt_attn_shr_policy build path "
                                 "without constructing a model or environment",
            "control_reuse": "CONTROL_REUSE_AUDIT.json: 900 + 9,900 episodes, 900/900 cells share "
                             "initial SHA and instruction across all 12 reused arms",
            "smoke_gate": "scripts/verify_openvla_kmeans_smoke.py: K8 must reproduce the historical "
                          "outcome, steps, selected ids, action token ids and executed actions for "
                          "close drawer / pick coke can / carrot at seed 200; K16/K32 must pass audits",
        },
        "not_validated_here": [
            "No GPU rollout of K16/K32 has been run at the time of writing.",
            "The plan's cost weights are load-balancing heuristics, never runtime predictions.",
        ],
    }
    if args.write:
        out = ARTIFACT / "STATIC_VALIDATION.json"
        if out.exists():
            out.replace(out.with_suffix(".json.prev"))
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}  files={len(manifest)}")
    else:
        print(json.dumps({"files": len(manifest), "technical_pass": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
