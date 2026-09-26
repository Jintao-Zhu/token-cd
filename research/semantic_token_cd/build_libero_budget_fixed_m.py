#!/usr/bin/env python3
"""Prepare the held-out fixed-task-budget arm using discovery-only m values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FIXED_M = {3: 16, 10: 25, 49: 47, 72: 42, 73: 46}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="held-out init IDs 8..15 cases")
    parser.add_argument("--discovery", type=Path, required=True, help="independent init IDs 0..7 m traces")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2,3,4,5,6")
    args = parser.parse_args()
    source, discovery, artifact = args.source.resolve(), args.discovery.resolve(), args.artifact.resolve()
    manifest = json.loads((source / "cases_manifest.json").read_text())
    cases = manifest["cases"]
    if len(cases) != 40 or any(int(c["init_state_id"]) < 8 for c in cases):
        raise RuntimeError("expected the fixed held-out 40-case cohort at init IDs 8..15")
    discovery_manifest = json.loads((discovery / "cases_manifest.json").read_text())
    if any(int(c["init_state_id"]) >= 8 for c in discovery_manifest["cases"]):
        raise RuntimeError("discovery cohort must be disjoint init IDs 0..7")
    gpus = [int(x) for x in args.gpus.split(",") if x]
    if len(set(gpus)) != len(gpus) or not gpus:
        raise ValueError("GPU list must be nonempty and unique")
    artifact.mkdir(parents=True, exist_ok=True)
    for name in ("cases", "pairs", "episodes", "logs"):
        path = artifact / name
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {path}")
        path.mkdir(exist_ok=True)
    queues = {gpu: [] for gpu in gpus}
    for i, case in enumerate(cases):
        task_id = int(case["task_id"])
        if task_id not in FIXED_M:
            raise RuntimeError(f"task {task_id} has no preregistered fixed-m value")
        row = dict(case)
        row["arms"] = ["matched_fixed_m"]
        row["fixed_m"] = FIXED_M[task_id]
        row["model_gpu"] = gpus[i % len(gpus)]
        queues[gpus[i % len(gpus)]].append(row)
    for gpu, rows in queues.items():
        (artifact / "cases" / f"gpu{gpu}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    locked = {
        "protocol_id": "LIBERO90_L11_MATCHED_FIXED_TASK_M_V1",
        "evaluation_source": str(source), "discovery_source": str(discovery),
        "case_count": len(cases), "gpus": gpus, "renderer_backend": "osmesa",
        "checkpoint": "/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90",
        "fixed_m_by_task_id": FIXED_M,
        "m_estimator": "nearest integer, half-up, median of per-replan canonical m_t over all discovery episodes in each task; outcomes unused",
        "changed_component": "intervention token count only",
        "fixed": ["L11 instruction-only score ranking", "harmonic beta=0", "lambda=0.5", "clean prefix/decode", "same task/init state", "same checkpoint", "OSMesa"],
        "cases": cases,
    }
    (artifact / "PREREGISTRATION.json").write_text(json.dumps(locked, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(artifact), "cases": len(cases), "fixed_m": FIXED_M,
                      "queue_sizes": {str(k): len(v) for k, v in queues.items()}}, indent=2))


if __name__ == "__main__":
    main()
