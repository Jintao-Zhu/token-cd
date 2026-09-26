#!/usr/bin/env python3
"""Prepare the fixed 40-case L11 spatial-placebo rollout from an audited cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2,3,4,5,6")
    args = parser.parse_args()
    source = args.source.resolve()
    artifact = args.artifact.resolve()
    manifest = json.loads((source / "cases_manifest.json").read_text())
    cases = manifest["cases"]
    if len(cases) != 40 or len({c["case_id"] for c in cases}) != 40:
        raise RuntimeError("expected the audited 40-case held-out cohort")
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
        row = dict(case)
        row["arms"] = ["matched_rot180"]
        row["model_gpu"] = gpus[i % len(gpus)]
        queues[gpus[i % len(gpus)]].append(row)
    for gpu, rows in queues.items():
        (artifact / "cases" / f"gpu{gpu}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    locked = {
        "protocol_id": "LIBERO90_L11_MATCHED_SELECTOR_ROT180_V1",
        "parent_cohort": str(source),
        "parent_protocol_id": manifest["protocol_id"],
        "case_count": len(cases),
        "arms_added": ["matched_rot180"],
        "gpus": gpus,
        "renderer_backend": "osmesa",
        "checkpoint": "/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90",
        "component_changed": "rotate L11 16x16 attention score map by 180 degrees before top-m selection",
        "fixed": ["m from canonical source-target KMeans/entity union", "harmonic beta=0", "lambda=0.5", "clean prefix/decode", "task/init state", "renderer", "checkpoint"],
        "cases": cases,
    }
    (artifact / "PREREGISTRATION.json").write_text(json.dumps(locked, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(artifact), "cases": len(cases), "queue_sizes": {g: len(q) for g, q in queues.items()}}, indent=2))


if __name__ == "__main__":
    main()
