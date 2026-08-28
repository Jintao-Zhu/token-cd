#!/usr/bin/env python3
"""Run an append-only integrity replacement trio without altering the locked manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.coreact_closed_loop.runtime import load_policy_and_processors
from research.coreact_task4_replication.run import run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reset-count", type=int, default=2)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    specs = [json.loads(line) for line in (artifact / "episode_manifest.amendment.jsonl").read_text().splitlines()]
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(
        workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",
        weights_only=True,
        map_location="cpu",
    )
    for index, spec in enumerate(specs, 1):
        run_one(artifact, config, policy, preprocessor, postprocessor, means, spec, reset_count=args.reset_count)
        print(f"repair {index}/{len(specs)} {spec['episode_id']}", flush=True)


if __name__ == "__main__":
    main()
