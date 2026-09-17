"""Prepare and audit labels for the L11 initial-state representation probe."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from research.semantic_token_cd.l11_state_probe_protocol import (
    FEATURE_NAMES,
    PROTOCOL,
    SEEDS,
    SHORT_TASKS,
    TASKS,
    atomic_json,
    file_sha256,
    load_outcomes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    canonical = args.canonical.resolve()
    paired_results = args.paired_results.resolve()
    outcomes = load_outcomes(paired_results)

    snapshot_missing = []
    category_counts = {}
    for baseline in ("shr", "vanilla"):
        by_task = defaultdict(Counter)
        for (task, _seed), result in outcomes.items():
            pair = (result["l11_matched"], result[baseline])
            category = {
                (True, False): "prefer_l11",
                (False, True): f"prefer_{baseline}",
                (True, True): "both_success",
                (False, False): "both_fail",
            }[pair]
            by_task[SHORT_TASKS[task]][category] += 1
        category_counts[baseline] = {
            task: dict(counts) for task, counts in sorted(by_task.items())
        }

    for task in TASKS:
        for seed in SEEDS:
            path = canonical / "snapshots" / task / f"seed_{seed:03d}.pkl"
            if not path.exists():
                snapshot_missing.append(str(path))
    if snapshot_missing:
        raise RuntimeError(f"missing canonical snapshots, first entries: {snapshot_missing[:5]}")

    repo = Path(__file__).resolve().parents[2]
    config = {
        "protocol_id": PROTOCOL,
        "purpose": "test whether frozen initial scene/action representations predict when L11 beats a baseline",
        "tasks": list(TASKS),
        "seeds": [min(SEEDS), max(SEEDS)],
        "state_count": len(TASKS) * len(SEEDS),
        "canonical_snapshot_artifact": str(canonical),
        "paired_results": str(paired_results),
        "paired_results_sha256": file_sha256(paired_results),
        "features": list(FEATURE_NAMES),
        "labels": {
            "unit": "episode",
            "primary": "L11-vs-SHR discordant outcome",
            "secondary": "L11-vs-Vanilla discordant outcome",
            "rule": "train only on discordant episodes; never broadcast an episode label to timesteps",
        },
        "evaluation": {
            "within_task": "stratified out-of-fold prediction independently per task",
            "cross_task": "leave-one-task-out without task identity features",
            "task_only_baseline": True,
            "pca": "fit inside each training fold only",
            "go": "AUC>=0.65, >=task-only+0.05, and cross-task median AUC>=0.60",
        },
        "category_counts": category_counts,
        "code_sha256": {
            "protocol": file_sha256(repo / "research/semantic_token_cd/l11_state_probe_protocol.py"),
            "prepare": file_sha256(Path(__file__).resolve()),
            "collector": file_sha256(repo / "research/semantic_token_cd/collect_l11_initial_state_features.py"),
            "analyzer": file_sha256(repo / "research/semantic_token_cd/analyze_l11_state_probe.py"),
        },
    }
    config_path = artifact / "CONFIG_LOCK.json"
    if config_path.exists():
        previous = __import__("json").loads(config_path.read_text())
        if previous != config:
            raise RuntimeError("state-probe config lock differs")
    atomic_json(config_path, config)
    print(__import__("json").dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
