#!/usr/bin/env python3
"""Preregister episode-separated single-suite CoreAct development states."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from research.coreact_exploration.metrics import split_manifest
from research.coreact_exploration.qualify_single_suite import CHECKPOINT_REVISION, DATASET_REVISION


SPLIT_SEED = 314159
NOISE_SEEDS = [1729, 9473]
RANDOM_RANKING_SEED = 271828
OBJECT_TASKS = list(range(20, 30))


def actual_complete_episodes(dataset: Path) -> dict[int, list[int]]:
    observed: dict[int, list[int]] = defaultdict(list)
    task_for_episode: dict[int, int] = {}
    for path in sorted((dataset / "data/chunk-000").glob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"])
        for episode, frame, task in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            table["task_index"].to_pylist(),
            strict=True,
        ):
            observed[episode].append(frame)
            task_for_episode[episode] = task
    output: dict[int, list[int]] = defaultdict(list)
    for episode, frames in observed.items():
        if sorted(frames) == list(range(max(frames) + 1)) and task_for_episode[episode] in OBJECT_TASKS:
            output[task_for_episode[episode]].append(episode)
    return {task: sorted(episodes) for task, episodes in output.items()}


def choose_manifest(dataset: Path) -> list[dict]:
    rng = np.random.default_rng(SPLIT_SEED)
    by_task = actual_complete_episodes(dataset)
    episode_meta = {
        row["episode_index"]: row
        for row in pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    }
    task_text = {
        row["task_index"]: row["__index_level_0__"]
        for row in pq.read_table(dataset / "meta/tasks.parquet").to_pylist()
    }
    development_episodes: list[tuple[int, int]] = []
    calibration_episodes: list[tuple[int, int]] = []
    for offset, task in enumerate(OBJECT_TASKS):
        episodes = np.asarray(by_task[task], dtype=np.int64)
        episodes = episodes[rng.permutation(len(episodes))]
        dev_count = 3 if offset < 4 else 2
        development_episodes.extend((task, int(ep)) for ep in episodes[:dev_count])
        calibration_episodes.extend((task, int(ep)) for ep in episodes[dev_count : dev_count + 26])
    # Exactly 256 calibration episodes: 26 for six tasks and 25 for four tasks.
    calibration_episodes = calibration_episodes[:256]

    rows: list[dict] = []
    phases = [("early", 0.2), ("middle", 0.5), ("late", 0.8)]
    for index, (task, episode) in enumerate(development_episodes):
        meta = episode_meta[episode]
        phase, fraction = phases[index % len(phases)]
        frame = min(meta["length"] - 1, int(round((meta["length"] - 1) * fraction)))
        rows.append(
            {
                "task_id": task,
                "episode_id": episode,
                "frame_id": frame,
                "instruction": task_text[task],
                "phase": phase,
                "split": "development",
                "split_seed": SPLIT_SEED,
                "noise_seeds": NOISE_SEEDS,
                "random_ranking_seed": RANDOM_RANKING_SEED,
            }
        )
    for task, episode in calibration_episodes:
        meta = episode_meta[episode]
        for fraction in (1 / 3, 2 / 3):
            frame = min(meta["length"] - 1, int(round((meta["length"] - 1) * fraction)))
            rows.append(
                {
                    "task_id": task,
                    "episode_id": episode,
                    "frame_id": frame,
                    "instruction": task_text[task],
                    "phase": "unknown",
                    "split": "mean_calibration",
                    "split_seed": SPLIT_SEED,
                    "noise_seeds": NOISE_SEEDS,
                    "random_ranking_seed": RANDOM_RANKING_SEED,
                }
            )
    if split_manifest(rows)["development"] != 24 or split_manifest(rows)["mean_calibration"] != 512:
        raise AssertionError("preregistered split sizes were not met")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    dataset = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero"
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = workspace / "artifacts" / f"coreact_exploration_v6_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "experiment_name": "coreact_smolvla_offline_attribution_v1_single_suite_development",
        "stage": "development_only",
        "primary_question": "token ranking versus grouped intervention magnitude",
        "secondary_question": "anchor versus nuisance sign",
        "backbone": {
            "repo": "lerobot/smolvla_libero",
            "revision": CHECKPOINT_REVISION,
        },
        "dataset": {
            "repo": "HuggingFaceVLA/libero",
            "evaluation_snapshot_revision": DATASET_REVISION,
            "claimed_as_training_revision": False,
            "task_subset": "LIBERO-Object task_index 20-29",
        },
        "precision": "fp32",
        "integrity_tolerances": {
            "determinism_forward_noop": 1.0e-6,
            "batch_serial_velocity": 1.0e-4,
            "attention_probability": 1.0e-6,
        },
        "tau_values": [0.2, 0.5, 0.8],
        "noise_seeds": NOISE_SEEDS,
        "split_seed": SPLIT_SEED,
        "random_ranking_seed": RANDOM_RANKING_SEED,
        "development_states": 24,
        "mean_calibration_states": 512,
        "heldout_states": 0,
        "heldout_status": "deferred_until_development_integrity_and_second_verified_suite",
        "heldout_tasks_required_by_original_protocol": 12,
        "heldout_task_suites_or_families_required": 2,
        "visual_uniform_groups_per_state": 24,
        "visual_top_groups_per_state": 8,
        "visual_bottom_groups_per_state": 8,
        "replacement_primary": "position_conditioned_modality_mean",
        "replacement_sensitivity": ["global_modality_mean", "zero"],
        "selection_score": "late_half_action_to_context_attention",
        "ranking_methods": [
            "deterministic_random",
            "embedding_norm",
            "raw_last_expert_layer_attention",
            "late_half_action_to_context_attention",
        ],
        "primary_metric": "per_state_spearman_on_uniformly_sampled_visual_groups",
        "secondary_metrics": [
            "top_minus_random_intervention_magnitude",
            "signed_relative_loss_change",
            "anchor_precision_at_k",
        ],
        "bootstrap_clusters": ["task", "episode"],
        "bootstrap_replicates": 2000,
        "multiple_comparison": "holm",
        "amendment": {
            "reason": "The verified local evaluation snapshot has complete coverage for one suite only.",
            "allowed_scope": "calibration and development integrity gates",
            "forbidden_scope": "held-out analysis or final PROCEED decision",
            "old_artifact_results_reused": False,
            "development_numerical_amendment": {
                "source_artifacts": [
                    "coreact_exploration_v4_20260806_232408",
                    "coreact_exploration_v5_20260806_233417",
                ],
                "heldout_viewed": False,
                "reason": "Separate same-shape no-op parity from batch-shape numerical drift.",
                "observed_batch_serial_max_abs_diff": 1.1831521987915039e-5,
                "locked_batch_serial_tolerance": 1.0e-4,
                "determinism_or_noop_tolerance_relaxed": False,
            },
        },
    }
    (output / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    manifest = choose_manifest(dataset)
    with (output / "sample_manifest.jsonl").open("w") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    split_counts = split_manifest(manifest)
    (output / "split_audit.json").write_text(
        json.dumps(
            {
                "split_counts": split_counts,
                "episode_disjoint": True,
                "development_task_count": len({r["task_id"] for r in manifest if r["split"] == "development"}),
                "heldout_rows_created": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
