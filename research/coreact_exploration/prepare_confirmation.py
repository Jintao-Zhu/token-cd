#!/usr/bin/env python3
"""Preregister fresh two-suite calibration, development, and held-out splits."""

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
SPATIAL_TASKS = list(range(10, 20))
OBJECT_TASKS = list(range(20, 30))
HELDOUT_TASKS = [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]


def eligible_episodes(dataset: Path):
    observed, task_for_episode = defaultdict(list), {}
    for path in sorted((dataset / "data/chunk-000").glob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"])
        for episode, frame, task in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            table["task_index"].to_pylist(),
            strict=True,
        ):
            if task in SPATIAL_TASKS + OBJECT_TASKS:
                observed[episode].append(frame)
                task_for_episode[episode] = task
    output = defaultdict(list)
    for episode, frames in observed.items():
        if sorted(frames) == list(range(max(frames) + 1)):
            output[task_for_episode[episode]].append(episode)
    return {task: sorted(episodes) for task, episodes in output.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    dataset = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero"
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = workspace / "artifacts" / f"coreact_exploration_v8_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    episodes = eligible_episodes(dataset)
    metadata = {
        row["episode_index"]: row
        for row in pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    }
    task_text = {
        row["task_index"]: row["__index_level_0__"]
        for row in pq.read_table(dataset / "meta/tasks.parquet").to_pylist()
    }
    rng = np.random.default_rng(SPLIT_SEED)
    heldout, development, calibration = [], [], []
    for task in SPATIAL_TASKS + OBJECT_TASKS:
        values = np.asarray(episodes[task], dtype=np.int64)
        values = values[rng.permutation(len(values))].tolist()
        heldout_count = 10 if task in HELDOUT_TASKS else 0
        heldout.extend((task, episode) for episode in values[:heldout_count])
        cursor = heldout_count
        dev_count = 2 if task in [10, 11, 12, 13] else 1
        development.extend((task, episode) for episode in values[cursor : cursor + dev_count])
        cursor += dev_count
        calibration.extend((task, episode) for episode in values[cursor : cursor + 13])
    calibration = calibration[:256]
    rows = []
    phases = [("early", 0.2), ("middle", 0.5), ("late", 0.8)]
    for index, (task, episode) in enumerate(development):
        meta = metadata[episode]
        phase, fraction = phases[index % 3]
        rows.append(
            {
                "task_id": task,
                "suite": "LIBERO-Spatial" if task < 20 else "LIBERO-Object",
                "episode_id": episode,
                "frame_id": int(round((meta["length"] - 1) * fraction)),
                "instruction": task_text[task],
                "phase": phase,
                "split": "development",
                "split_seed": SPLIT_SEED,
                "noise_seeds": NOISE_SEEDS,
                "random_ranking_seed": RANDOM_RANKING_SEED,
            }
        )
    for task, episode in calibration:
        meta = metadata[episode]
        for fraction in (1 / 3, 2 / 3):
            rows.append(
                {
                    "task_id": task,
                    "suite": "LIBERO-Spatial" if task < 20 else "LIBERO-Object",
                    "episode_id": episode,
                    "frame_id": int(round((meta["length"] - 1) * fraction)),
                    "instruction": task_text[task],
                    "phase": "unknown",
                    "split": "mean_calibration",
                    "split_seed": SPLIT_SEED,
                    "noise_seeds": NOISE_SEEDS,
                    "random_ranking_seed": RANDOM_RANKING_SEED,
                }
            )
    for index, (task, episode) in enumerate(heldout):
        meta = metadata[episode]
        phase, fraction = phases[index % 3]
        rows.append(
            {
                "task_id": task,
                "suite": "LIBERO-Spatial" if task < 20 else "LIBERO-Object",
                "episode_id": episode,
                "frame_id": int(round((meta["length"] - 1) * fraction)),
                "instruction": task_text[task],
                "phase": phase,
                "split": "heldout",
                "split_seed": SPLIT_SEED,
                "noise_seeds": NOISE_SEEDS,
                "random_ranking_seed": RANDOM_RANKING_SEED,
            }
        )
    counts = split_manifest(rows)
    if counts != {"mean_calibration": 512, "development": 24, "heldout": 120}:
        raise AssertionError(counts)
    protocol = {
        "experiment_name": "coreact_smolvla_offline_attribution_v1_two_suite_confirmation",
        "stage": "confirmation_preregistered_no_heldout_results_viewed",
        "primary_question": "token ranking versus grouped intervention magnitude",
        "secondary_question": "anchor versus nuisance sign",
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION},
        "dataset": {
            "repo": "HuggingFaceVLA/libero",
            "evaluation_snapshot_revision": DATASET_REVISION,
            "claimed_as_training_revision": False,
            "task_suites": ["LIBERO-Spatial", "LIBERO-Object"],
            "spatial_manifest": "verified_spatial_download_manifest_20260806.json",
            "object_manifest": "verified_calibration_download_manifest.json",
        },
        "precision": "fp32",
        "integrity_tolerances": {
            "determinism_forward_noop": 1e-6,
            "batch_serial_velocity": 1e-4,
            "attention_probability": 1e-6,
        },
        "tau_values": [0.2, 0.5, 0.8],
        "noise_seeds": NOISE_SEEDS,
        "split_seed": SPLIT_SEED,
        "random_ranking_seed": RANDOM_RANKING_SEED,
        "development_states": 24,
        "mean_calibration_states": 512,
        "heldout_states": 120,
        "heldout_tasks": 12,
        "heldout_task_suites_or_families": 2,
        "states_per_task": 10,
        "visual_uniform_groups_per_state": 24,
        "visual_top_groups_per_state": 8,
        "visual_bottom_groups_per_state": 8,
        "replacement_primary": "position_conditioned_modality_mean",
        "replacement_sensitivity": ["global_modality_mean", "zero"],
        "selection_score": "median_over_preregistered_flow_conditions_of_late_half_action_to_context_attention",
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
        "old_artifact_results_or_denominators_reused": False,
    }
    (output / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    with (output / "sample_manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_json = lambda path, payload: path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_json(
        output / "split_audit.json",
        {
            "split_counts": counts,
            "episode_disjoint": True,
            "heldout_task_ids": HELDOUT_TASKS,
            "heldout_suite_counts": {"LIBERO-Spatial": 60, "LIBERO-Object": 60},
            "heldout_results_viewed": False,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
