#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from research.coreact_local_ev_confirmation.modeling import (
    LOCAL_FEATURES,
    MODEL_SPEC,
    PRIMARY_BRANCH,
    freeze_from_spatial,
)
from research.coreact_quality_negative_branch.quality_branch import FLOW_TIMES


DATA_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
OBJECT_SOURCE_TASKS = tuple(range(20, 30))
SELECTION_SEED = 20260811
EPISODES_PER_TASK = 40
NOISE_SEEDS_PER_STATE = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def scalar_stat(value) -> int:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"expected scalar task statistic, got {value}")
        return int(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"expected scalar task statistic, got {value}")
        return int(value[0])
    return int(value)


def complete_object_episodes(dataset: Path) -> tuple[dict[int, list[dict]], list[dict]]:
    episode_table = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    metadata = {int(row["episode_index"]): row for row in episode_table}
    observed: dict[int, dict] = {}
    file_audit = []
    for path in sorted((dataset / "data/chunk-000").glob("*.parquet")):
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "index", "task_index"]
        ).to_pylist()
        relevant = [row for row in table if int(row["task_index"]) in OBJECT_SOURCE_TASKS]
        if not relevant:
            continue
        file_audit.append(
            {
                "path": str(path),
                "file_index": int(path.stem.split("-")[-1]),
                "bytes": path.stat().st_size,
            }
        )
        for row in relevant:
            episode = int(row["episode_index"])
            item = observed.setdefault(
                episode,
                {"task_indices": set(), "frames": set(), "indices": set()},
            )
            item["task_indices"].add(int(row["task_index"]))
            item["frames"].add(int(row["frame_index"]))
            item["indices"].add(int(row["index"]))

    by_task = {task: [] for task in OBJECT_SOURCE_TASKS}
    for episode, item in observed.items():
        if len(item["task_indices"]) != 1:
            raise RuntimeError(f"episode {episode} spans task indices {item['task_indices']}")
        task = next(iter(item["task_indices"]))
        meta = metadata[episode]
        length = int(meta["length"])
        expected_frames = set(range(length))
        expected_indices = set(range(int(meta["dataset_from_index"]), int(meta["dataset_to_index"])))
        metadata_task = scalar_stat(meta["stats/task_index/min"])
        if (
            task == metadata_task
            and item["frames"] == expected_frames
            and item["indices"] == expected_indices
        ):
            by_task[task].append(
                {
                    "episode_id": episode,
                    "episode_length": length,
                    "instruction": str(meta["tasks"][0]),
                }
            )
    for rows in by_task.values():
        rows.sort(key=lambda row: row["episode_id"])
    return by_task, file_audit


def build_manifests(dataset: Path) -> tuple[list[dict], list[dict], dict, list[dict]]:
    available, file_audit = complete_object_episodes(dataset)
    rng = np.random.default_rng(SELECTION_SEED)
    states = []
    units = []
    availability = {}
    for task_id, source_task_index in enumerate(OBJECT_SOURCE_TASKS):
        candidates = available[source_task_index]
        availability[str(source_task_index)] = len(candidates)
        if len(candidates) < EPISODES_PER_TASK:
            raise RuntimeError(
                f"source task {source_task_index} has only {len(candidates)} complete episodes"
            )
        chosen_indices = sorted(
            int(value)
            for value in rng.choice(len(candidates), size=EPISODES_PER_TASK, replace=False)
        )
        for episode_ordinal, candidate_index in enumerate(chosen_indices):
            candidate = candidates[candidate_index]
            frame = (candidate["episode_length"] - 1) // 2
            state_id = f"object_task{task_id:02d}__episode{candidate['episode_id']:04d}__middle"
            state = {
                "state_id": state_id,
                "task_id": task_id,
                "source_task_index": source_task_index,
                "episode_id": candidate["episode_id"],
                "episode_ordinal": episode_ordinal,
                "episode_length": candidate["episode_length"],
                "frame_id": frame,
                "instruction": candidate["instruction"],
                "target_progress": 0.5,
            }
            states.append(state)
            for noise_ordinal in range(NOISE_SEEDS_PER_STATE):
                units.append(
                    {
                        **state,
                        "unit_id": f"{state_id}__noise{noise_ordinal}",
                        "noise_ordinal": noise_ordinal,
                        "noise_seed": 202608310000
                        + task_id * 10000
                        + episode_ordinal * 10
                        + noise_ordinal,
                    }
                )
    return states, units, availability, file_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = (
        args.output
        or workspace
        / "artifacts"
        / f"coreact_local_ev_object_confirmation_v1_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    if artifact.exists():
        raise FileExistsError(artifact)
    for child in ("frozen", "raw", "plots", "status", "logs"):
        (artifact / child).mkdir(parents=True, exist_ok=True)

    phase1 = workspace / "artifacts/coreact_selective_cfg_ev_validity_phase1_v1_20260811_203336"
    spatial_path = phase1 / "point_level_metrics.parquet"
    spatial = pd.read_parquet(spatial_path)
    bundle, predictions, metadata = freeze_from_spatial(spatial)
    model_path = artifact / "frozen/spatial_local_ev_gate.joblib"
    joblib.dump(bundle, model_path)
    predictions.to_parquet(artifact / "frozen/spatial_out_of_task_predictions.parquet", index=False)
    write_json(artifact / "frozen/spatial_freeze.json", metadata)

    dataset = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero"
    states, units, availability, source_files = build_manifests(dataset)
    write_jsonl(artifact / "state_manifest.jsonl", states)
    write_jsonl(artifact / "unit_manifest.jsonl", units)

    code_paths = sorted((workspace / "research/coreact_local_ev_confirmation").glob("*.py"))
    code_hashes = {str(path.relative_to(workspace)): sha256(path) for path in code_paths}
    manifest_path = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/verified_calibration_download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_revision"] != DATA_REVISION:
        raise RuntimeError("dataset revision mismatch")
    verified_files = {int(row["index"]): row for row in manifest["records"]}
    for source in source_files:
        record = verified_files.get(source["file_index"])
        if record is None or int(record["bytes"]) != source["bytes"]:
            raise RuntimeError(f"source file is absent from verified manifest: {source['path']}")
        source["sha256"] = record["sha256"]
    protocol = {
        "experiment_name": "coreact_local_ev_object_confirmation_v1",
        "stage": "independent_offline_libero_object_confirmation_no_rollout",
        "research_question": "Does the frozen W4 local-only Spatial EV-validity gate enrich EV-positive points on independent LIBERO-Object tasks?",
        "scope": {
            "suite": "libero_object",
            "source_task_indices": list(OBJECT_SOURCE_TASKS),
            "confirmation_task_ids": list(range(10)),
            "states_per_task": EPISODES_PER_TASK,
            "state_selection": "fixed-seed sample of complete episodes; one middle state at floor((length-1)/2); no outcome filtering",
            "states": len(states),
            "state_noise_units": len(units),
            "noise_seeds_per_state": NOISE_SEEDS_PER_STATE,
            "flow_times": list(FLOW_TIMES),
            "point_rows": len(units) * len(FLOW_TIMES),
            "closed_loop_forbidden": True,
        },
        "frozen_gate": {
            "branch": PRIMARY_BRANCH,
            "features": list(LOCAL_FEATURES),
            "model_class": "sklearn.ensemble.HistGradientBoostingClassifier",
            "model_spec": MODEL_SPEC,
            "calibration": "Platt logistic fitted only on Spatial out-of-task predictions",
            "threshold": float(bundle["threshold"]),
            "threshold_source": "Spatial out-of-task predictions only; frozen before Object capture",
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "object_data_used_for_fit_calibration_or_threshold": False,
        },
        "go_gate": {
            "mean_coverage_range_inclusive": [0.20, 0.35],
            "minimum_each_task_coverage": 0.10,
            "macro_precision_min": 0.70,
            "macro_precision_lift_min": 0.10,
            "tasks_positive_lift_min": 8,
            "task_bootstrap_precision_ci_lower_min": 0.65,
        },
        "decision_rules": {
            "pass": "LOCAL_EV_VALIDITY_INDEPENDENTLY_CONFIRMED_READY_FOR_LOCKED_CLOSED_LOOP_PROTOCOL",
            "fail": "LOCAL_EV_VALIDITY_NOT_CONFIRMED_STOP_NO_ROLLOUT",
            "automatic_closed_loop": False,
            "no_posthoc_changes": [
                "branch", "features", "classifier", "calibration", "threshold",
                "task-specific handling", "confirmation gate",
            ],
        },
        "data": {
            "repo": "HuggingFaceVLA/libero",
            "revision": DATA_REVISION,
            "path": str(dataset),
            "download_manifest": str(manifest_path),
            "download_manifest_sha256": sha256(manifest_path),
            "complete_episode_availability": availability,
            "selected_source_file_count": len(source_files),
            "selected_source_files": source_files,
            "selection_seed": SELECTION_SEED,
            "state_manifest_sha256": sha256(artifact / "state_manifest.jsonl"),
            "unit_manifest_sha256": sha256(artifact / "unit_manifest.jsonl"),
        },
        "development_source": {
            "artifact": str(phase1),
            "point_metrics": str(spatial_path),
            "point_metrics_sha256": sha256(spatial_path),
            "role": "development-only model, calibration, and threshold source",
        },
        "checkpoint": {
            "repo": "lerobot/smolvla_libero",
            "revision": CHECKPOINT_REVISION,
        },
        "implementation_sha256": code_hashes,
    }
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    audit = {
        "artifact": str(artifact),
        "states": len(states),
        "units": len(units),
        "tasks": len(set(row["task_id"] for row in states)),
        "states_each_task": {
            str(task): sum(row["task_id"] == task for row in states) for task in range(10)
        },
        "frozen_threshold": float(bundle["threshold"]),
        "frozen_model_sha256": sha256(model_path),
        "protocol_sha256": sha256(artifact / "protocol.lock.yaml"),
        "object_ev_observed": False,
    }
    write_json(artifact / "pre_capture_audit.json", audit)
    (artifact / "status/protocol.locked").write_text("protocol frozen before Object capture\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
