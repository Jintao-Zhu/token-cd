#!/usr/bin/env python3
"""Qualify the verified LIBERO-Spatial source snapshot for two-suite confirmation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch

from research.coreact_exploration.instrumentation import predict_teacher_forced_velocity
from research.coreact_exploration.qualify_single_suite import (
    CHECKPOINT_REVISION,
    Paths,
    load_preprocessor,
    tensor_sha256,
)
from research.coreact_exploration.run_development_gates import (
    CAMERA_IDS,
    SnapshotRows,
    load_policy,
    processed_state,
    write_json,
)


FIRST_INDEX = 101469
LAST_INDEX_EXCLUSIVE = 153511
EPISODE = 500
NOISE_SEED = 1729


def audit(dataset: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    episode_meta = {
        row["episode_index"]: row
        for row in pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    }
    frames, indices = defaultdict(list), defaultdict(list)
    tasks, schemas = Counter(), set()
    for record in manifest["records"]:
        path = dataset / record["path"]
        parquet = pq.ParquetFile(path)
        schemas.add(str(parquet.schema_arrow))
        table = pq.read_table(path, columns=["index", "episode_index", "frame_index", "task_index"])
        for index, episode, frame, task in zip(
            *(table[column].to_pylist() for column in ["index", "episode_index", "frame_index", "task_index"]),
            strict=True,
        ):
            if FIRST_INDEX <= index < LAST_INDEX_EXCLUSIVE:
                frames[episode].append(frame)
                indices[episode].append(index)
                tasks[task] += 1
    failures = []
    for episode, episode_frames in frames.items():
        meta = episode_meta[episode]
        if sorted(episode_frames) != list(range(meta["length"])):
            failures.append({"episode": episode, "reason": "frame discontinuity"})
        if sorted(indices[episode]) != list(range(meta["dataset_from_index"], meta["dataset_to_index"])):
            failures.append({"episode": episode, "reason": "global-index discontinuity"})
    return {
        "manifest": str(manifest_path),
        "manifest_all_verified": manifest["all_verified"],
        "shard_count": len(manifest["records"]),
        "total_source_bytes": manifest["total_bytes"],
        "schema_count": len(schemas),
        "suite_global_index_interval": [FIRST_INDEX, LAST_INDEX_EXCLUSIVE],
        "row_count": sum(tasks.values()),
        "expected_row_count": LAST_INDEX_EXCLUSIVE - FIRST_INDEX,
        "episode_count": len(frames),
        "episode_range": [min(frames), max(frames)],
        "rows_per_task": dict(sorted(tasks.items())),
        "continuity_failures": failures,
        "pass": manifest["all_verified"]
        and len(schemas) == 1
        and sum(tasks.values()) == LAST_INDEX_EXCLUSIVE - FIRST_INDEX
        and len(frames) == 428
        and not failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = workspace / "artifacts" / f"coreact_exploration_v7_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    dataset = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero"
    paths = Paths(
        workspace,
        dataset,
        workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots" / CHECKPOINT_REVISION,
        workspace / "counterfactual-flow-vla/pi05_svcpd_p0/verified_calibration_download_manifest.json",
    )
    source = audit(dataset, args.manifest.resolve())
    write_json(output / "spatial_source_audit.json", source)
    store = SnapshotRows(dataset)
    meta = store.episode_rows[EPISODE]
    frame = meta["length"] // 2
    manifest_row = {
        "episode_id": EPISODE,
        "frame_id": frame,
        "instruction": meta["tasks"][0],
    }
    preprocessor = load_preprocessor(paths)
    batch = processed_state(store, preprocessor, manifest_row)
    policy = load_policy(paths)
    images, masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    generator = torch.Generator(device=actions.device).manual_seed(NOISE_SEED)
    noise = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
    tau = torch.tensor([0.5], device=actions.device, dtype=actions.dtype)
    with torch.inference_mode():
        probe = predict_teacher_forced_velocity(
            policy.model,
            images,
            masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            state,
            actions,
            tau,
            noise,
            record_attention=True,
            camera_ids=CAMERA_IDS,
        )
        loss, _ = policy(batch, noise=noise, time=tau, reduction="none")
    action_dim = batch["action"].shape[-1]
    valid = (~batch["action_is_pad"]).unsqueeze(-1).expand(-1, -1, action_dim)
    manual = (probe["v_pred"][:, :, :action_dim] - probe["u_target"][:, :, :action_dim]).square()
    manual_loss = manual[valid].mean()
    batch_gate = {
        "episode_id": EPISODE,
        "frame_id": frame,
        "instruction": meta["tasks"][0],
        "processed_batch_sha256": tensor_sha256(batch),
        "camera_shapes": [list(image.shape) for image in images],
        "camera_masks_all_valid": [bool(mask.all()) for mask in masks],
        "state_input_dim": int(batch["observation.state"].shape[-1]),
        "state_model_dim": int(state.shape[-1]),
        "action_input_dim": action_dim,
        "language_valid_tokens": int(batch["observation.language.attention_mask"].sum()),
        "prefix_length": len(probe["prefix_span_map"][0]),
        "expert_trace_count": len(probe["attention_trace"]),
        "forward_parity_abs_diff": float((manual_loss - loss[0]).abs()),
        "policy_eval": not policy.training,
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
    }
    batch_gate["pass"] = (
        batch_gate["camera_masks_all_valid"] == [True, True]
        and batch_gate["state_input_dim"] == 8
        and batch_gate["state_model_dim"] == 32
        and batch_gate["action_input_dim"] == 7
        and batch_gate["language_valid_tokens"] > 0
        and batch_gate["expert_trace_count"] == 16
        and batch_gate["forward_parity_abs_diff"] <= 1e-6
        and batch_gate["policy_eval"]
        and batch_gate["all_parameters_frozen"]
    )
    write_json(output / "real_batch_forward_gate.json", batch_gate)
    decision = {
        "status": "QUALIFIED_DATA_READY" if source["pass"] and batch_gate["pass"] else "INCONCLUSIVE_DATA_OR_RESOURCES",
        "suite": "LIBERO-Spatial",
        "source_gate": source["pass"],
        "real_batch_forward_gate": batch_gate["pass"],
        "two_verified_suites_now_available": source["pass"] and batch_gate["pass"],
        "heldout_created_or_viewed": False,
    }
    write_json(output / "decision.json", decision)
    print(output)
    print(json.dumps(decision, indent=2))
    return 0 if decision["status"] == "QUALIFIED_DATA_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
