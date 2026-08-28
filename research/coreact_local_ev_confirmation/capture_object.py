#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORKSPACE / "lerobot/src"), str(WORKSPACE)]

from research.coreact_exploration.qualify_single_suite import (
    CHECKPOINT_REVISION,
    Paths,
    load_preprocessor,
)
from research.coreact_exploration.run_development_gates import (
    SnapshotRows,
    load_policy,
    processed_state,
)
from research.coreact_quality_negative_branch.quality_branch import (
    FLOW_TIMES,
    point_metrics,
    reconstruct_training_pair,
    tensor_sha256,
    velocity_from_embeddings,
)
from research.coreact_local_ev_confirmation.modeling import LOCAL_FEATURES, PRIMARY_BRANCH


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def repeated(value: torch.Tensor, count: int) -> torch.Tensor:
    return value.expand(count, *value.shape[1:])


def local_features(strong: torch.Tensor, weak: torch.Tensor, valid: torch.Tensor) -> dict:
    strong_vector = strong[valid].float()
    weak_vector = weak[valid].float()
    direction = strong_vector - weak_vector
    strong_norm = float(torch.linalg.vector_norm(strong_vector))
    weak_norm = float(torch.linalg.vector_norm(weak_vector))
    direction_norm = float(torch.linalg.vector_norm(direction))
    output = {
        "strong_weak_cosine": float(
            torch.dot(strong_vector, weak_vector)
            / (torch.linalg.vector_norm(strong_vector) * torch.linalg.vector_norm(weak_vector) + 1e-12)
        ),
        "relative_correction_norm": direction_norm / (strong_norm + 1e-12),
        "log_strong_norm": float(math.log1p(strong_norm)),
        "log_weak_norm": float(math.log1p(weak_norm)),
        "log_direction_norm": float(math.log1p(direction_norm)),
        "weak_to_strong_norm_ratio": weak_norm / (strong_norm + 1e-12),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise FloatingPointError("nonfinite local feature")
    return output


@torch.no_grad()
def evaluate_state(model, batch: dict, units: list[dict], *, run_parity: bool):
    actions = batch["action"].to(device=model.action_in_proj.weight.device, dtype=model.action_in_proj.weight.dtype)
    actions = actions if actions.ndim == 3 else actions.unsqueeze(0)
    action_is_pad = batch["action_is_pad"].to(device=actions.device, dtype=torch.bool)
    if actions.shape[0] != 1 or len(units) != 3:
        raise ValueError("expected one action chunk and three noise units")
    noises = []
    for unit in units:
        generator = torch.Generator(device=actions.device).manual_seed(unit["noise_seed"])
        noises.append(
            torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
        )
    noise = torch.cat(noises, dim=0)
    actions3 = repeated(actions, 3)
    prefix, prefix_pad, prefix_att = model.embed_prefix(
        batch["images"],
        batch["image_masks"],
        batch["lang_tokens"],
        batch["lang_masks"],
        state=batch["state"],
    )
    scales = torch.ones(
        3, model.vlm_with_expert.num_expert_layers, device=actions.device, dtype=actions.dtype
    )
    scales[:, -2:] = 0.5
    valid = (~action_is_pad).expand(3, -1)
    valid7 = valid.unsqueeze(-1).expand(-1, -1, 7)
    rows = []
    parity_max_abs = 0.0
    deterministic_max_abs = 0.0
    for flow_step, tau_value in enumerate(FLOW_TIMES):
        timestep = torch.full((3,), tau_value, device=actions.device, dtype=actions.dtype)
        x_t, target = reconstruct_training_pair(model, actions3, noise, timestep)
        strong = velocity_from_embeddings(
            model,
            repeated(prefix, 3),
            repeated(prefix_pad, 3),
            repeated(prefix_att, 3),
            x_t,
            timestep,
        )
        weak = velocity_from_embeddings(
            model,
            repeated(prefix, 3),
            repeated(prefix_pad, 3),
            repeated(prefix_att, 3),
            x_t,
            timestep,
            expert_residual_scales=scales,
        )
        if run_parity:
            parity = velocity_from_embeddings(
                model,
                repeated(prefix, 3),
                repeated(prefix_pad, 3),
                repeated(prefix_att, 3),
                x_t,
                timestep,
                expert_residual_scales=torch.ones_like(scales),
            )
            parity_max_abs = max(parity_max_abs, float((parity - strong).abs().max()))
            if flow_step == 0:
                repeat_weak = velocity_from_embeddings(
                    model,
                    repeated(prefix, 3),
                    repeated(prefix_pad, 3),
                    repeated(prefix_att, 3),
                    x_t,
                    timestep,
                    expert_residual_scales=scales,
                )
                deterministic_max_abs = float((repeat_weak - weak).abs().max())
        for noise_index, unit in enumerate(units):
            metrics = point_metrics(
                strong[noise_index, :, :7],
                weak[noise_index, :, :7],
                target[noise_index, :, :7],
                valid7[noise_index],
            )
            features = local_features(
                strong[noise_index, :, :7], weak[noise_index, :, :7], valid7[noise_index]
            )
            row = {
                "point_id": f"{unit['unit_id']}__step{flow_step:02d}",
                "state_id": unit["state_id"],
                "unit_id": unit["unit_id"],
                "task_id": unit["task_id"],
                "source_task_index": unit["source_task_index"],
                "episode_id": unit["episode_id"],
                "noise_ordinal": unit["noise_ordinal"],
                "noise_seed": unit["noise_seed"],
                "flow_step": flow_step,
                "timestep": tau_value,
                "branch": PRIMARY_BRANCH,
                "prefix_sha256": tensor_sha256(prefix),
                "action_sha256": tensor_sha256(actions),
                "noise_sha256": tensor_sha256(noise[noise_index]),
                "x_t_sha256": tensor_sha256(x_t[noise_index]),
                "target_sha256": tensor_sha256(target[noise_index]),
                **features,
                **metrics,
            }
            missing = set(LOCAL_FEATURES) - set(row)
            if missing:
                raise RuntimeError(f"missing frozen features: {missing}")
            rows.append(row)
    return rows, {
        "rows": len(rows),
        "finite": all(
            np.isfinite(value)
            for row in rows
            for value in row.values()
            if isinstance(value, float)
        ),
        "strong_ones_parity_max_abs": parity_max_abs if run_parity else None,
        "deterministic_rerun_max_abs": deterministic_max_abs if run_parity else None,
        "prefix_sha256": tensor_sha256(prefix),
        "action_sha256": tensor_sha256(actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    if protocol["stage"] != "independent_offline_libero_object_confirmation_no_rollout":
        raise RuntimeError("unexpected protocol stage")
    if not (artifact / "status/unit_tests.pass").exists():
        raise RuntimeError("unit tests must pass before capture")
    units = [json.loads(line) for line in (artifact / "unit_manifest.jsonl").read_text().splitlines()]
    states = [json.loads(line) for line in (artifact / "state_manifest.jsonl").read_text().splitlines()]

    paths = Paths(
        workspace=workspace,
        dataset=Path(protocol["data"]["path"]),
        checkpoint=workspace
        / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots"
        / CHECKPOINT_REVISION,
        download_manifest=workspace
        / "counterfactual-flow-vla/pi05_svcpd_p0/verified_calibration_download_manifest.json",
    )
    policy = load_policy(paths)
    policy.to(device="cuda", dtype=torch.float32)
    preprocessor = load_preprocessor(paths)
    store = SnapshotRows(paths.dataset)
    model = policy.model
    state_integrity = []
    completed = 0
    for state in states:
        group = [unit for unit in units if unit["state_id"] == state["state_id"]]
        output_path = artifact / "raw" / f"{state['state_id']}.json"
        if output_path.exists():
            existing = json.loads(output_path.read_text())
            if len(existing["rows"]) != 30:
                raise RuntimeError(f"partial state file: {output_path}")
            state_integrity.append(existing["integrity"])
            completed += 1
            continue
        processed = processed_state(store, preprocessor, state)
        processed = {
            key: value.to("cuda") if isinstance(value, torch.Tensor) else value
            for key, value in processed.items()
        }
        images, image_masks = policy.prepare_images(processed)
        batch = {
            "images": images,
            "image_masks": image_masks,
            "state": policy.prepare_state(processed),
            "action": policy.prepare_action(processed),
            "action_is_pad": processed["action_is_pad"],
            "lang_tokens": processed["observation.language.tokens"],
            "lang_masks": processed["observation.language.attention_mask"],
        }
        rows, integrity = evaluate_state(model, batch, group, run_parity=state["episode_ordinal"] == 0)
        integrity.update(
            {
                "state_id": state["state_id"],
                "task_id": state["task_id"],
                "source_task_index": state["source_task_index"],
                "episode_id": state["episode_id"],
                "frame_id": state["frame_id"],
                "camera1_sha256": tensor_sha256(processed["observation.images.camera1"]),
                "camera2_sha256": tensor_sha256(processed["observation.images.camera2"]),
                "state_sha256": tensor_sha256(batch["state"]),
                "lang_tokens_sha256": tensor_sha256(batch["lang_tokens"]),
            }
        )
        atomic_json(output_path, {"integrity": integrity, "rows": rows})
        state_integrity.append(integrity)
        completed += 1
        print(
            json.dumps(
                {"states_complete": completed, "planned": len(states), "state_id": state["state_id"]}
            ),
            flush=True,
        )

    raw_files = sorted((artifact / "raw").glob("*.json"))
    all_rows = []
    for path in raw_files:
        all_rows.extend(json.loads(path.read_text())["rows"])
    frame = pd.DataFrame(all_rows)
    frame.to_parquet(artifact / "object_point_metrics.parquet", index=False)
    capture = {
        "states": len(raw_files),
        "units": len(raw_files) * 3,
        "point_rows": len(frame),
        "expected_point_rows": protocol["scope"]["point_rows"],
        "unique_point_ids": int(frame.point_id.nunique()),
        "tasks": int(frame.task_id.nunique()),
        "finite": bool(frame.select_dtypes(include=[np.number]).apply(np.isfinite).all().all()),
        "all_state_rows_complete": all(value["rows"] == 30 for value in state_integrity),
        "strong_ones_parity_max_abs": max(
            value["strong_ones_parity_max_abs"] or 0.0 for value in state_integrity
        ),
        "deterministic_rerun_max_abs": max(
            value["deterministic_rerun_max_abs"] or 0.0 for value in state_integrity
        ),
    }
    (artifact / "capture_summary.json").write_text(
        json.dumps(capture, indent=2, sort_keys=True) + "\n"
    )
    if (
        capture["states"] == protocol["scope"]["states"]
        and capture["point_rows"] == protocol["scope"]["point_rows"]
        and capture["unique_point_ids"] == protocol["scope"]["point_rows"]
        and capture["tasks"] == 10
        and capture["finite"]
        and capture["all_state_rows_complete"]
    ):
        (artifact / "status/capture.complete").write_text("object capture complete\n")
    print(json.dumps(capture, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
