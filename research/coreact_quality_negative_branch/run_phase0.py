#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORKSPACE / "LIBERO"), str(WORKSPACE / "lerobot/src"), str(WORKSPACE)]

from lerobot.envs.factory import make_env_pre_post_processors
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors
from research.coreact_expert_direction.direction_audit import (
    action_chunk,
    array_sha,
    prepare_demo_state,
    rewrite_demo_xml,
)
from research.coreact_quality_negative_branch.quality_branch import (
    BRANCHES,
    FLOW_TIMES,
    branch_scale_matrix,
    point_metrics,
    reconstruct_training_pair,
    tensor_sha256,
    velocity_from_embeddings,
)
from research.coreact_region.segmented_runtime import make_segmented_env


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def repeated(value: torch.Tensor, count: int) -> torch.Tensor:
    return value.expand(count, *value.shape[1:])


@torch.no_grad()
def evaluate_state(model, batch: dict, units: list[dict], *, run_parity: bool) -> tuple[list[dict], dict]:
    actions = batch["actions"]
    if actions.shape[0] != 1 or len(units) != 3:
        raise ValueError("each expert state requires one action chunk and three noise units")
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
    num_expert_layers = model.vlm_with_expert.num_expert_layers
    scales = branch_scale_matrix(
        num_expert_layers, 3, device=actions.device, dtype=actions.dtype
    )
    rows = []
    parity_max_abs = 0.0
    deterministic_max_abs = 0.0
    valid = (~batch["action_is_pad"]).expand(3, -1)
    valid7 = valid.unsqueeze(-1).expand(-1, -1, 7)

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
        x_weak = torch.cat([x_t] * len(BRANCHES), dim=0)
        timestep_weak = timestep.repeat(len(BRANCHES))
        weak = velocity_from_embeddings(
            model,
            repeated(prefix, len(BRANCHES) * 3),
            repeated(prefix_pad, len(BRANCHES) * 3),
            repeated(prefix_att, len(BRANCHES) * 3),
            x_weak,
            timestep_weak,
            expert_residual_scales=scales,
        ).reshape(len(BRANCHES), 3, *strong.shape[1:])

        if run_parity:
            parity = velocity_from_embeddings(
                model,
                repeated(prefix, 3),
                repeated(prefix_pad, 3),
                repeated(prefix_att, 3),
                x_t,
                timestep,
                expert_residual_scales=torch.ones(
                    3, num_expert_layers, device=actions.device, dtype=actions.dtype
                ),
            )
            parity_max_abs = max(parity_max_abs, float((parity - strong).abs().max()))
            if flow_step == 0:
                weak_repeat = velocity_from_embeddings(
                    model,
                    repeated(prefix, len(BRANCHES) * 3),
                    repeated(prefix_pad, len(BRANCHES) * 3),
                    repeated(prefix_att, len(BRANCHES) * 3),
                    x_weak,
                    timestep_weak,
                    expert_residual_scales=scales,
                ).reshape_as(weak)
                deterministic_max_abs = float((weak_repeat - weak).abs().max())

        for branch_index, branch in enumerate(BRANCHES):
            for noise_index, unit in enumerate(units):
                metrics = point_metrics(
                    strong[noise_index, :, :7],
                    weak[branch_index, noise_index, :, :7],
                    target[noise_index, :, :7],
                    valid7[noise_index],
                )
                rows.append(
                    {
                        "point_id": f"{unit['unit_id']}__step{flow_step:02d}__{branch.name}",
                        "state_id": unit["state_id"],
                        "unit_id": unit["unit_id"],
                        "task_id": unit["task_id"],
                        "demo_ordinal": unit["demo_ordinal"],
                        "noise_ordinal": unit["noise_ordinal"],
                        "noise_seed": unit["noise_seed"],
                        "flow_step": flow_step,
                        "timestep": tau_value,
                        "branch": branch.name,
                        "affected_expert_layers": branch.affected_layers,
                        "residual_scale": branch.residual_scale,
                        "valid_action_steps": int(valid[noise_index].sum()),
                        "prefix_sha256": tensor_sha256(prefix),
                        "action_sha256": tensor_sha256(actions),
                        "noise_sha256": tensor_sha256(noise[noise_index]),
                        "x_t_sha256": tensor_sha256(x_t[noise_index]),
                        "target_sha256": tensor_sha256(target[noise_index]),
                        **metrics,
                    }
                )
    integrity = {
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
        "noise_sha256": [tensor_sha256(value) for value in noise],
        "expert_layer_count": num_expert_layers,
    }
    return rows, integrity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    units = [json.loads(line) for line in (artifact / "unit_manifest.jsonl").read_text().splitlines()]
    if protocol["stage"] != "offline_phase0_negative_branch_qualification_no_rollout":
        raise RuntimeError("unexpected protocol stage")
    target_test = artifact / "status/target_reconstruction.pass"
    if not target_test.exists():
        raise RuntimeError("target reconstruction unit test must pass before capture")

    os.chdir(workspace / "LIBERO")
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    model = policy.model
    runtime_architecture = {
        "vlm_layers": model.vlm_with_expert.num_vlm_layers,
        "expert_layers": model.vlm_with_expert.num_expert_layers,
        "expert_hidden_size": model.vlm_with_expert.expert_hidden_size,
        "attention_mode": model.vlm_with_expert.attention_mode,
        "self_attn_every_n_layers": model.vlm_with_expert.self_attn_every_n_layers,
        "action_output_head": str(model.action_out_proj),
        "frozen_eval": not policy.training and not any(p.requires_grad for p in policy.parameters()),
    }
    if (
        runtime_architecture["vlm_layers"] != 16
        or runtime_architecture["expert_layers"] != 16
        or not runtime_architecture["frozen_eval"]
    ):
        raise RuntimeError(f"architecture/freeze gate failed: {runtime_architecture}")
    (artifact / "architecture_runtime.json").write_text(
        json.dumps(runtime_architecture, indent=2, sort_keys=True) + "\n"
    )

    state_integrity = []
    completed = 0
    for task_id in protocol["scope"]["tasks"]:
        task_units = [unit for unit in units if unit["task_id"] == task_id]
        env_preprocessor, _ = make_env_pre_post_processors(
            env_cfg=env_config("libero_spatial", task_id), policy_cfg=config
        )
        env = make_segmented_env("libero_spatial", task_id)
        try:
            with h5py.File(task_units[0]["demo_path"], "r") as handle:
                for ordinal in range(50):
                    group = [unit for unit in task_units if unit["demo_ordinal"] == ordinal]
                    output_path = artifact / "raw" / f"{group[0]['state_id']}.json"
                    if output_path.exists():
                        existing = json.loads(output_path.read_text())
                        if len(existing["rows"]) != 120:
                            raise RuntimeError(f"partial state file: {output_path}")
                        state_integrity.append(existing["integrity"])
                        completed += 1
                        continue
                    row = group[0]
                    episode = handle["data"][row["demo_id"]]
                    states = np.asarray(episode["states"])
                    actions = np.asarray(episode["actions"])
                    xml = episode.attrs["model_file"]
                    xml = xml.decode() if isinstance(xml, bytes) else xml
                    env._env.reset()
                    env._env.reset_from_xml_string(
                        postprocess_model_xml(rewrite_demo_xml(xml, workspace), {}, demo_generation=False)
                    )
                    env._env.env.sim.reset()
                    raw = env._env.regenerate_obs_from_state(states[row["resolved_frame"]])
                    batch = prepare_demo_state(
                        policy,
                        preprocessor,
                        env_preprocessor,
                        env,
                        raw,
                        row["language"],
                        actions,
                        row["resolved_frame"],
                    )
                    rows, integrity = evaluate_state(
                        model,
                        batch,
                        group,
                        run_parity=ordinal == 0,
                    )
                    integrity.update(
                        {
                            "state_id": row["state_id"],
                            "task_id": task_id,
                            "sim_state_sha256": array_sha(states[row["resolved_frame"]]),
                            "camera1_sha256": array_sha(raw["agentview_image"]),
                            "camera2_sha256": array_sha(raw["robot0_eye_in_hand_image"]),
                            "preprocessing_sha256": batch["preprocessing_sha256"],
                        }
                    )
                    atomic_json(output_path, {"integrity": integrity, "rows": rows})
                    state_integrity.append(integrity)
                    completed += 1
                    print(
                        json.dumps(
                            {"states_complete": completed, "planned": 100, "state_id": row["state_id"]}
                        ),
                        flush=True,
                    )
        finally:
            env.close()

    raw_files = sorted((artifact / "raw").glob("*.json"))
    all_rows = []
    for path in raw_files:
        all_rows.extend(json.loads(path.read_text())["rows"])
    frame = pd.DataFrame(all_rows)
    frame.to_parquet(artifact / "point_level_metrics.parquet", index=False)
    capture = {
        "states": len(raw_files),
        "units": len(raw_files) * 3,
        "point_rows": len(frame),
        "expected_point_rows": 12000,
        "finite": bool(frame.select_dtypes(include=[np.number]).apply(np.isfinite).all().all()),
        "unique_point_ids": int(frame["point_id"].nunique()),
        "strong_ones_parity_max_abs": max(
            value["strong_ones_parity_max_abs"] or 0.0 for value in state_integrity
        ),
        "deterministic_rerun_max_abs": max(
            value["deterministic_rerun_max_abs"] or 0.0 for value in state_integrity
        ),
        "all_state_rows_complete": all(value["rows"] == 120 for value in state_integrity),
        "expert_layer_count": runtime_architecture["expert_layers"],
        "branch_scale_contract": {
            branch.name: {
                "affected_layers": branch.affected_layers,
                "residual_scale": branch.residual_scale,
            }
            for branch in BRANCHES
        },
    }
    (artifact / "capture_summary.json").write_text(
        json.dumps(capture, indent=2, sort_keys=True) + "\n"
    )
    if len(frame) == 12000 and capture["finite"] and capture["all_state_rows_complete"]:
        (artifact / "status/capture.complete").write_text("100 states, 300 units, 12000 points\n")
    print(json.dumps(capture, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
