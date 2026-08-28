#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from lerobot.envs.factory import make_env_pre_post_processors
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_closed_loop.runtime import env_config
from research.coreact_expert_direction.direction_audit import prepare_demo_state, rewrite_demo_xml
from research.coreact_quality_negative_branch.quality_branch import tensor_sha256
from research.coreact_region.segmented_runtime import make_segmented_env
from research.coreact_trained_weak.runtime import load_locked_pair
from research.coreact_flow_timestep_compatibility.metrics import exact_training_pair, point_metrics
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


FLOW_TIMES = tuple(round(1.0 - 0.1 * i, 1) for i in range(10))


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def joined_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


@torch.no_grad()
def velocity_from_embeddings(model, prefix, prefix_pad, prefix_att, x_t, timestep):
    suffix, suffix_pad, suffix_att = model.embed_suffix(x_t, timestep)
    pads = torch.cat([prefix_pad, suffix_pad], dim=1)
    atts = torch.cat([prefix_att, suffix_att], dim=1)
    mask = make_att_2d_masks(pads, atts)
    positions = torch.cumsum(pads, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=mask,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        fill_kv_cache=False,
    )
    return model.action_out_proj(suffix_out[:, -model.config.chunk_size :].float())


@torch.no_grad()
def evaluate_state(strong_model, weak_model, batch, state_row):
    actions = batch["actions"]
    noises = torch.cat([
        torch.randn(actions.shape, generator=torch.Generator(device=actions.device).manual_seed(seed), device=actions.device, dtype=actions.dtype)
        for seed in state_row["noise_seeds"]
    ])
    actions3 = actions.expand(3, -1, -1)
    strong_prefix, strong_pad, strong_att = strong_model.embed_prefix(
        batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], state=batch["state"]
    )
    weak_prefix, weak_pad, weak_att = weak_model.embed_prefix(
        batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], state=batch["state"]
    )
    if not torch.equal(strong_pad, weak_pad) or not torch.equal(strong_att, weak_att):
        raise RuntimeError("Strong/Weak masks differ")
    common_input_hash = joined_hash([
        *batch["images"], *batch["image_masks"], batch["state"], batch["lang_tokens"], batch["lang_masks"], actions
    ])
    valid_steps = ~batch["action_is_pad"][0]
    rows = []
    max_target_error = 0.0
    max_xt_error = 0.0
    max_direction_error = 0.0
    for flow_step, tau_value in enumerate(FLOW_TIMES):
        timestep = torch.full((3,), tau_value, device=actions.device, dtype=actions.dtype)
        manual_x = timestep[:, None, None] * noises + (1.0 - timestep[:, None, None]) * actions3
        manual_target = noises - actions3
        helper_x, helper_target = exact_training_pair(actions3, noises, timestep)
        max_xt_error = max(max_xt_error, float((manual_x - helper_x).abs().max()))
        max_target_error = max(max_target_error, float((manual_target - helper_target).abs().max()))
        strong = velocity_from_embeddings(strong_model, strong_prefix.expand(3, -1, -1), strong_pad.expand(3, -1), strong_att.expand(3, -1), helper_x, timestep)
        weak = velocity_from_embeddings(weak_model, weak_prefix.expand(3, -1, -1), weak_pad.expand(3, -1), weak_att.expand(3, -1), helper_x, timestep)
        for noise_ordinal, noise_seed in enumerate(state_row["noise_seeds"]):
            metrics, applied = point_metrics(
                strong[noise_ordinal, :, :7], weak[noise_ordinal, :, :7], helper_target[noise_ordinal, :, :7], valid_steps
            )
            recomputed = strong[noise_ordinal, :, :7] - weak[noise_ordinal, :, :7]
            max_direction_error = max(max_direction_error, float((recomputed - (strong[noise_ordinal, :, :7] - weak[noise_ordinal, :, :7])).abs().max()))
            rows.append({
                "point_id": f"{state_row['state_id']}__noise{noise_ordinal}__flow{flow_step:02d}",
                "state_id": state_row["state_id"],
                "split": state_row["split"],
                "task_id": state_row["task_id"],
                "phase": state_row["phase"],
                "progress": state_row["target_progress"],
                "noise_ordinal": noise_ordinal,
                "noise_seed": noise_seed,
                "flow_step": flow_step,
                "timestep": tau_value,
                "noise_level_order": "highest" if flow_step == 0 else ("lowest" if flow_step == 9 else "intermediate"),
                "common_input_sha256_strong": common_input_hash,
                "common_input_sha256_weak": common_input_hash,
                "expert_chunk_sha256": tensor_sha256(actions[0]),
                "epsilon_sha256": tensor_sha256(noises[noise_ordinal]),
                "x_t_sha256_strong": tensor_sha256(helper_x[noise_ordinal]),
                "x_t_sha256_weak": tensor_sha256(helper_x[noise_ordinal]),
                "target_sha256": tensor_sha256(helper_target[noise_ordinal]),
                "applied_correction_sha256": tensor_sha256(applied),
                "valid_action_steps": int(valid_steps.sum()),
                **metrics,
            })
    return rows, {
        "common_input_hash_equal": True,
        "strong_prefix_shape": list(strong_prefix.shape),
        "weak_prefix_shape": list(weak_prefix.shape),
        "prefix_masks_equal": True,
        "manual_vs_helper_x_t_max_abs": max_xt_error,
        "manual_vs_helper_target_max_abs": max_target_error,
        "recompute_direction_max_abs": max_direction_error,
        "finite": all(np.isfinite(value) for row in rows for value in row.values() if isinstance(value, float)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "confirmation"), required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-states-per-task", type=int)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if args.split == "confirmation":
        selection_lock = artifact / "selected_segment.lock.json"
        if not selection_lock.exists():
            raise RuntimeError("confirmation is sealed until Selection passes and freezes a segment")
        if json.loads(selection_lock.read_text()).get("selection_decision") != "PASS":
            raise RuntimeError("selection lock does not authorize confirmation")
    manifest = [json.loads(line) for line in (artifact / "state_manifest.jsonl").read_text().splitlines()]
    states = [row for row in manifest if row["split"] == args.split and row["task_id"] in args.task_ids]
    if args.max_states_per_task is not None:
        states = [row for task in args.task_ids for row in [x for x in states if x["task_id"] == task][:args.max_states_per_task]]
    output_dir = artifact / args.split
    os.chdir(workspace / "LIBERO")
    source_artifact = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
    _, strong, weak = load_locked_pair(source_artifact)
    config, strong_policy, preprocessor, _ = strong
    _, weak_policy, _, _ = weak
    for task_id in args.task_ids:
        task_states = [row for row in states if row["task_id"] == task_id]
        if not task_states:
            continue
        env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_config("libero_spatial", task_id), policy_cfg=config)
        env = make_segmented_env("libero_spatial", task_id)
        try:
            with h5py.File(task_states[0]["demo_path"], "r") as handle:
                for index, row in enumerate(task_states, 1):
                    output = output_dir / f"{row['state_id']}.json"
                    if output.exists():
                        continue
                    episode = handle["data"][row["demo_id"]]
                    states_np, actions_np = np.asarray(episode["states"]), np.asarray(episode["actions"])
                    xml = episode.attrs["model_file"]
                    xml = xml.decode() if isinstance(xml, bytes) else xml
                    env._env.reset()
                    env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml, workspace), {}, demo_generation=False))
                    env._env.env.sim.reset()
                    raw = env._env.regenerate_obs_from_state(states_np[row["resolved_frame"]])
                    batch = prepare_demo_state(strong_policy, preprocessor, env_preprocessor, env, raw, row["language"], actions_np, row["resolved_frame"])
                    rows, integrity = evaluate_state(strong_policy.model, weak_policy.model, batch, row)
                    atomic_json(output, {"state": row, "preprocessing_sha256": batch["preprocessing_sha256"], "integrity": integrity, "rows": rows})
                    print(json.dumps({"split": args.split, "task": task_id, "state": index, "of": len(task_states), "points": len(rows)}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
