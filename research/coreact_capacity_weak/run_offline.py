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
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_closed_loop.runtime import env_config
from research.coreact_expert_direction.direction_audit import prepare_demo_state, rewrite_demo_xml
from research.coreact_quality_negative_branch.quality_branch import tensor_sha256
from research.coreact_region.segmented_runtime import make_segmented_env
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_capacity_weak.metrics import exact_training_pair, point_metrics


TIMES = tuple(round(1.0 - 0.1 * index, 1) for index in range(10))


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def joined_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


@torch.no_grad()
def velocity(model, prefix, pad, att, x_t, timestep):
    suffix, suffix_pad, suffix_att = model.embed_suffix(x_t, timestep)
    pads, atts = torch.cat([pad, suffix_pad], 1), torch.cat([att, suffix_att], 1)
    mask = make_att_2d_masks(pads, atts)
    positions = torch.cumsum(pads, dim=1) - 1
    (_, output), _ = model.vlm_with_expert.forward(attention_mask=mask, position_ids=positions, past_key_values=None, inputs_embeds=[prefix, suffix], use_cache=False, fill_kv_cache=False)
    return model.action_out_proj(output[:, -model.config.chunk_size :].float())


@torch.no_grad()
def evaluate(models, batch, state):
    actions = batch["actions"]
    noises = torch.cat([torch.randn(actions.shape, generator=torch.Generator(device=actions.device).manual_seed(seed), device=actions.device, dtype=actions.dtype) for seed in state["noise_seeds"]])
    actions3 = actions.expand(3, -1, -1)
    prefixes = {}
    for name, model in models.items():
        prefixes[name] = model.embed_prefix(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], state=batch["state"])
    if any(not torch.equal(prefixes["strong"][index], prefixes[name][index]) for name in models if name != "strong" for index in (1, 2)):
        raise RuntimeError("prefix masks differ")
    common_hash = joined_hash([*batch["images"], *batch["image_masks"], batch["state"], batch["lang_tokens"], batch["lang_masks"], actions])
    valid = ~batch["action_is_pad"][0]
    rows = []
    target_errors = []
    for flow_step, tau in enumerate(TIMES):
        timestep = torch.full((3,), tau, device=actions.device, dtype=actions.dtype)
        x_t, target = exact_training_pair(actions3, noises, timestep)
        target_errors.append(float((target - (noises - actions3)).abs().max()))
        predictions = {name: velocity(model, prefix.expand(3, -1, -1), pad.expand(3, -1), att.expand(3, -1), x_t, timestep) for name, model in models.items() for prefix, pad, att in [prefixes[name]]}
        for candidate in (name for name in models if name != "strong"):
            for noise_ordinal, noise_seed in enumerate(state["noise_seeds"]):
                rows.append({
                    "point_id": f"{state['state_id']}__{candidate}__noise{noise_ordinal}__flow{flow_step:02d}", "state_id": state["state_id"], "split": state["split"], "task_id": state["task_id"], "phase": state["phase"], "candidate": candidate,
                    "noise_ordinal": noise_ordinal, "noise_seed": noise_seed, "flow_step": flow_step, "timestep": tau,
                    "common_input_sha256_strong": common_hash, "common_input_sha256_weak": common_hash, "expert_chunk_sha256": tensor_sha256(actions[0]), "epsilon_sha256": tensor_sha256(noises[noise_ordinal]), "x_t_sha256_strong": tensor_sha256(x_t[noise_ordinal]), "x_t_sha256_weak": tensor_sha256(x_t[noise_ordinal]), "target_sha256": tensor_sha256(target[noise_ordinal]),
                    **point_metrics(predictions["strong"][noise_ordinal, :, :7], predictions[candidate][noise_ordinal, :, :7], target[noise_ordinal, :, :7], valid),
                })
    return rows, {"common_inputs_equal": True, "prefix_masks_equal": True, "target_max_abs_error": max(target_errors), "finite": all(np.isfinite(value) for row in rows for value in row.values() if isinstance(value, float))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "confirmation"), required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-states-per-task", type=int)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    candidates = ["weak_a_8l", "weak_b_4l"]
    if args.split == "confirmation":
        lock = artifact / "selected_weak.lock.json"
        if not lock.exists():
            raise RuntimeError("Confirmation is sealed")
        candidates = [json.loads(lock.read_text())["selected_weak"]]
    manifest = [json.loads(line) for line in (artifact / "state_manifest.jsonl").read_text().splitlines()]
    states = [row for row in manifest if row["split"] == args.split and row["task_id"] in args.task_ids]
    if args.max_states_per_task is not None:
        states = [row for task in args.task_ids for row in [item for item in states if item["task_id"] == task][:args.max_states_per_task]]
    source = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
    strong_path = source / "training_run/trajectory/checkpoints/015000/pretrained_model"
    paths = {"strong": strong_path, **{name: artifact / "training" / name / "checkpoints/015000/pretrained_model" for name in candidates}}
    os.chdir(workspace / "LIBERO")
    loaded = {name: load_policy(path) for name, path in paths.items()}
    config, strong_policy, preprocessor, _ = loaded["strong"]
    models = {name: item[1].model for name, item in loaded.items()}
    output_dir = artifact / args.split
    for task_id in args.task_ids:
        task_states = [row for row in states if row["task_id"] == task_id]
        if not task_states:
            continue
        env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_config("libero_spatial", task_id), policy_cfg=config)
        env = make_segmented_env("libero_spatial", task_id)
        try:
            with h5py.File(task_states[0]["demo_path"], "r") as handle:
                for index, state in enumerate(task_states, 1):
                    output = output_dir / f"{state['state_id']}.json"
                    if output.exists():
                        continue
                    episode = handle["data"][state["demo_id"]]
                    states_np, actions_np = np.asarray(episode["states"]), np.asarray(episode["actions"])
                    xml = episode.attrs["model_file"]
                    xml = xml.decode() if isinstance(xml, bytes) else xml
                    env._env.reset(); env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml, workspace), {}, demo_generation=False)); env._env.env.sim.reset()
                    raw = env._env.regenerate_obs_from_state(states_np[state["resolved_frame"]])
                    batch = prepare_demo_state(strong_policy, preprocessor, env_preprocessor, env, raw, state["language"], actions_np, state["resolved_frame"])
                    rows, integrity = evaluate(models, batch, state)
                    atomic_json(output, {"state": state, "integrity": integrity, "preprocessing_sha256": batch["preprocessing_sha256"], "rows": rows})
                    print(json.dumps({"split": args.split, "task": task_id, "state": index, "of": len(task_states), "points": len(rows)}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
