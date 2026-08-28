#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors
from libero.libero.envs.utils import postprocess_model_xml

from research.coreact_capacity_weak.metrics import exact_training_pair
from research.coreact_capacity_weak.run_offline import velocity
from research.coreact_closed_loop.runtime import env_config
from research.coreact_expert_direction.direction_audit import action_chunk, prepare_demo_state, rewrite_demo_xml
from research.coreact_region.segmented_runtime import make_segmented_env
from research.coreact_trained_weak.runtime import load_policy


TIMES = tuple(round(1.0 - 0.1 * index, 1) for index in range(10))
EPS = 1e-12


def normalize_candidates(preprocessor, raw: np.ndarray, device: torch.device, dtype: torch.dtype, target_dim: int) -> torch.Tensor:
    normalizer = preprocessor.steps[-1]
    tensor = torch.as_tensor(raw, device=device, dtype=dtype)
    normalized = normalizer._normalize_action(tensor, inverse=False)
    return torch.nn.functional.pad(normalized, (0, target_dim - normalized.shape[-1]))


def core_metrics(strong, weak, single_target, candidate_actions, x_t, tau, valid):
    flat = valid[:, None].expand_as(strong)
    s, w, single = strong[flat].float(), weak[flat].float(), single_target[flat].float()
    d = s - w
    candidate_flat = candidate_actions[:, valid, :7].float().reshape(len(candidate_actions), -1)
    x_flat = x_t[valid, :7].float().reshape(-1)
    targets = (x_flat[None, :] - candidate_flat) / tau
    log_likelihood = -torch.sum((x_flat[None, :] - (1.0 - tau) * candidate_flat) ** 2, dim=1) / (2.0 * tau * tau)
    weights = torch.softmax(log_likelihood - torch.max(log_likelihood), dim=0)
    marginal = torch.sum(weights[:, None] * targets, dim=0)
    g_single = single - s
    g_marginal = marginal - s
    single_dot = torch.dot(d, g_single)
    marginal_dot = torch.dot(d, g_marginal)
    target_distances = torch.sum((targets - s[None, :]) ** 2, dim=1)
    nearest = targets[torch.argmin(target_distances)]
    nearest_dot = torch.dot(d, nearest - s)
    raw = s + 0.5 * d
    full_direction = strong - weak
    raw_correction = 0.5 * full_direction
    clip_scale = torch.clamp(0.25 * torch.linalg.vector_norm(strong) / (torch.linalg.vector_norm(raw_correction) + EPS), max=1.0)
    applied = s + 0.5 * d * clip_scale
    d_strong = torch.min(target_distances)
    d_raw = torch.min(torch.sum((targets - raw[None, :]) ** 2, dim=1))
    d_applied = torch.min(torch.sum((targets - applied[None, :]) ** 2, dim=1))
    return {
        "single_g_positive": bool(single_dot > 0),
        "marginal_g_positive": bool(marginal_dot > 0),
        "marginal_cosine": float(marginal_dot / (torch.linalg.vector_norm(d) * torch.linalg.vector_norm(g_marginal) + EPS)),
        "nearest_g_positive": bool(nearest_dot > 0),
        "nearest_cosine": float(nearest_dot / (torch.linalg.vector_norm(d) * torch.linalg.vector_norm(nearest - s) + EPS)),
        "delta_d_raw": float(d_strong - d_raw),
        "delta_d_raw_positive": bool(d_raw < d_strong),
        "delta_d_applied": float(d_strong - d_applied),
        "delta_d_applied_positive": bool(d_applied < d_strong),
        "posterior_ess": float(1.0 / torch.sum(weights ** 2)),
        "posterior_max_weight": float(torch.max(weights)),
        "nearest_index": int(torch.argmin(target_distances)),
        "clip_scale": float(clip_scale),
    }


@torch.no_grad()
def evaluate(models, preprocessor, batch, raw_candidates, raw_anchor, state):
    actions = batch["actions"]
    candidates = normalize_candidates(preprocessor, raw_candidates, actions.device, actions.dtype, actions.shape[-1])
    normalized_anchor = normalize_candidates(preprocessor, raw_anchor[None], actions.device, actions.dtype, actions.shape[-1])
    normalization_max_abs = float((normalized_anchor - actions).abs().max())
    if normalization_max_abs >= 1e-6:
        raise RuntimeError(f"candidate normalization differs from training pipeline: {normalization_max_abs}")
    noises = torch.cat([torch.randn(actions.shape, generator=torch.Generator(device=actions.device).manual_seed(seed), device=actions.device, dtype=actions.dtype) for seed in state["noise_seeds"]])
    actions3 = actions.expand(3, -1, -1)
    valid = ~batch["action_is_pad"][0]
    prefixes = {name: model.embed_prefix(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], state=batch["state"]) for name, model in models.items()}
    rows = []
    for flow_step, tau in enumerate(TIMES):
        timestep = torch.full((3,), tau, device=actions.device, dtype=actions.dtype)
        x_t, target = exact_training_pair(actions3, noises, timestep)
        predictions = {name: velocity(model, prefix.expand(3, -1, -1), pad.expand(3, -1), att.expand(3, -1), x_t, timestep)[:, :, :7] for name, model in models.items() for prefix, pad, att in [prefixes[name]]}
        for candidate in (name for name in models if name != "strong"):
            for noise_ordinal in range(3):
                metrics = core_metrics(predictions["strong"][noise_ordinal], predictions[candidate][noise_ordinal], target[noise_ordinal, :, :7], candidates, x_t[noise_ordinal, :, :7], tau, valid)
                rows.append({"candidate": candidate, "flow_step": flow_step, "timestep": tau, "noise_ordinal": noise_ordinal, **metrics})
    return rows, {"finite": all(np.isfinite(value) for row in rows for value in row.values() if isinstance(value, float)), "rows": len(rows), "candidate_shape": list(candidates.shape), "normalization_max_abs": normalization_max_abs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-states-per-task", type=int)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    states = [json.loads(line) for line in (artifact / "state_manifest.jsonl").read_text().splitlines() if json.loads(line)["task_id"] in args.task_ids]
    if args.max_states_per_task is not None:
        states = [row for task in args.task_ids for row in [x for x in states if x["task_id"] == task][:args.max_states_per_task]]
    capacity = workspace / "artifacts/coreact_capacity_weak_v1_20260815_202000"
    source = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
    paths = {"strong": source / "training_run/trajectory/checkpoints/015000/pretrained_model", "weak_a_8l": capacity / "training/weak_a_8l/checkpoints/015000/pretrained_model", "weak_b_4l": capacity / "training/weak_b_4l/checkpoints/015000/pretrained_model"}
    os.chdir(workspace / "LIBERO")
    loaded = {name: load_policy(path) for name, path in paths.items()}
    config, strong_policy, preprocessor, _ = loaded["strong"]
    models = {name: item[1].model for name, item in loaded.items()}
    for task_id in args.task_ids:
        task_states = [row for row in states if row["task_id"] == task_id]
        if not task_states:
            continue
        env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_config("libero_spatial", task_id), policy_cfg=config)
        env = make_segmented_env("libero_spatial", task_id)
        try:
            with h5py.File(task_states[0]["demo_path"], "r") as handle:
                for index, state in enumerate(task_states, 1):
                    output = artifact / "states" / f"{state['state_id']}.json"
                    if output.exists():
                        continue
                    episode = handle["data"][state["demo_id"]]
                    states_np, actions_np = np.asarray(episode["states"]), np.asarray(episode["actions"])
                    xml = episode.attrs["model_file"]
                    xml = xml.decode() if isinstance(xml, bytes) else xml
                    env._env.reset(); env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml, workspace), {}, demo_generation=False)); env._env.env.sim.reset()
                    raw = env._env.regenerate_obs_from_state(states_np[state["resolved_frame"]])
                    batch = prepare_demo_state(strong_policy, preprocessor, env_preprocessor, env, raw, state["language"], actions_np, state["resolved_frame"])
                    raw_candidates = np.load(artifact / "neighbors" / f"{state['state_id']}.npz")["actions"]
                    raw_anchor = action_chunk(actions_np, state["resolved_frame"])[0].numpy()
                    rows, integrity = evaluate(models, preprocessor, batch, raw_candidates, raw_anchor, state)
                    temporary = output.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"state": state, "rows": rows, "integrity": integrity}, sort_keys=True) + "\n")
                    temporary.replace(output)
                    print(json.dumps({"task": task_id, "state": index, "of": len(task_states), "rows": len(rows)}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
