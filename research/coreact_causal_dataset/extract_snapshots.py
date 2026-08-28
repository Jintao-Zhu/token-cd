#!/usr/bin/env python3
"""Collect clean trajectories and save one phase-targeted replan snapshot per trajectory."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_causal_dataset.common import canonical_sha256, select_phase_index
from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors, prepare
from research.coreact_region.effect_existence import array_sha256
from research.coreact_region.segmented_runtime import (
    batched_observation,
    make_segmented_env,
    progress_snapshot,
    step_without_autoreset,
)


MAX_STEPS = 280
EXECUTED_PREFIX = 10


def run_clean(artifact, config, policy, preprocessor, postprocessor, spec):
    output = artifact / "clean_extraction" / spec["snapshot_id"]
    record_path = output / "trajectory.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    env = make_segmented_env(spec["suite"], spec["task_id"])
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=config
    )
    try:
        env.init_state_id = spec["init_state_id"]
        observation, _ = env.reset(seed=spec["reset_seed"])
        queue = []
        replans = 0
        recent_actions: list[list[float]] = []
        boundary_rows, boundary_states = [], []
        all_actions, noise_hashes, latencies = [], [], []
        success = False
        for control_step in range(MAX_STEPS):
            if not queue:
                snapshot = progress_snapshot(env)
                state = np.asarray(env._env.get_sim_state()).copy()
                boundary_states.append(state)
                boundary_rows.append({
                    "boundary_index": len(boundary_rows), "control_step": control_step,
                    "normalized_progress": control_step / MAX_STEPS, **asdict(snapshot),
                    "sim_state_sha256": array_sha256(state),
                    "recent_model_actions": recent_actions[-EXECUTED_PREFIX:],
                    "recent_actions_sha256": canonical_sha256(recent_actions[-EXECUTED_PREFIX:]),
                })
                prepared = prepare(
                    policy, preprocessor, env_preprocessor, batched_observation(observation), spec["language"]
                )
                generator = torch.Generator(device=prepared["state"].device).manual_seed(
                    spec["extraction_noise_seed"] * 1000 + replans
                )
                noise = torch.randn(
                    (1, config.chunk_size, config.max_action_dim), generator=generator,
                    device=prepared["state"].device, dtype=prepared["state"].dtype,
                )
                noise_hashes.append(tensor_sha256(noise))
                started = time.perf_counter()
                with torch.inference_mode():
                    chunk = policy.model.sample_actions(
                        prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                        prepared["lang_masks"], prepared["state"], noise=noise,
                    )
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - started)
                queue.extend(chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1))
                replans += 1
            model_action = queue.pop(0)
            if not bool(torch.isfinite(model_action).all()):
                raise RuntimeError("nonfinite clean extraction action")
            action_row = model_action[0].detach().float().cpu().tolist()
            recent_actions.append(action_row)
            all_actions.append(action_row)
            physical = postprocessor(model_action)
            legal = env_postprocessor({"action": physical})["action"][0].detach().cpu().numpy()
            observation, after = step_without_autoreset(env, legal)
            success = bool(after.predicate)
            if success:
                break
        index, assigned_phase, fallback = select_phase_index(boundary_rows, spec["target_phase"])
        selected = boundary_rows[index]
        state_path = output / "snapshot_state.npy"
        np.save(state_path, boundary_states[index], allow_pickle=False)
        record = {
            **spec, "status": "complete", "clean_success": success,
            "clean_control_steps": len(all_actions), "clean_replans": replans,
            "selected_boundary_index": index, "assigned_phase": assigned_phase,
            "phase_fallback": fallback, "selected_boundary": selected,
            "state_path": str(state_path.relative_to(artifact)),
            "sim_state_sha256": selected["sim_state_sha256"],
            "actions_sha256": canonical_sha256(all_actions), "noise_sha256_by_replan": noise_hashes,
            "policy_seconds_median": float(np.median(latencies)),
            "boundary_metadata_sha256": canonical_sha256(boundary_rows),
        }
        write_json(output / "boundaries.json", {"boundaries": boundary_rows})
        write_json(record_path, record)
        return record
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    specs = [row for i, row in enumerate(read_jsonl(artifact / "snapshot_plan.jsonl"))
             if i % args.shard_count == args.shard_index]
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    for ordinal, spec in enumerate(specs, 1):
        row = run_clean(artifact, config, policy, preprocessor, postprocessor, spec)
        print(f"[{args.shard_index}] {ordinal}/{len(specs)} {spec['snapshot_id']} "
              f"success={row['clean_success']} phase={row['assigned_phase']} fallback={row['phase_fallback']}", flush=True)


if __name__ == "__main__":
    main()

