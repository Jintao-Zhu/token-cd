#!/usr/bin/env python3
"""Run paired first-chunk pre-encoder image intervention rollouts."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.utils.io_utils import write_video

from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors, prepare
from research.coreact_region.audit_effect_candidates import restore
from research.coreact_region.effect_existence import HORIZONS, array_sha256, summarize_progress
from research.coreact_region.image_intervention import apply_image_condition, build_image_masks
from research.coreact_region.segmented_runtime import (
    batched_observation,
    make_segmented_env,
    progress_snapshot,
    raw_observation,
    step_without_autoreset,
)


CONTROL_STEPS = 60
EXECUTED_PREFIX = 10


def run_one(artifact, config, policy, preprocessor, postprocessor, spec):
    episode_dir = artifact / "episodes" / spec["episode_id"]
    record_path = episode_dir / "episode.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    episode_dir.mkdir(parents=True, exist_ok=True)
    env = make_segmented_env(spec["suite"], spec["task_id"])
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=config
    )
    try:
        env.init_state_id = spec["init_state_id"]
        env.reset(seed=spec["reset_seed"])
        state = np.load(artifact / spec["state_path"], allow_pickle=False)
        observation = restore(env, state)
        initial_hash = array_sha256(np.asarray(env._env.get_sim_state()))
        if initial_hash != spec["sim_state_sha256"]:
            raise RuntimeError("critical simulator state restoration mismatch")
        initial_progress = progress_snapshot(env)
        bundle, mask_audit = build_image_masks(
            env, raw_observation(env), random_seed=spec["selection_seed"]
        )
        if mask_audit["bundle_sha256"] != spec["mask_bundle_sha256"]:
            raise RuntimeError("image mask bundle changed after candidate lock")
        queue, frames, snapshots, actions, step_rows, noise_hashes = [], [], [], [], [], []
        replans, first_trace = 0, None
        policy_seconds = []
        for control_step in range(CONTROL_STEPS):
            frames.append(np.asarray(observation["pixels"]["image"]).copy())
            if not queue:
                policy_observation = observation
                if replans == 0:
                    policy_observation, first_trace = apply_image_condition(
                        observation, bundle, spec["condition"]
                    )
                prepared = prepare(
                    policy, preprocessor, env_preprocessor,
                    batched_observation(policy_observation), spec["language"],
                )
                generator = torch.Generator(device=prepared["state"].device).manual_seed(
                    spec["rollout_noise_seed"] * 1000 + replans
                )
                noise = torch.randn(
                    (1, config.chunk_size, config.max_action_dim), generator=generator,
                    dtype=prepared["state"].dtype, device=prepared["state"].device,
                )
                noise_hashes.append(tensor_sha256(noise))
                started = time.perf_counter()
                with torch.inference_mode():
                    chunk = policy.model.sample_actions(
                        prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                        prepared["lang_masks"], prepared["state"], noise=noise,
                    )
                torch.cuda.synchronize()
                policy_seconds.append(time.perf_counter() - started)
                queue.extend(chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1))
                replans += 1
            model_action = queue.pop(0)
            if not bool(torch.isfinite(model_action).all()):
                raise RuntimeError("nonfinite model action")
            physical = postprocessor(model_action)
            legal = env_postprocessor({"action": physical})["action"][0].detach().cpu().numpy()
            observation, snapshot = step_without_autoreset(env, legal)
            snapshots.append(snapshot)
            actions.append(model_action[0].detach().float().cpu().numpy())
            step_rows.append({
                "control_step": control_step, "replan": replans - 1,
                "model_action": actions[-1].tolist(), "legal_action": legal.tolist(),
                **asdict(snapshot),
            })
        progress = {str(h): summarize_progress(initial_progress, snapshots, h) for h in HORIZONS}
        video_path, step_path = episode_dir / "rollout.mp4", episode_dir / "steps.jsonl"
        write_video(video_path, np.asarray(frames), fps=20)
        with step_path.open("w") as stream:
            for row in step_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        action_array = np.asarray(actions)
        record = {
            **spec, "status": "complete", "control_steps": len(snapshots),
            "initial_sim_state_sha256": initial_hash, "noise_sha256_by_replan": noise_hashes,
            "mask_bundle_sha256_runtime": mask_audit["bundle_sha256"],
            "intervention_replans": [] if spec["condition"] == "vanilla" else [0],
            "first_intervention_trace": first_trace, "progress_by_horizon": progress,
            "actions_sha256": array_sha256(action_array),
            "all_outputs_finite": bool(np.isfinite(action_array).all()),
            "policy_seconds_median_per_replan": float(np.median(policy_seconds)),
            "video": str(video_path.relative_to(artifact)),
            "step_log": str(step_path.relative_to(artifact)),
        }
        if not record["all_outputs_finite"] or not math.isfinite(record["policy_seconds_median_per_replan"]):
            raise RuntimeError("nonfinite rollout output")
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
    gate = json.loads((artifact / "candidate_gate.json").read_text())
    if not gate.get("pass"):
        raise RuntimeError("image candidate qualification did not pass")
    rows = read_jsonl(artifact / "rollout_manifest.jsonl")
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    for ordinal, spec in enumerate(rows, 1):
        record = run_one(artifact, config, policy, preprocessor, postprocessor, spec)
        print(
            f"[{args.shard_index}] {ordinal}/{len(rows)} {spec['episode_id']} "
            f"H60={record['progress_by_horizon']['60']['composite_progress']:.4f}", flush=True,
        )


if __name__ == "__main__":
    main()
