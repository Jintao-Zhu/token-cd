#!/usr/bin/env python3
"""Run append-only 60-step paired rollouts for the Effect-Existence Gate."""

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
from research.coreact_region.effect_existence import HORIZONS, array_sha256, build_effect_groups, summarize_progress
from research.coreact_region.fixed_mask_sampler import prepare_ranked_prefix, sample_fixed_mask_actions
from research.coreact_region.segmented_runtime import (
    batched_observation,
    make_segmented_env,
    progress_snapshot,
    raw_observation,
    step_without_autoreset,
)


CONTROL_STEPS = 60
EXECUTED_PREFIX = 10


def run_one(artifact, config, policy, preprocessor, postprocessor, means, spec):
    episode_dir = artifact / "episodes" / spec["episode_id"]
    record_path = episode_dir / "episode.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    episode_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((artifact / spec["candidate_audit_path"]).read_text())
    selected = [] if spec["group_name"] is None else audit["groups"][spec["group_name"]]
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
        initial_raw = raw_observation(env)
        queue, frames, snapshots, actions, step_rows, noise_hashes = [], [], [], [], [], []
        replans, first_trace, mapping_hash = 0, None, None
        policy_seconds = []
        for control_step in range(CONTROL_STEPS):
            frames.append(np.asarray(observation["pixels"]["image"]).copy())
            if not queue:
                prepared = prepare(
                    policy, preprocessor, env_preprocessor, batched_observation(observation), spec["language"]
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
                    if replans == 0:
                        ranked = prepare_ranked_prefix(
                            policy.model, prepared["images"], prepared["image_masks"],
                            prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise,
                        )
                        mapped, current_groups, current_audit = build_effect_groups(
                            env, initial_raw, ranked, random_seed=spec["selection_seed"]
                        )
                        mapping_hash = current_audit["groups_sha256"]
                        if current_groups != audit["groups"]:
                            raise RuntimeError("candidate groups changed between audit and rollout")
                        if spec["condition"] == "clean":
                            chunk = policy.model.sample_actions(
                                prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                                prepared["lang_masks"], prepared["state"], noise=noise,
                            )
                        else:
                            chunk, first_trace = sample_fixed_mask_actions(
                                policy.model, ranked, noise, means["visual_position_mean"], selected,
                                replacement_type=spec["replacement_type"],
                            )
                    else:
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
        summaries = {str(h): summarize_progress(initial_progress, snapshots, h) for h in HORIZONS}
        video_path = episode_dir / "rollout.mp4"
        step_path = episode_dir / "steps.jsonl"
        write_video(video_path, np.asarray(frames), fps=20)
        with step_path.open("w") as stream:
            for row in step_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        action_array = np.asarray(actions)
        record = {
            **spec, "status": "complete", "control_steps": len(snapshots),
            "initial_sim_state_sha256": initial_hash, "noise_sha256_by_replan": noise_hashes,
            "candidate_groups_sha256": mapping_hash,
            "selected_indices": selected, "selected_count": len(selected),
            "masked_replans": [] if spec["condition"] == "clean" else [0],
            "mask_trace": first_trace, "progress_by_horizon": summaries,
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
        raise RuntimeError("candidate gate did not pass")
    rows = read_jsonl(artifact / "rollout_manifest.jsonl")
    rows = [row for i, row in enumerate(rows) if i % args.shard_count == args.shard_index]
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(
        workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",
        weights_only=True, map_location="cpu",
    )
    for i, spec in enumerate(rows, 1):
        record = run_one(artifact, config, policy, preprocessor, postprocessor, means, spec)
        print(
            f"[{args.shard_index}] {i}/{len(rows)} {spec['episode_id']} "
            f"H60={record['progress_by_horizon']['60']['composite_progress']:.4f}", flush=True,
        )


if __name__ == "__main__":
    main()
