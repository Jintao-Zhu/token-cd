#!/usr/bin/env python3
"""Run 2-3 replan image-space interventions on a new confirmation split."""

from __future__ import annotations

import argparse
import json
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
from research.coreact_region.segmented_runtime import batched_observation, make_segmented_env, progress_snapshot, raw_observation, step_without_autoreset


def run_one(artifact, config, policy, preprocessor, postprocessor, spec):
    episode_dir = artifact / "episodes" / spec["episode_id"]
    record_path = episode_dir / "episode.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    episode_dir.mkdir(parents=True, exist_ok=True)
    env = make_segmented_env(spec["suite"], spec["task_id"])
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=config)
    try:
        env.init_state_id = spec["init_state_id"]
        env.reset(seed=spec["reset_seed"])
        observation = restore(env, np.load(artifact / spec["state_path"], allow_pickle=False))
        initial_hash = array_sha256(np.asarray(env._env.get_sim_state()))
        if initial_hash != spec["sim_state_sha256"]:
            raise RuntimeError("simulator state restoration mismatch")
        initial_progress = progress_snapshot(env)
        initial_bundle, initial_audit = build_image_masks(env, raw_observation(env), random_seed=spec["selection_seed"] * 100)
        if initial_audit["bundle_sha256"] != spec["initial_mask_bundle_sha256"]:
            raise RuntimeError("initial image mask changed after lock")
        queue, frames, snapshots, actions, rows, noise_hashes, intervention_traces = [], [], [], [], [], [], []
        replans, policy_seconds = 0, []
        for control_step in range(60):
            frames.append(np.asarray(observation["pixels"]["image"]).copy())
            if not queue:
                policy_observation = observation
                if replans < spec["intervention_replans"]:
                    if replans == 0:
                        bundle, mask_audit = initial_bundle, initial_audit
                    else:
                        bundle, mask_audit = build_image_masks(
                            env, raw_observation(env), random_seed=spec["selection_seed"] * 100 + replans
                        )
                    policy_observation, trace = apply_image_condition(observation, bundle, spec["base_condition"])
                    intervention_traces.append({"replan": replans, "mask_bundle_sha256": mask_audit["bundle_sha256"], "trace": trace})
                prepared = prepare(policy, preprocessor, env_preprocessor, batched_observation(policy_observation), spec["language"])
                generator = torch.Generator(device=prepared["state"].device).manual_seed(spec["rollout_noise_seed"] * 1000 + replans)
                noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, dtype=prepared["state"].dtype, device=prepared["state"].device)
                noise_hashes.append(tensor_sha256(noise))
                started = time.perf_counter()
                with torch.inference_mode():
                    chunk = policy.model.sample_actions(prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise=noise)
                torch.cuda.synchronize(); policy_seconds.append(time.perf_counter() - started)
                queue.extend(chunk[:, :10, :7].transpose(0, 1)); replans += 1
            model_action = queue.pop(0)
            if not bool(torch.isfinite(model_action).all()):
                raise RuntimeError("nonfinite model action")
            physical = postprocessor(model_action)
            legal = env_postprocessor({"action": physical})["action"][0].detach().cpu().numpy()
            observation, snapshot = step_without_autoreset(env, legal)
            snapshots.append(snapshot); actions.append(model_action[0].detach().float().cpu().numpy())
            rows.append({"control_step": control_step, "replan": replans - 1, "model_action": actions[-1].tolist(), "legal_action": legal.tolist(), **asdict(snapshot)})
        video_path, step_path = episode_dir / "rollout.mp4", episode_dir / "steps.jsonl"
        write_video(video_path, np.asarray(frames), fps=20)
        with step_path.open("w") as stream:
            for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
        action_array = np.asarray(actions)
        record = {
            **spec, "status": "complete", "control_steps": 60,
            "initial_sim_state_sha256": initial_hash, "initial_mask_bundle_sha256_runtime": initial_audit["bundle_sha256"],
            "noise_sha256_by_replan": noise_hashes, "actual_intervention_replans": [row["replan"] for row in intervention_traces],
            "intervention_traces": intervention_traces,
            "progress_by_horizon": {str(h): summarize_progress(initial_progress, snapshots, h) for h in HORIZONS},
            "actions_sha256": array_sha256(action_array), "all_outputs_finite": bool(np.isfinite(action_array).all()),
            "policy_seconds_median_per_replan": float(np.median(policy_seconds)),
            "video": str(video_path.relative_to(artifact)), "step_log": str(step_path.relative_to(artifact)),
        }
        write_json(record_path, record); return record
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(); workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if not json.loads((artifact / "candidate_gate.json").read_text()).get("pass"):
        raise RuntimeError("candidate gate failed")
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    rows = read_jsonl(artifact / "rollout_manifest.jsonl")
    for ordinal, spec in enumerate(rows, 1):
        record = run_one(artifact, config, policy, preprocessor, postprocessor, spec)
        print(f"{ordinal}/{len(rows)} {spec['episode_id']} H60={record['progress_by_horizon']['60']['composite_progress']:.4f}", flush=True)


if __name__ == "__main__": main()
