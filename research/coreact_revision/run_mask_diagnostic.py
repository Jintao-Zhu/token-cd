#!/usr/bin/env python3
"""Run append-only mask-only revision-development episodes."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.io_utils import write_video
from research.coreact_closed_loop.guidance import GuidanceConfig
from research.coreact_closed_loop.run_pilot import read_jsonl, vector_info_value, write_json
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_revision.masked_sampler import sample_masked_actions


MAX_CONTROL_STEPS = 280
EXECUTED_PREFIX = 10
CAMERA_IDS = ("camera1", "camera2")
MASK_SELECTION = {
    "top8_mask_only": "top",
    "random8_mask_only": "random",
    "bottom8_mask_only": "bottom",
}


def run_episode(artifact, policy_config, policy, preprocessor, postprocessor, means, spec):
    episode_dir = artifact / "episodes" / spec["episode_id"]
    record_path = episode_dir / "episode.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    episode_dir.mkdir(parents=True, exist_ok=True)
    env, env_preprocessor, env_postprocessor = make_task_env(spec["suite"], spec["task_id"], policy_config)
    frames, actions, step_rows, mask_traces, queue = [], [], [], [], []
    replans = nonfinite_action_count = normalized_bound_violation_count = 0
    success = False
    policy_seconds, chunk_boundaries = [], []
    try:
        env.envs[0].init_state_id = spec["init_state_id"]
        observation, _ = env.reset(seed=spec["reset_seed"])
        torch.cuda.reset_peak_memory_stats()
        for control_step in range(MAX_CONTROL_STEPS):
            frames.append(np.asarray(env.render())[0].copy())
            if not queue:
                prepared = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
                generator = torch.Generator(device=prepared["state"].device).manual_seed(
                    spec["action_noise_seed"] * 1000 + replans
                )
                noise = torch.randn(
                    (1, policy_config.chunk_size, policy_config.max_action_dim),
                    generator=generator,
                    dtype=prepared["state"].dtype,
                    device=prepared["state"].device,
                )
                started = time.perf_counter()
                with torch.inference_mode():
                    if spec["condition"] == "vanilla":
                        chunk = policy.model.sample_actions(
                            prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                            prepared["lang_masks"], prepared["state"], noise=noise,
                        )
                        trace = None
                    else:
                        selection = MASK_SELECTION.get(spec["condition"])
                        if selection is None:
                            raise ValueError(f"unknown condition {spec['condition']}")
                        chunk, trace = sample_masked_actions(
                            policy.model, prepared["images"], prepared["image_masks"],
                            prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise,
                            means["visual_position_mean"], camera_ids=CAMERA_IDS,
                            config=GuidanceConfig(selection=selection),
                            selection_seed=spec["selection_seed"] * 1000 + replans,
                        )
                torch.cuda.synchronize()
                policy_seconds.append(time.perf_counter() - started)
                model_prefix = chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1)
                if actions:
                    chunk_boundaries.append(float(torch.linalg.vector_norm(model_prefix[0, 0].cpu() - actions[-1])))
                queue.extend(model_prefix)
                if trace is not None:
                    mask_traces.append({"replan": replans, **trace})
                replans += 1

            model_action = queue.pop(0)
            finite = bool(torch.isfinite(model_action).all())
            nonfinite_action_count += int(not finite)
            if not finite:
                raise RuntimeError("nonfinite model action")
            normalized_bound_violation_count += int(bool((model_action.abs() > 1.000001).any()))
            physical_action = postprocessor(model_action)
            legal_action = env_postprocessor({"action": physical_action})["action"]
            observation, _, terminated, _, info = env.step(legal_action.detach().cpu().numpy())
            action_cpu = model_action[0].detach().float().cpu()
            actions.append(action_cpu)
            success = bool(vector_info_value(info, "is_success"))
            step_rows.append({
                "control_step": control_step, "replan": replans - 1,
                "model_action": action_cpu.tolist(),
                "physical_action": physical_action[0].detach().float().cpu().tolist(),
                "legal_action": legal_action[0].detach().float().cpu().tolist(), "success": success,
            })
            if bool(terminated[0]) or success:
                frames.append(np.asarray(env.render())[0].copy())
                break

        action_array = torch.stack(actions) if actions else torch.empty((0, 7))
        differences = action_array[1:] - action_array[:-1]
        total_variation = float(torch.linalg.vector_norm(differences, dim=1).sum()) if len(actions) > 1 else 0.0
        video_path, steps_path = episode_dir / "rollout.mp4", episode_dir / "steps.jsonl"
        write_video(video_path, np.asarray(frames), fps=20)
        with steps_path.open("w") as stream:
            for row in step_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        record = {
            **spec, "status": "complete", "success": success, "control_steps": len(step_rows),
            "replans": replans, "action_total_variation": total_variation,
            "mean_chunk_boundary_discontinuity": float(np.mean(chunk_boundaries)) if chunk_boundaries else 0.0,
            "nonfinite_action_count": nonfinite_action_count,
            "normalized_bound_violation_count": normalized_bound_violation_count,
            "policy_seconds_total": float(sum(policy_seconds)),
            "policy_seconds_median_per_replan": float(np.median(policy_seconds)),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "all_mask_outputs_finite": all(trace["all_output_finite"] for trace in mask_traces),
            "mask_traces": mask_traces,
            "video": str(video_path.relative_to(artifact)), "step_log": str(steps_path.relative_to(artifact)),
        }
        if not math.isfinite(total_variation):
            raise RuntimeError("nonfinite action total variation")
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
    artifact, workspace = args.artifact.resolve(), args.workspace.resolve()
    gate = artifact / "integrity_report.json"
    if not gate.exists() or not json.loads(gate.read_text()).get("pass"):
        raise RuntimeError("mask rollouts require a passing integrity gate")
    rows = read_jsonl(artifact / "episode_manifest.jsonl")
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    policy_config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    for ordinal, spec in enumerate(rows, 1):
        record = run_episode(artifact, policy_config, policy, preprocessor, postprocessor, means, spec)
        print(f"[{args.shard_index}] {ordinal}/{len(rows)} {spec['episode_id']} success={record['success']} steps={record['control_steps']}", flush=True)


if __name__ == "__main__":
    main()
