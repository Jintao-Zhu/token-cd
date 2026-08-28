#!/usr/bin/env python3
"""Append-only runner for task-4 vanilla, CoreAct, and mask-only conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.utils.io_utils import write_video

from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.run_pilot import read_jsonl, vector_info_value, write_json
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_revision.masked_sampler import sample_masked_actions


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value); digest = hashlib.sha256(); digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes()); return digest.hexdigest()


def prepared_hash(prepared: dict) -> str:
    values = [*prepared["images"], *prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"]]
    return hashlib.sha256("".join(tensor_sha256(value) for value in values).encode("ascii")).hexdigest()


def condition_mode(condition: str, group_count: int) -> str:
    modes = {
        "vanilla": "vanilla",
        "coreact_top8": "away" if group_count == 8 else None,
        f"coreact_away_top{group_count}": "away",
        f"coreact_toward_top{group_count}": "toward",
        f"top{group_count}_mask_only": "mask_only",
    }
    mode = modes.get(condition)
    if mode is None:
        raise ValueError(f"condition {condition!r} does not match group_count={group_count}")
    return mode


def run_one(artifact, config, policy, preprocessor, postprocessor, means, spec, *, reset_count=1):
    episode_dir = artifact / "episodes" / spec["episode_id"]; record_path = episode_dir / "episode.json"
    if record_path.exists(): return json.loads(record_path.read_text())
    episode_dir.mkdir(parents=True, exist_ok=True)
    env, env_preprocessor, env_postprocessor = make_task_env(spec["suite"], spec["task_id"], config)
    group_count = int(spec.get("group_count", 8))
    mode = condition_mode(spec["condition"], group_count)
    guidance_config = GuidanceConfig(selection="top", group_count=group_count, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10)
    frames, actions, rows, traces, queue, noise_hashes = [], [], [], [], [], []
    replans = nonfinite = bound_violations = 0; success = False; latencies, boundaries = [], []
    try:
        inner = env.envs[0]; inner.init_state_id = spec["init_state_id"]
        for _ in range(reset_count):
            observation, _ = env.reset(seed=spec["reset_seed"])
        initial_state_hash = array_hash(np.asarray(inner._env.get_sim_state()))
        initial_prepared_hash = None; torch.cuda.reset_peak_memory_stats()
        for control_step in range(280):
            frames.append(np.asarray(env.render())[0].copy())
            if not queue:
                prepared = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
                if initial_prepared_hash is None: initial_prepared_hash = prepared_hash(prepared)
                generator = torch.Generator(device=prepared["state"].device).manual_seed(spec["action_noise_seed"] * 1000 + replans)
                noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, dtype=prepared["state"].dtype, device=prepared["state"].device); noise_hashes.append(tensor_sha256(noise))
                started = time.perf_counter()
                with torch.inference_mode():
                    if mode == "vanilla":
                        chunk = policy.model.sample_actions(prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise=noise); trace = None
                    elif mode == "away":
                        chunk, trace = sample_coreact_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"], config=guidance_config, selection_seed=spec["selection_seed"] * 1000 + replans)
                    elif mode == "toward":
                        toward_config = GuidanceConfig(selection="top", group_count=group_count, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10, direction="toward")
                        chunk, trace = sample_coreact_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"], config=toward_config, selection_seed=spec["selection_seed"] * 1000 + replans)
                    elif mode == "mask_only":
                        chunk, trace = sample_masked_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"], config=guidance_config, selection_seed=spec["selection_seed"] * 1000 + replans)
                torch.cuda.synchronize(); latencies.append(time.perf_counter() - started)
                prefix = chunk[:, :10, :7].transpose(0, 1)
                if actions: boundaries.append(float(torch.linalg.vector_norm(prefix[0, 0].cpu() - actions[-1])))
                queue.extend(prefix)
                if trace is not None: traces.append({"replan": replans, **trace})
                replans += 1
            model_action = queue.pop(0); finite = bool(torch.isfinite(model_action).all()); nonfinite += int(not finite)
            if not finite: raise RuntimeError("nonfinite action")
            bound_violations += int(bool((model_action.abs() > 1.000001).any()))
            physical = postprocessor(model_action); legal = env_postprocessor({"action": physical})["action"]
            observation, _, terminated, _, info = env.step(legal.detach().cpu().numpy())
            action_cpu = model_action[0].detach().float().cpu(); actions.append(action_cpu); success = bool(vector_info_value(info, "is_success"))
            rows.append({"control_step": control_step, "replan": replans - 1, "model_action": action_cpu.tolist(), "physical_action": physical[0].detach().float().cpu().tolist(), "legal_action": legal[0].detach().float().cpu().tolist(), "success": success})
            if bool(terminated[0]) or success:
                frames.append(np.asarray(env.render())[0].copy()); break
        action_array = torch.stack(actions); total_variation = float(torch.linalg.vector_norm(action_array[1:] - action_array[:-1], dim=1).sum()) if len(actions) > 1 else 0.0
        video, steps = episode_dir / "rollout.mp4", episode_dir / "steps.jsonl"; write_video(video, np.asarray(frames), fps=20)
        with steps.open("w") as stream:
            for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
        record = {
            **spec, "status": "complete", "success": success, "control_steps": len(rows), "replans": replans,
            "initial_sim_state_sha256": initial_state_hash, "initial_prepared_input_sha256": initial_prepared_hash,
            "noise_sha256_by_replan": noise_hashes, "action_total_variation": total_variation,
            "mean_chunk_boundary_discontinuity": float(np.mean(boundaries)) if boundaries else 0.0,
            "nonfinite_action_count": nonfinite, "normalized_bound_violation_count": bound_violations,
            "policy_seconds_total": float(sum(latencies)), "policy_seconds_median_per_replan": float(np.median(latencies)),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "fallback_count": sum(trace["fallback_count"] for trace in traces),
            "all_sampler_outputs_finite": all(trace["all_output_finite"] for trace in traces),
            "replan_traces": traces, "video": str(video.relative_to(artifact)), "step_log": str(steps.relative_to(artifact)),
        }
        if not math.isfinite(total_variation): raise RuntimeError("nonfinite total variation")
        write_json(record_path, record); return record
    finally: env.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True); parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shard-count", type=int, default=1); parser.add_argument("--reset-count", type=int, default=1); args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve(); gate = json.loads((artifact / "integrity_report.json").read_text())
    if not gate.get("pass"): raise RuntimeError("integrity gate failed")
    rows = [r for i,r in enumerate(read_jsonl(artifact / "episode_manifest.jsonl")) if i % args.shard_count == args.shard_index]
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace); means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    for i,spec in enumerate(rows,1):
        record = run_one(artifact, config, policy, preprocessor, postprocessor, means, spec, reset_count=args.reset_count); print(f"[{args.shard_index}] {i}/{len(rows)} {spec['episode_id']} success={record['success']} steps={record['control_steps']}", flush=True)


if __name__ == "__main__": main()
