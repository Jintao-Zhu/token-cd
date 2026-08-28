#!/usr/bin/env python3
"""Append-only paired rollout runner for online consistency guidance."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.io_utils import write_video
from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions
from research.coreact_closed_loop.run_pilot import vector_info_value, write_json
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_revision.consistency_guidance import sample_consistency_guided_actions
from research.coreact_revision.run_online_consistency_qualification import fit_task_heldout_classifier

MAX_STEPS = 280
EXECUTED_PREFIX = 10


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, required=True); p.add_argument("--artifact", type=Path, required=True); args = p.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    manifest = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines() if line.strip()]
    source = workspace / "artifacts/coreact_region_sign_multitask_v1_20260808_093747"
    ablation = workspace / "artifacts/coreact_action_consistency_ablation_v1_20260808_113116"
    task_ids = {row["task_id"] for row in manifest}
    if len(task_ids) != 1: raise RuntimeError("pilot artifact must contain exactly one held-out task")
    classifier, _, numeric = fit_task_heldout_classifier(source / "raw_effects.jsonl", ablation / "consistency_scores.jsonl", heldout_task=next(iter(task_ids)))
    cfg, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    guidance = GuidanceConfig(selection="top", group_count=8, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10)
    for ordinal, spec in enumerate(manifest, 1):
        episode_dir = artifact / "episodes" / spec["episode_id"]; record_path = episode_dir / "episode.json"
        if record_path.exists():
            print(f"[{ordinal}/{len(manifest)}] skip complete {spec['episode_id']}", flush=True); continue
        episode_dir.mkdir(parents=True, exist_ok=True)
        env, env_pre, env_post = make_task_env(spec["suite"], spec["task_id"], cfg)
        frames, actions, steps, traces, queue, latencies = [], [], [], [], [], []
        success = False; replans = 0
        try:
            env.envs[0].init_state_id = spec["init_state_id"]
            observation, _ = env.reset(seed=spec["reset_seed"])
            torch.cuda.reset_peak_memory_stats()
            for control_step in range(MAX_STEPS):
                frames.append(np.asarray(env.render())[0].copy())
                if not queue:
                    prepared = prepare(policy, preprocessor, env_pre, observation, spec["language"])
                    generator = torch.Generator(device=prepared["state"].device).manual_seed(spec["action_noise_seed"] * 1000 + replans)
                    noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=generator, device=prepared["state"].device, dtype=prepared["state"].dtype)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        if spec["condition"] == "vanilla":
                            chunk = policy.model.sample_actions(prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise=noise); trace = None
                        elif spec["condition"] == "attention_toward":
                            chunk, trace = sample_coreact_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means["visual_position_mean"], config=GuidanceConfig(group_count=8, guidance_scale=.5, trust_region_kappa=.25, action_dim=7, num_steps=10, direction="toward"), selection_seed=spec["action_noise_seed"] * 1000 + replans)
                        else:
                            mode = {"consistency_guided": "away", "consistency_mask_only": "mask_only", "consistency_toward": "toward"}.get(spec["condition"])
                            if mode is None: raise ValueError(f"unknown condition {spec['condition']}")
                            chunk, trace = sample_consistency_guided_actions(policy.model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise, means, classifier, numeric, config=guidance, mode=mode, nuisance_threshold=.5)
                    torch.cuda.synchronize(); latencies.append(time.perf_counter() - started)
                    queue.extend(chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1)); replans += 1
                    if trace is not None: traces.append({"replan": replans - 1, **trace})
                model_action = queue.pop(0)
                if not bool(torch.isfinite(model_action).all()): raise RuntimeError("nonfinite model action")
                physical = postprocessor(model_action); legal = env_post({"action": physical})["action"]
                observation, _, terminated, _, info = env.step(legal.detach().cpu().numpy())
                action_cpu = model_action[0].detach().float().cpu(); actions.append(action_cpu)
                success = bool(vector_info_value(info, "is_success"))
                steps.append({"control_step": control_step, "replan": replans - 1, "model_action": action_cpu.tolist(), "success": success})
                if bool(terminated[0]) or success:
                    frames.append(np.asarray(env.render())[0].copy()); break
            action_array = torch.stack(actions); total_variation = float(torch.linalg.vector_norm(action_array[1:] - action_array[:-1], dim=1).sum()) if len(actions) > 1 else 0.0
            write_video(episode_dir / "rollout.mp4", np.asarray(frames), fps=20)
            with (episode_dir / "steps.jsonl").open("w") as stream:
                for row in steps: stream.write(json.dumps(row, sort_keys=True) + "\n")
            record = {**spec, "status": "complete", "success": success, "control_steps": len(actions), "replans": replans, "action_total_variation": total_variation, "policy_seconds_total": float(sum(latencies)), "policy_seconds_median_per_replan": float(np.median(latencies)), "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()), "fallback_count": sum(t["fallback_count"] for t in traces), "all_outputs_finite": all(t["all_output_finite"] for t in traces), "replan_traces": traces, "video": str((episode_dir / "rollout.mp4").relative_to(artifact))}
            if not math.isfinite(total_variation): raise RuntimeError("nonfinite action total variation")
            write_json(record_path, record)
            print(f"[{ordinal}/{len(manifest)}] {spec['episode_id']} success={success} steps={len(actions)}", flush=True)
        finally:
            env.close()


if __name__ == "__main__": main()
