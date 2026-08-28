#!/usr/bin/env python3
"""Run paired 30-step clean versus first-chunk region-mask diagnostics."""

from __future__ import annotations

import argparse
import hashlib
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
from research.coreact_region.fixed_mask_sampler import prepare_ranked_prefix, sample_fixed_mask_actions
from research.coreact_region.region_mapping import (
    attach_prefix_indices,
    instance_label_map,
    mapping_sha256,
    protected_instance_names,
    select_region_groups,
    token_regions,
)
from research.coreact_region.segmented_runtime import (
    batched_observation,
    make_segmented_env,
    progress_snapshot,
    raw_observation,
    step_without_autoreset,
)


CONDITIONS = ("clean", "relevant_high", "background_high", "relevant_low", "background_random")
CAMERAS = (
    ("camera1", "agentview_segmentation_instance"),
    ("camera2", "robot0_eye_in_hand_segmentation_instance"),
)
CONTROL_STEPS = 30
EXECUTED_PREFIX = 10


def bytes_hash(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(value)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def progress_summary(initial, snapshots) -> dict:
    min_eef = min(row.eef_object_distance for row in snapshots)
    min_goal = min(row.object_goal_distance for row in snapshots)
    reach = float(np.clip((initial.eef_object_distance - min_eef) / max(initial.eef_object_distance, 1e-9), -1, 1))
    transport = float(np.clip((initial.object_goal_distance - min_goal) / max(initial.object_goal_distance, 1e-9), -1, 1))
    grasp = float(any(row.grasped for row in snapshots))
    predicate = float(any(row.predicate for row in snapshots))
    return {
        "initial": asdict(initial), "final": asdict(snapshots[-1]),
        "minimum_eef_object_distance": min_eef, "minimum_object_goal_distance": min_goal,
        "reach_progress": reach, "transport_progress": transport,
        "grasp_ever": bool(grasp), "predicate_ever": bool(predicate),
        "composite_progress": (reach + transport + grasp + predicate) / 4.0,
    }


def build_regions(env, raw, ranked, selection_seed: int) -> tuple[list[dict], dict, int]:
    instances = env._env.env.model.instances_to_ids
    labels = instance_label_map(instances)
    relevant = set(env._env.obj_of_interest)
    protected = protected_instance_names(instances)
    regions = []
    for camera_id, key in CAMERAS:
        regions.extend(
            token_regions(
                raw[key], camera_id=camera_id, label_by_name=labels,
                relevant_names=relevant, protected_names=protected,
            )
        )
    mapped = attach_prefix_indices(ranked["span_map"], regions)
    groups, count = select_region_groups(
        mapped, ranked["scores"], cap=4, minimum=2, random_seed=selection_seed
    )
    return mapped, groups, count


def run_episode(artifact, policy_config, policy, preprocessor, postprocessor, means, spec):
    episode_dir = artifact / "episodes" / spec["episode_id"]
    record_path = episode_dir / "episode.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    episode_dir.mkdir(parents=True, exist_ok=True)
    env = make_segmented_env(spec["suite"], spec["task_id"])
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=policy_config
    )
    try:
        env.init_state_id = spec["init_state_id"]
        observation, _ = env.reset(seed=spec["reset_seed"])
        initial_state_hash = bytes_hash(env._env.get_sim_state())
        initial_progress = progress_snapshot(env)
        initial_raw = raw_observation(env)
        frames, actions, snapshots, step_rows, noise_hashes = [], [], [], [], []
        queue, replans, first_trace, mapped, groups, group_count = [], 0, None, None, None, None
        policy_seconds = []
        for control_step in range(CONTROL_STEPS):
            frames.append(np.asarray(observation["pixels"]["image"]))
            if not queue:
                prepared = prepare(
                    policy,
                    preprocessor,
                    env_preprocessor,
                    batched_observation(observation),
                    spec["language"],
                )
                generator = torch.Generator(device=prepared["state"].device).manual_seed(
                    spec["action_noise_seed"] * 1000 + replans
                )
                noise = torch.randn(
                    (1, policy_config.chunk_size, policy_config.max_action_dim), generator=generator,
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
                        mapped, groups, group_count = build_regions(
                            env, initial_raw, ranked, spec["selection_seed"]
                        )
                        if spec["condition"] == "clean":
                            chunk = policy.model.sample_actions(
                                prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                                prepared["lang_masks"], prepared["state"], noise=noise,
                            )
                        else:
                            chunk, first_trace = sample_fixed_mask_actions(
                                policy.model, ranked, noise, means["visual_position_mean"],
                                groups[spec["condition"]],
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
            actions.append(model_action[0].detach().float().cpu())
            step_rows.append({
                "control_step": control_step, "replan": replans - 1,
                "model_action": actions[-1].tolist(), "legal_action": legal.tolist(),
                **asdict(snapshot),
            })
        progress = progress_summary(initial_progress, snapshots)
        action_array = torch.stack(actions)
        total_variation = float(torch.linalg.vector_norm(action_array[1:] - action_array[:-1], dim=1).sum())
        video_path, step_path = episode_dir / "rollout.mp4", episode_dir / "steps.jsonl"
        write_video(video_path, np.asarray(frames), fps=20)
        with step_path.open("w") as stream:
            for row in step_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        record = {
            **spec, "status": "complete", "initial_sim_state_sha256": initial_state_hash,
            "noise_sha256_by_replan": noise_hashes, "control_steps": len(step_rows),
            "masked_replans": [] if spec["condition"] == "clean" else [0],
            "region_mapping_sha256": mapping_sha256(mapped), "region_mapping": mapped,
            "all_candidate_groups": groups, "matched_group_count": group_count,
            "mask_trace": first_trace, "progress": progress,
            "action_total_variation": total_variation,
            "policy_seconds_median_per_replan": float(np.median(policy_seconds)),
            "all_outputs_finite": math.isfinite(total_variation),
            "video": str(video_path.relative_to(artifact)), "step_log": str(step_path.relative_to(artifact)),
        }
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
    gate = artifact / "qualification_gate.json"
    if not gate.exists() or not json.loads(gate.read_text()).get("pass"):
        raise RuntimeError("region rollout requires passing qualification")
    rows = read_jsonl(artifact / "episode_manifest.jsonl")
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    policy_config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    for ordinal, spec in enumerate(rows, 1):
        record = run_episode(artifact, policy_config, policy, preprocessor, postprocessor, means, spec)
        print(f"[{args.shard_index}] {ordinal}/{len(rows)} {spec['episode_id']} progress={record['progress']['composite_progress']:.4f}", flush=True)


if __name__ == "__main__":
    main()
