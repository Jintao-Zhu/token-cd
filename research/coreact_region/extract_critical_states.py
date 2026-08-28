#!/usr/bin/env python3
"""Extract critical simulator snapshots using clean trajectories only."""

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
from research.coreact_closed_loop.run_pilot import write_json
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors, prepare
from research.coreact_region.effect_existence import array_sha256, select_critical_indices
from research.coreact_region.segmented_runtime import (
    batched_observation,
    make_segmented_env,
    progress_snapshot,
    step_without_autoreset,
)


SUITES = ("libero_spatial", "libero_object")
TASK_ID = 7
INIT_IDS = tuple(range(15, 20))
CONTROL_STEPS = 60
EXECUTED_PREFIX = 10


def read_tasks(artifact: Path) -> dict:
    return {row["suite"]: row for row in json.loads((artifact / "task_manifest.json").read_text())}


def extract_one(
    artifact, config, policy, preprocessor, postprocessor, suite, init_id, task,
    *, control_steps: int, seed_base: int,
):
    trajectory_id = f"clean_extract__{suite}__task{TASK_ID:02d}__init{init_id:02d}"
    output_dir = artifact / "clean_extraction" / trajectory_id
    record_path = output_dir / "trajectory.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_segmented_env(suite, TASK_ID)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config(suite, TASK_ID), policy_cfg=config
    )
    reset_seed = seed_base + SUITES.index(suite) * 1_000_000 + TASK_ID * 10_000 + init_id * 100 + 1
    noise_seed = reset_seed + 1
    try:
        env.init_state_id = init_id
        observation, _ = env.reset(seed=reset_seed)
        queue, frames, states, before, after, step_rows, noise_hashes = [], [], [], [], [], [], []
        actions, replans, policy_seconds = [], 0, []
        for control_step in range(control_steps):
            states.append(np.asarray(env._env.get_sim_state()).copy())
            before.append(progress_snapshot(env))
            frames.append(np.asarray(observation["pixels"]["image"]).copy())
            if not queue:
                prepared = prepare(
                    policy, preprocessor, env_preprocessor, batched_observation(observation), task["language"]
                )
                generator = torch.Generator(device=prepared["state"].device).manual_seed(noise_seed * 1000 + replans)
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
            physical = postprocessor(model_action)
            legal = env_postprocessor({"action": physical})["action"][0].detach().cpu().numpy()
            observation, snapshot = step_without_autoreset(env, legal)
            after.append(snapshot)
            actions.append(model_action[0].detach().float().cpu().numpy())
            step_rows.append({
                "control_step": control_step, "replan": replans - 1,
                "before": asdict(before[-1]), "after": asdict(snapshot),
                "model_action": actions[-1].tolist(), "legal_action": legal.tolist(),
                "sim_state_sha256": array_sha256(states[-1]),
            })
        selections = select_critical_indices(
            before, after, near_reach_threshold_m=0.08, near_place_threshold_m=0.12
        )
        critical = []
        for selection in selections:
            index = selection["state_index"]
            state_path = output_dir / f"{selection['phase']}_state.npy"
            np.save(state_path, states[index], allow_pickle=False)
            critical.append({
                **selection,
                "critical_state_id": f"{trajectory_id}__{selection['phase']}",
                "suite": suite, "task_id": TASK_ID, "init_state_id": init_id,
                "language": task["language"], "reset_seed": reset_seed,
                "rollout_noise_seed": noise_seed + 50,
                "selection_seed": noise_seed + 51,
                "state_path": str(state_path.relative_to(artifact)),
                "sim_state_sha256": array_sha256(states[index]),
                "clean_before_progress": asdict(before[index]),
                "clean_after_progress": asdict(after[index]),
            })
        step_path = output_dir / "steps.jsonl"
        with step_path.open("w") as stream:
            for row in step_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        video_path = output_dir / "clean_rollout.mp4"
        write_video(video_path, np.asarray(frames), fps=20)
        record = {
            "trajectory_id": trajectory_id, "suite": suite, "task_id": TASK_ID,
            "init_state_id": init_id, "language": task["language"], "reset_seed": reset_seed,
            "action_noise_seed": noise_seed, "control_steps": control_steps,
            "noise_sha256_by_replan": noise_hashes,
            "actions_sha256": array_sha256(np.asarray(actions)),
            "sim_states_sha256": array_sha256(np.asarray(states)),
            "selected_critical_states": critical,
            "first_grasp_step": next((i for i, row in enumerate(after) if row.grasped), None),
            "first_predicate_step": next((i for i, row in enumerate(after) if row.predicate), None),
            "policy_seconds_median_per_replan": float(np.median(policy_seconds)),
            "step_log": str(step_path.relative_to(artifact)),
            "video": str(video_path.relative_to(artifact)),
        }
        write_json(record_path, record)
        return record
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--init-ids", type=int, nargs="+", default=list(INIT_IDS))
    parser.add_argument("--control-steps", type=int, default=CONTROL_STEPS)
    parser.add_argument("--seed-base", type=int, default=71_000_000)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if (artifact / "critical_state_manifest.jsonl").exists():
        raise RuntimeError("critical-state manifest already locked")
    tasks = read_tasks(artifact)
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    records = []
    for suite in SUITES:
        for init_id in args.init_ids:
            record = extract_one(
                artifact, config, policy, preprocessor, postprocessor, suite, init_id, tasks[suite],
                control_steps=args.control_steps, seed_base=args.seed_base,
            )
            records.append(record)
            print(f"{record['trajectory_id']} critical={len(record['selected_critical_states'])}", flush=True)
    critical = [row for record in records for row in record["selected_critical_states"]]
    with (artifact / "critical_state_manifest.jsonl").open("x") as stream:
        for row in critical:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    counts = {suite: sum(row["suite"] == suite for row in critical) for suite in SUITES}
    report = {
        "clean_trajectories": len(records), "critical_states": len(critical),
        "critical_states_by_suite": counts,
        "critical_states_by_phase": {
            phase: sum(row["phase"] == phase for row in critical)
            for phase in sorted({row["phase"] for row in critical})
        },
        "selection_used_intervention_results": False,
        "candidate_minimum_met": len(critical) >= 6 and all(count >= 2 for count in counts.values()),
    }
    write_json(artifact / "critical_extraction_gate.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
