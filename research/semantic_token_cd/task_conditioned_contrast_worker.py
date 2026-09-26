#!/usr/bin/env python3
"""Formal 960-episode task-conditioned contrast + APC worker."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.task_conditioned_contrast_protocol import (
    APC_BETA,
    ARTIFACT,
    ATTENTION_LAYERS,
    CANONICAL,
    CONTROL_INSTRUCTIONS,
    KMEANS_K,
    KMEANS_SEED,
    LAMBDA0,
    PCD_SOURCE,
    PROTOCOL,
    TASKS,
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_policy(base, task: str, arm: str):
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
    from research.semantic_token_cd.task_conditioned_contrast_policy import TaskConditionedContrastInference

    policy = copy.copy(base)
    policy.__class__ = TaskConditionedContrastInference
    _init_common(policy, LAMBDA0)
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = ATTENTION_LAYERS
    policy.selection_count = None
    policy.selection_top_p = None
    policy.selection_budget_schedule = None
    policy.selection_budget_scale = 1.0
    policy.selection_budget_label = None
    policy.selection_budget_source = "matched"
    policy.selection_budget_entities = None
    policy.save_prompt_attention = False
    policy.arm = arm
    policy.apc_beta = APC_BETA
    policy.control_instruction = CONTROL_INSTRUCTIONS[task]
    policy.task_key = task
    policy.save_condition_logits = True
    return policy


def claim_job(db: Path, worker_id: str) -> dict | None:
    con = sqlite3.connect(str(db), timeout=60)
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT job_id, task, seed, arm, attempt FROM jobs "
            "WHERE status='pending' ORDER BY job_id LIMIT 1"
        ).fetchone()
        if row is None:
            con.commit()
            return None
        job_id, task, seed, arm, attempt = row
        con.execute(
            "UPDATE jobs SET status='running', worker=?, heartbeat=?, start_time=?, attempt=? WHERE job_id=?",
            (worker_id, int(time.time()), int(time.time()), int(attempt) + 1, job_id),
        )
        con.commit()
        return {"job_id": job_id, "task": task, "seed": int(seed), "arm": arm, "attempt": int(attempt) + 1}
    finally:
        con.close()


def claim_task_job(db: Path, worker_id: str, task: str) -> dict | None:
    con = sqlite3.connect(str(db), timeout=60)
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT job_id, task, seed, arm, attempt FROM jobs "
            "WHERE status='pending' AND task=? ORDER BY job_id LIMIT 1",
            (task,),
        ).fetchone()
        if row is None:
            con.commit()
            return None
        job_id, task, seed, arm, attempt = row
        con.execute(
            "UPDATE jobs SET status='running', worker=?, heartbeat=?, start_time=?, attempt=? WHERE job_id=?",
            (worker_id, int(time.time()), int(time.time()), int(attempt) + 1, job_id),
        )
        con.commit()
        return {"job_id": job_id, "task": task, "seed": int(seed), "arm": arm, "attempt": int(attempt) + 1}
    finally:
        con.close()


def finish_job(db: Path, job_id: str, status: str, error: str | None = None) -> None:
    con = sqlite3.connect(str(db), timeout=60)
    try:
        con.execute(
            "UPDATE jobs SET status=?, finish_time=?, error=?, heartbeat=? WHERE job_id=?",
            (status, int(time.time()), error, int(time.time()), job_id),
        )
        con.commit()
    finally:
        con.close()


def run_episode(env, policy, instruction, obs, trace_path: Path):
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize
    from research.semantic_token_cd.rollout_pilot import flatten_action

    predicted_terminated = truncated = False
    timestep = 0
    step_infos = []
    executed_actions = []
    image = get_image_from_maniskill2_obs_dict(env, obs)
    trace_file = trace_path.open("w")
    try:
        while not (predicted_terminated or truncated):
            _raw_action, actions, _aux = policy.step(
                image, None, instruction, proprio=obs["agent"]["eef_pos"]
            )
            trace_file.write(json.dumps(jsonable(policy._episode_trace[-1]), sort_keys=True) + "\n")
            if not isinstance(actions, list):
                actions = [actions]
            for action in actions:
                executed = flatten_action(action)
                if executed.shape != (7,) or not np.isfinite(executed).all():
                    raise FloatingPointError(f"invalid executed action: {executed}")
                executed_actions.append(executed.copy())
                obs, _reward, _success, truncated, info = env.step(executed)
                image = get_image_from_maniskill2_obs_dict(env, obs)
                timestep += 1
                step_infos.append(convert_numpy_or_torch_to_python(info))
                predicted_terminated = bool(action["terminate_episode"][0] > 0)
                if predicted_terminated and not env.unwrapped.is_final_subtask():
                    predicted_terminated = False
                    env.advance_to_next_subtask()
                instruction = env.unwrapped.get_language_instruction()
    finally:
        trace_file.close()
    result = summarize(step_infos)
    result.update(stat_first(step_infos))
    result.update(stat_final(step_infos))
    failure_reason = None if result["success"] else (
        "environment_time_limit" if truncated else "policy_terminated_without_success"
    )
    actions_array = np.asarray(executed_actions, dtype=np.float32)
    jitter = float(np.linalg.norm(np.diff(actions_array, axis=0), axis=1).mean()) if len(actions_array) > 1 else 0.0
    return result, timestep, failure_reason, actions_array, jitter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="pin this worker to one task to keep one model resident")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--jobs", type=Path, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    repo = Path("/home/leju-suzhou/zjt_ws/token-cd")
    for path in (str(repo / "task1/shim_site"), str(repo), str(PCD_SOURCE)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import restore_snapshot, snapshot_sha
    from research.semantic_token_cd.rollout_pilot import array_sha256
    from research.semantic_token_cd.spatial_grid_rollout import make_environment

    artifact = args.artifact.resolve()
    db = args.jobs or (artifact / "jobs.sqlite")
    envs = {}
    base_models = {}
    processed = 0
    while args.max_jobs <= 0 or processed < args.max_jobs:
        job = claim_task_job(db, args.worker_id, args.task) if args.task else claim_job(db, args.worker_id)
        if job is None:
            break
        task = job["task"]
        arm = job["arm"]
        seed = job["seed"]
        started = time.monotonic()
        try:
            if task not in envs:
                envs[task] = make_environment(task)
                checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
                config = get_policy_config("openvla", checkpoint, task, {}, False)
                base_models[task] = OpenVLAInference(**config)
            env, _environment_id = envs[task]
            snapshot_path = CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl"
            with snapshot_path.open("rb") as handle:
                snapshot = pickle.load(handle)
            canonical = snapshot_sha(snapshot)
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            instruction = env.unwrapped.get_language_instruction()
            policy = build_policy(base_models[task], task, arm)
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            trace_path = artifact / "traces" / task / arm / f"episode_{seed:03d}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs, trace_path)
            arrays_path = artifact / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"
            arrays_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(arrays_path, executed_actions=actions)
            summary = {
                "protocol_id": PROTOCOL,
                "task": task,
                "seed": seed,
                "arm": arm,
                "checkpoint": str(PCD_SOURCE / "pretrained/openvla-7b"),
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "instruction": instruction,
                "success": bool(result["success"]),
                "termination_reason": reason,
                "episode_steps": steps,
                "wall_clock_seconds": time.monotonic() - started,
                "worker": args.worker_id,
                "retry_count": job["attempt"] - 1,
                "trace_path": str(trace_path),
                "action_jitter_index": jitter,
                "result": jsonable(result),
                "lambda": LAMBDA0,
                "apc_beta": APC_BETA if arm.endswith("_APC") else None,
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "attention_layers": list(ATTENTION_LAYERS),
            }
            atomic_json(artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json", summary)
            finish_job(db, job["job_id"], "complete")
            print(json.dumps({"worker": args.worker_id, "task": task, "seed": seed, "arm": arm, "success": summary["success"], "seconds": round(summary["wall_clock_seconds"], 1)}), flush=True)
        except Exception as exc:
            import traceback
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            # Technical failures are retried up to two times and never counted
            # as successful episodes.
            status = "pending" if job["attempt"] < 2 else "error"
            finish_job(db, job["job_id"], status, error)
            print(json.dumps({"worker": args.worker_id, "task": task, "seed": seed, "arm": arm, "error": error}), flush=True)
        processed += 1


if __name__ == "__main__":
    main()
