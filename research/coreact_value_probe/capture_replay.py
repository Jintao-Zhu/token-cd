#!/usr/bin/env python3
"""Replay logged vanilla actions and append frozen SmolVLA value representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_value_probe.representation import extract_value_representation


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def prepared_sha256(prepared: dict) -> str:
    values = [
        *prepared["images"],
        *prepared["image_masks"],
        prepared["lang_tokens"],
        prepared["lang_masks"],
        prepared["state"],
    ]
    return hashlib.sha256("".join(tensor_sha256(value) for value in values).encode("ascii")).hexdigest()


def progress(inner) -> tuple[float, float, bool, bool]:
    # LiberoEnv wraps the robosuite ControlEnv, which in turn wraps the task env.
    raw = inner._env.env
    goals = raw.parsed_problem["goal_state"]
    if len(goals) != 1 or len(goals[0]) != 3 or goals[0][0] not in ("in", "on"):
        raise RuntimeError(f"unsupported task goal {goals}")
    goal = goals[0]
    object_name, goal_name = goal[1], goal[2]
    object_position = raw.object_states_dict[object_name].get_geom_state()["pos"]
    goal_position = raw.object_states_dict[goal_name].get_geom_state()["pos"]
    eef_position = raw.sim.data.site_xpos[raw.robots[0].eef_site_id]
    grasped = raw._check_grasp(raw.robots[0].gripper, raw.get_object(object_name))
    predicate = raw._eval_predicate(goal)
    return (
        float(np.linalg.norm(eef_position - object_position)),
        float(np.linalg.norm(object_position - goal_position)),
        bool(grasped),
        bool(predicate),
    )


def phase_name(control_step: int, eef_distance: float, grasped: bool, predicate: bool, first_grasp: int | None) -> str:
    if first_grasp is None:
        return "pre_grasp" if eef_distance <= 0.08 else "approach"
    if grasped and control_step - first_grasp < 10:
        return "grasp"
    if not predicate:
        return "transport"
    return "completed"


def replay_one(workspace: Path, artifact: Path, config, policy, preprocessor, spec: dict) -> dict:
    output = artifact / "captures" / f"{spec['capture_id']}.npz"
    audit_path = artifact / "captures" / f"{spec['capture_id']}.json"
    if output.exists() and audit_path.exists():
        return json.loads(audit_path.read_text())
    steps = read_jsonl(workspace / spec["source_steps"])
    if len(steps) != spec["expected_control_steps"]:
        raise RuntimeError(f"source step count mismatch for {spec['capture_id']}")
    env, env_preprocessor, _ = make_task_env(spec["suite"], spec["task_id"], config)
    representations, metadata = [], []
    first_grasp = None
    success = False
    try:
        inner = env.envs[0]
        inner.init_state_id = spec["init_state_id"]
        observation = None
        for _ in range(2):
            observation, _ = env.reset(seed=spec["reset_seed"])
        assert observation is not None
        initial_sim_hash = array_sha256(np.asarray(inner._env.get_sim_state()))
        if initial_sim_hash != spec["expected_initial_sim_state_sha256"]:
            raise RuntimeError(f"initial simulator hash mismatch for {spec['capture_id']}")
        previous_replan = None
        initial_prepared_hash = None
        for row in steps:
            control_step, replan = int(row["control_step"]), int(row["replan"])
            if replan != previous_replan:
                prepared = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
                current_hash = prepared_sha256(prepared)
                if initial_prepared_hash is None:
                    initial_prepared_hash = current_hash
                representation, pool_indices = extract_value_representation(policy.model, prepared)
                eef_distance, object_goal_distance, grasped, predicate = progress(inner)
                if grasped and first_grasp is None:
                    first_grasp = control_step
                metadata.append({
                    "control_step": control_step,
                    "replan": replan,
                    "phase": phase_name(control_step, eef_distance, grasped, predicate, first_grasp),
                    "eef_object_distance": eef_distance,
                    "object_goal_distance": object_goal_distance,
                    "grasped": grasped,
                    "predicate": predicate,
                })
                representations.append(representation[0].detach().cpu().numpy())
                previous_replan = replan
            observation, _, terminated, _, info = env.step(np.asarray([row["legal_action"]], dtype=np.float32))
            success = bool(vector_info_value(info, "is_success"))
            if bool(terminated[0]) or success:
                break
        if success != bool(spec["expected_success"]):
            raise RuntimeError(f"terminal success mismatch for {spec['capture_id']}: {success}")
        if len(representations) != max(row["replan"] for row in steps) + 1:
            raise RuntimeError(f"replan count mismatch for {spec['capture_id']}")
        array = np.stack(representations).astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise RuntimeError(f"nonfinite representation for {spec['capture_id']}")
        np.savez_compressed(
            output,
            representation=array,
            control_step=np.asarray([row["control_step"] for row in metadata], dtype=np.int32),
            phase=np.asarray([row["phase"] for row in metadata]),
            eef_object_distance=np.asarray([row["eef_object_distance"] for row in metadata], dtype=np.float32),
            object_goal_distance=np.asarray([row["object_goal_distance"] for row in metadata], dtype=np.float32),
            grasped=np.asarray([row["grasped"] for row in metadata], dtype=np.bool_),
            label=np.full(len(metadata), bool(spec["expected_success"]), dtype=np.bool_),
        )
        record = {
            **{key: spec[key] for key in ("capture_id", "task_id", "init_state_id", "split")},
            "status": "complete",
            "states": len(metadata),
            "representation_shape": list(array.shape),
            "representation_sha256": array_sha256(array),
            "initial_sim_state_sha256": initial_sim_hash,
            "initial_prepared_sha256": initial_prepared_hash,
            "expected_initial_prepared_sha256": spec["expected_initial_prepared_sha256"],
            "initial_prepared_hash_match": initial_prepared_hash == spec["expected_initial_prepared_sha256"],
            "pool_indices": pool_indices,
            "success": success,
            "phase_counts": {name: sum(row["phase"] == name for row in metadata) for name in sorted({row["phase"] for row in metadata})},
        }
        audit_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return record
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    specs = [row for index, row in enumerate(read_jsonl(artifact / "capture_manifest.jsonl")) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        specs = specs[: args.limit]
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    for ordinal, spec in enumerate(specs, 1):
        record = replay_one(workspace, artifact, config, policy, preprocessor, spec)
        print(f"[{args.shard_index}] {ordinal}/{len(specs)} {spec['capture_id']} states={record['states']} success={record['success']}", flush=True)


if __name__ == "__main__":
    main()
