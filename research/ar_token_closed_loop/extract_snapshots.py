from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.ar_token_counterfactual.libero_runtime import load_policy, predict_action, prepare_agentview, prepare_env_action, set_determinism

from .common import array_sha256, make_env, progress_dict, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    plans = [row for index, row in enumerate(read_jsonl(artifact / "snapshot_plan.jsonl")) if index % args.shard_count == args.shard_index]
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for ordinal, plan in enumerate(plans, 1):
        output = artifact / "snapshots" / plan["snapshot_id"]
        record_path = output / "record.json"
        if record_path.exists():
            continue
        task = suite.get_task(plan["task_id"])
        env = make_env(task, get_libero_path, OffScreenRenderEnv)
        states, progresses = [], []
        try:
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(plan["task_id"])[plan["init_state_index"]])
            for _ in range(10):
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            steps = 0
            while steps < 220 and not done:
                states.append(np.asarray(env.get_sim_state()).copy())
                progresses.append(progress_dict(env))
                _, image = prepare_agentview(obs)
                action = prepare_env_action(predict_action(model, processor, image, task.language))
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
        finally:
            env.close()
        if not states:
            raise RuntimeError(f"No snapshot candidates for {plan['snapshot_id']}")
        selected_index = int(round(plan["target_fraction"] * (len(states) - 1)))
        output.mkdir(parents=True, exist_ok=True)
        state_path = output / "state.npy"
        np.save(state_path, states[selected_index], allow_pickle=False)
        record = {
            **plan,
            "task_description": task.language,
            "source_trajectory_policy_steps": len(states),
            "source_trajectory_success": bool(done),
            "selected_control_step": selected_index,
            "selected_fraction_realized": selected_index / max(len(states) - 1, 1),
            "sim_state_path": str(state_path.relative_to(artifact)),
            "sim_state_sha256": array_sha256(states[selected_index]),
            "initial_progress": progresses[selected_index],
            "maximum_remaining_steps": max(1, 220 - selected_index),
        }
        write_json(record_path, record)
        print(json.dumps({"shard": args.shard_index, "ordinal": ordinal, "snapshot_id": plan["snapshot_id"], "complete": True}), flush=True)


if __name__ == "__main__":
    main()
