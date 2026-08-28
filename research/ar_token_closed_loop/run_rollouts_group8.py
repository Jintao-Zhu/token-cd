from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import decode_action_ids, masked_action_token_ids
from research.ar_token_counterfactual.libero_runtime import (
    build_prompt,
    load_policy,
    predict_action,
    prepare_agentview,
    prepare_env_action,
    set_determinism,
)

from .common import array_sha256, file_sha256, make_env, progress_dict, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    integrity = json.loads((artifact / "pre_rollout_integrity.json").read_text())
    if integrity["status"] != "PASS":
        raise RuntimeError("Pre-rollout integrity gate did not pass")
    rows = [row for i, row in enumerate(read_jsonl(artifact / "rollout_manifest.lock.jsonl")) if i % args.shard_count == args.shard_index]
    source = workspace / "artifacts/ar_token_counterfactual_qualification_v1_20260808_231044"
    mean = torch.load(source / "position_conditioned_visual_mean.pt", map_location="cpu", weights_only=True)["mean"]
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for ordinal, row in enumerate(rows, 1):
        output = artifact / "episodes" / f"{row['episode_id']}.json"
        if output.exists():
            continue
        candidate_path = artifact / "groups" / f"{row['snapshot_id']}.json"
        if file_sha256(candidate_path) != row["candidate_file_sha256"]:
            raise RuntimeError(f"Candidate changed after lock: {row['snapshot_id']}")
        snapshot = json.loads((artifact / "snapshots" / row["snapshot_id"] / "record.json").read_text())
        state = np.load(artifact / snapshot["sim_state_path"], allow_pickle=False)
        task = suite.get_task(row["task_id"])
        env = make_env(task, get_libero_path, OffScreenRenderEnv)
        actions, latencies, intervention_calls = [], [], 0
        try:
            env.reset()
            obs = env.set_init_state(state)
            restored_hash = array_sha256(np.asarray(env.get_sim_state()).copy())
            if restored_hash != snapshot["sim_state_sha256"]:
                raise RuntimeError(f"State restoration mismatch: {row['episode_id']}")
            initial = progress_dict(env)
            done = bool(initial["predicate"])
            steps = 0
            while steps < snapshot["maximum_remaining_steps"] and not done:
                _, image = prepare_agentview(obs)
                started = time.perf_counter()
                if steps == 0 and row["condition"] != "vanilla":
                    inputs = processor(build_prompt(task.language), image).to(model.device, dtype=torch.bfloat16)
                    ids, trace = masked_action_token_ids(model, inputs, row["visual_token_indices"], mean)
                    if trace.changed_indices != tuple(sorted(row["visual_token_indices"])):
                        raise RuntimeError("Runtime changed-index mismatch")
                    raw_action = decode_action_ids(model, ids)
                    intervention_calls += 1
                else:
                    raw_action = predict_action(model, processor, image, task.language)
                latency_ms = (time.perf_counter() - started) * 1000.0
                action = prepare_env_action(raw_action)
                if not np.isfinite(action).all():
                    raise RuntimeError("Nonfinite action")
                obs, _, done, _ = env.step(action.tolist())
                actions.append(action)
                latencies.append(latency_ms)
                steps += 1
            final = progress_dict(env)
        finally:
            env.close()
        action_array = np.asarray(actions, dtype=np.float64)
        total_variation = float(np.abs(np.diff(action_array, axis=0)).sum()) if len(actions) > 1 else 0.0
        expected_calls = 0 if row["condition"] == "vanilla" else 1
        result = {
            **row,
            "success": bool(done),
            "control_steps": steps,
            "maximum_remaining_steps": snapshot["maximum_remaining_steps"],
            "initial_progress": initial,
            "final_progress": final,
            "intervention_calls": intervention_calls,
            "single_first_action_intervention_verified": intervention_calls == expected_calls,
            "action_total_variation": total_variation,
            "mean_inference_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "max_abs_env_action": float(np.abs(action_array).max()) if len(actions) else 0.0,
            "all_actions_finite": bool(np.isfinite(action_array).all()),
            "restored_state_sha256": restored_hash,
        }
        if not result["single_first_action_intervention_verified"] or not result["all_actions_finite"]:
            raise RuntimeError(f"Episode integrity failure: {row['episode_id']}")
        write_json(output, result)
        print(json.dumps({"shard": args.shard_index, "ordinal": ordinal, "episode_id": row["episode_id"], "complete": True}), flush=True)


if __name__ == "__main__":
    main()
