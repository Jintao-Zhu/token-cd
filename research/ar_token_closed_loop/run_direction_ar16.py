from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import decode_action_ids, masked_action_token_ids
from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, predict_action, prepare_agentview, prepare_env_action, set_determinism
from .common import file_sha256, make_env, progress_dict, read_jsonl, write_json


CONDITIONS = {"vanilla", "top16_mask_only", "away", "toward"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    rows = [row for i, row in enumerate(read_jsonl(artifact / "rollout_manifest.lock.jsonl")) if i % args.shard_count == args.shard_index]
    unknown = {row["condition"] for row in rows} - CONDITIONS
    if unknown:
        raise RuntimeError(f"Unknown conditions: {sorted(unknown)}")
    mean = torch.load(workspace / "artifacts/ar_token_counterfactual_qualification_v1_20260808_231044/position_conditioned_visual_mean.pt", map_location="cpu", weights_only=True)["mean"]
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for row in rows:
        output = artifact / "episodes" / f"{row['episode_id']}.json"
        if output.exists():
            continue
        group_path = artifact / "groups" / f"{row['snapshot_id']}.json"
        if file_sha256(group_path) != row["candidate_file_sha256"]:
            raise RuntimeError("Group changed after manifest lock")
        record = json.loads((artifact / "snapshots" / row["snapshot_id"] / "record.json").read_text())
        state = np.load(artifact / record["sim_state_path"], allow_pickle=False)
        task = suite.get_task(row["task_id"])
        env = make_env(task, get_libero_path, OffScreenRenderEnv)
        actions, latencies = [], []
        intervention_calls = 0
        try:
            env.reset()
            obs = env.set_init_state(state)
            restored_hash = __import__("research.ar_token_closed_loop.common", fromlist=["array_sha256"]).array_sha256(np.asarray(env.get_sim_state()).copy())
            if restored_hash != record["sim_state_sha256"]:
                raise RuntimeError("State restoration mismatch")
            initial = progress_dict(env)
            done = bool(initial["predicate"])
            steps = 0
            while steps < record["maximum_remaining_steps"] and not done:
                _, image = prepare_agentview(obs)
                started = time.perf_counter()
                if row["condition"] == "vanilla":
                    raw = predict_action(model, processor, image, task.language)
                else:
                    inputs = processor(build_prompt(task.language), image).to(model.device, dtype=torch.bfloat16)
                    masked_ids, trace = masked_action_token_ids(model, inputs, row["visual_token_indices"], mean)
                    if trace.changed_indices != tuple(sorted(row["visual_token_indices"])):
                        raise RuntimeError("Changed-index mismatch")
                    masked = decode_action_ids(model, masked_ids)
                    intervention_calls += 1
                    if row["condition"] == "top16_mask_only":
                        raw = masked
                    elif row["condition"] in {"away", "toward"}:
                        clean = predict_action(model, processor, image, task.language)
                        delta = clean - masked
                        raw = clean + (0.5 * delta if row["condition"] == "away" else -0.5 * delta)
                    else:
                        raise RuntimeError(f"Unknown condition: {row['condition']}")
                latency = (time.perf_counter() - started) * 1000.0
                action = prepare_env_action(raw)
                if not np.isfinite(action).all():
                    raise RuntimeError("Nonfinite action")
                obs, _, done, _ = env.step(action.tolist())
                actions.append(action); latencies.append(latency); steps += 1
            final = progress_dict(env)
        finally:
            env.close()
        arr = np.asarray(actions, dtype=np.float64)
        result = {**row, "success": bool(done), "control_steps": steps, "initial_progress": initial, "final_progress": final, "intervention_calls": intervention_calls, "all_replans_masked": intervention_calls == (0 if row["condition"] == "vanilla" else steps), "all_actions_finite": bool(np.isfinite(arr).all()), "mean_inference_latency_ms": float(np.mean(latencies)) if latencies else 0.0, "action_total_variation": float(np.abs(np.diff(arr, axis=0)).sum()) if len(arr) > 1 else 0.0, "restored_state_sha256": restored_hash}
        if not result["all_replans_masked"] or not result["all_actions_finite"]:
            raise RuntimeError("Episode integrity failure")
        write_json(output, result)
        print(json.dumps({"shard": args.shard_index, "episode_id": row["episode_id"], "complete": True}), flush=True)


if __name__ == "__main__":
    main()
