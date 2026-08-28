#!/usr/bin/env python3
"""Run resumable append-only matched clean/masked rollouts from locked snapshots."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_causal_dataset.common import canonical_sha256
from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors, prepare
from research.coreact_region.audit_effect_candidates import restore
from research.coreact_region.effect_existence import array_sha256
from research.coreact_region.fixed_mask_sampler import prepare_ranked_prefix, sample_fixed_mask_actions
from research.coreact_region.segmented_runtime import (
    batched_observation,
    make_segmented_env,
    progress_snapshot,
    step_without_autoreset,
)


MAX_STEPS = 280
EXECUTED_PREFIX = 10


def run_one(artifact, config, policy, preprocessor, postprocessor, means, spec):
    episode_dir = artifact / "episodes" / spec["episode_id"]
    record_path = episode_dir / "episode.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    episode_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((artifact / spec["candidate_audit_path"]).read_text())
    if not audit["valid"]:
        raise RuntimeError("candidate audit is not valid")
    env = make_segmented_env(spec["suite"], spec["task_id"])
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config(spec["suite"], spec["task_id"]), policy_cfg=config
    )
    try:
        env.init_state_id = spec["init_state_id"]
        env.reset(seed=spec["reset_seed"])
        state = np.load(artifact / spec["state_path"], allow_pickle=False)
        observation = restore(env, state)
        initial_hash = array_sha256(np.asarray(env._env.get_sim_state()))
        if initial_hash != spec["sim_state_sha256"]:
            raise RuntimeError("simulator snapshot restoration mismatch")
        initial_progress = progress_snapshot(env)
        if initial_progress.predicate:
            raise RuntimeError("snapshot already satisfies task predicate")
        queue, action_rows, noise_hashes, latencies = [], [], [], []
        replans, success, first_trace, initial_prefix_hash = 0, False, None, None
        first_chunk_hash = None
        for control_step in range(MAX_STEPS):
            if not queue:
                prepared = prepare(
                    policy, preprocessor, env_preprocessor, batched_observation(observation), spec["language"]
                )
                generator = torch.Generator(device=prepared["state"].device).manual_seed(
                    spec["rollout_seed"] * 1000 + replans
                )
                noise = torch.randn(
                    (1, config.chunk_size, config.max_action_dim), generator=generator,
                    device=prepared["state"].device, dtype=prepared["state"].dtype,
                )
                noise_hashes.append(tensor_sha256(noise))
                started = time.perf_counter()
                with torch.inference_mode():
                    if replans == 0:
                        ranked = prepare_ranked_prefix(
                            policy.model, prepared["images"], prepared["image_masks"],
                            prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise,
                        )
                        initial_prefix_hash = tensor_sha256(ranked["prefix"])
                        if initial_prefix_hash != audit["native_prefix_sha256"]:
                            raise RuntimeError("native prefix changed since candidate lock")
                        if spec["condition"] == "masked" and replans < int(spec.get("duration", 1)):
                            chunk, first_trace = sample_fixed_mask_actions(
                                policy.model, ranked, noise, means, spec["token_indices"]
                            )
                            if first_trace["changed_indices"] != spec["token_indices"]:
                                raise RuntimeError("changed indices do not match locked group")
                        else:
                            chunk = policy.model.sample_actions(
                                prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                                prepared["lang_masks"], prepared["state"], noise=noise,
                            )
                        first_chunk_hash = tensor_sha256(chunk)
                    else:
                        if spec["condition"] == "masked" and replans < int(spec.get("duration", 1)):
                            ranked_now = prepare_ranked_prefix(
                                policy.model, prepared["images"], prepared["image_masks"],
                                prepared["lang_tokens"], prepared["lang_masks"], prepared["state"], noise,
                            )
                            chunk, trace_now = sample_fixed_mask_actions(
                                policy.model, ranked_now, noise, means, spec["token_indices"]
                            )
                            if first_trace is None:
                                first_trace = trace_now
                        else:
                            chunk = policy.model.sample_actions(
                                prepared["images"], prepared["image_masks"], prepared["lang_tokens"],
                                prepared["lang_masks"], prepared["state"], noise=noise,
                            )
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - started)
                if not bool(torch.isfinite(chunk).all()):
                    raise RuntimeError("nonfinite action chunk")
                queue.extend(chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1))
                replans += 1
            model_action = queue.pop(0)
            action_rows.append(model_action[0].detach().float().cpu().tolist())
            physical = postprocessor(model_action)
            legal = env_postprocessor({"action": physical})["action"][0].detach().cpu().numpy()
            observation, after = step_without_autoreset(env, legal)
            success = bool(after.predicate)
            if success:
                break
        action_array = np.asarray(action_rows, dtype=np.float32)
        record = {
            **spec, "status": "complete", "success": success, "control_steps": len(action_rows),
            "replans": replans, "initial_sim_state_sha256": initial_hash,
            "initial_prefix_sha256": initial_prefix_hash, "first_chunk_sha256": first_chunk_hash,
            "noise_sha256_by_replan": noise_hashes, "actions_sha256": array_sha256(action_array),
            "all_outputs_finite": bool(np.isfinite(action_array).all()),
            "masked_replans": list(range(int(spec.get("duration", 1))))
            if spec["condition"] == "masked" else [],
            "mask_trace": first_trace, "initial_progress": initial_progress.__dict__,
            "final_progress": progress_snapshot(env).__dict__,
            "policy_seconds_median": float(np.median(latencies)),
            "recent_action_history_sha256": spec["selected_boundary"]["recent_actions_sha256"],
        }
        if not record["all_outputs_finite"] or not math.isfinite(record["policy_seconds_median"]):
            raise RuntimeError("nonfinite rollout diagnostics")
        write_json(record_path, record)
        return record
    except Exception as exc:
        failure = {**spec, "status": "failed", "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc()}
        write_json(episode_dir / "failure.json", failure)
        raise
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
    gate = json.loads((artifact / "integrity_gate.json").read_text())
    if not gate.get("pass"):
        raise RuntimeError("integrity gate did not pass")
    rows = [row for i, row in enumerate(read_jsonl(artifact / "rollout_manifest.jsonl"))
            if i % args.shard_count == args.shard_index]
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",
                       weights_only=True, map_location="cpu")["visual_position_mean"]
    for ordinal, spec in enumerate(rows, 1):
        try:
            record = run_one(artifact, config, policy, preprocessor, postprocessor, means, spec)
            print(f"[{args.shard_index}] {ordinal}/{len(rows)} {spec['episode_id']} "
                  f"success={record['success']} steps={record['control_steps']}", flush=True)
        except Exception as exc:
            print(f"[{args.shard_index}] FAILED {spec['episode_id']}: {exc}", flush=True)


if __name__ == "__main__":
    main()
