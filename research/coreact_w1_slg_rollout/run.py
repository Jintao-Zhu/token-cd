from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import prepare
from research.coreact_self_guidance.reference_snapshot_gate import fingerprints
from research.coreact_trained_weak.run_libero10_quality_screen import env_config
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_w1_slg_rollout.sampler import ARMS, sample_w1_slg_actions


MAX_STEPS = 520
EXECUTED_PREFIX = 10


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_manifest(path: Path) -> dict[str, dict[str, dict]]:
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        pairs[row["pair_id"]][row["arm"]] = row
    if len(pairs) != 500 or any(set(group) != set(ARMS) for group in pairs.values()):
        raise RuntimeError("manifest coverage/arm contract failed")
    return dict(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--pair-id")
    parser.add_argument("--max-new-pairs", type=int)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    os.environ.setdefault("HF_HOME", str(workspace / "task1/.hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(workspace / "task1/.hf-cache/hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    protocol = json.loads((artifact / "protocol.lock.json").read_text())
    checkpoint = Path(protocol["strong"]["checkpoint"])
    config, policy, preprocessor, postprocessor = load_policy(checkpoint)
    pairs = read_manifest(artifact / "episode_manifest.jsonl")
    selected = [item for index, item in enumerate(sorted(pairs.items())) if index % args.shard_count == args.shard_index]
    new_pairs = 0

    for pair_id, specs in selected:
        if args.pair_id and pair_id != args.pair_id:
            continue
        outputs = {arm: artifact / "episodes" / f"{specs[arm]['episode_id']}.json" for arm in ARMS}
        if all(path.exists() for path in outputs.values()):
            continue
        if any(path.exists() for path in outputs.values()):
            raise RuntimeError(f"partial pair exists: {pair_id}")
        initial = {}
        records = {}
        canonical_observation = None
        for arm in ARMS:
            spec = specs[arm]
            cfg = env_config(spec["task_id"])
            env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=config)
            env = cfg.create_envs(n_envs=1, use_async_envs=False)["libero_10"][spec["task_id"]]
            actions, queue, replans, noise_hashes, replan_traces, latencies = [], [], 0, [], [], []
            success, termination_reason = False, "horizon"
            try:
                inner = env.envs[0]
                inner.init_state_id = spec["init_state_id"]
                real_observation, _ = env.reset(seed=spec["reset_seed"])
                language = inner.task_description
                real_batch = prepare(policy, preprocessor, env_pre, real_observation, language)
                reset_fingerprint = fingerprints(env, real_observation, real_batch, [])
                if canonical_observation is None:
                    canonical_observation = copy.deepcopy(real_observation)
                observation = copy.deepcopy(canonical_observation)
                first_policy_fingerprint = None
                for _ in range(MAX_STEPS):
                    if not queue:
                        batch = prepare(policy, preprocessor, env_pre, observation, language)
                        if first_policy_fingerprint is None:
                            first_policy_fingerprint = fingerprints(env, observation, batch, [])
                        generator = torch.Generator(device=batch["state"].device).manual_seed(
                            spec["flow_noise_seed_base"] + replans
                        )
                        noise = torch.randn(
                            (1, config.chunk_size, config.max_action_dim), generator=generator,
                            device=batch["state"].device, dtype=batch["state"].dtype,
                        )
                        noise_hashes.append(tensor_sha256(noise))
                        started = time.perf_counter()
                        chunk, trace = sample_w1_slg_actions(
                            policy.model, batch["images"], batch["image_masks"],
                            batch["lang_tokens"], batch["lang_masks"], batch["state"], noise,
                            arm=arm,
                        )
                        torch.cuda.synchronize()
                        latencies.append(time.perf_counter() - started)
                        if replans == 0:
                            initial[arm] = {
                                "reset": reset_fingerprint,
                                "policy_input": first_policy_fingerprint,
                                "instruction": language,
                                "noise_sha256": noise_hashes[0],
                                "strong_velocity_sha256": trace["initial_strong_velocity_sha256"],
                            }
                        replan_traces.append({
                            "replan": replans,
                            "initial_strong_velocity_sha256": trace["initial_strong_velocity_sha256"],
                            "active_steps": trace["active_steps"],
                            "clip_scales": [row["clip_scale"] for row in trace["step_traces"] if row["active"]],
                            "applied_correction_norms": [row["applied_correction_norm"] for row in trace["step_traces"] if row["active"]],
                        })
                        queue = [value.detach().cpu() for value in chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1)]
                        replans += 1
                    action = queue.pop(0)
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("nonfinite action")
                    physical = postprocessor(action)
                    legal = env_post({"action": physical})["action"]
                    observation, _, terminated, _, info = env.step(legal.detach().cpu().numpy())
                    actions.append(action[0].detach().float().cpu())
                    success = bool(vector_info_value(info, "is_success"))
                    if success:
                        termination_reason = "success"
                        break
                    if bool(terminated[0]):
                        termination_reason = "terminated"
                        break
            finally:
                env.close()
            finite = bool(actions and torch.isfinite(torch.stack(actions)).all())
            records[arm] = {
                **spec, "status": "complete", "success": success,
                "termination_reason": termination_reason, "control_steps": len(actions),
                "replans": replans, "all_actions_finite": finite,
                "maximum_control_steps": MAX_STEPS, "executed_action_prefix": EXECUTED_PREFIX,
                "noise_sha256_by_replan": noise_hashes, "replan_traces": replan_traces,
                "initial_integrity": initial[arm],
                "median_replan_latency_seconds": float(np.median(latencies)),
            }
            if not finite or not math.isfinite(records[arm]["median_replan_latency_seconds"]):
                raise RuntimeError(f"episode integrity failure: {spec['episode_id']}")

        strict_fields = (
            "simulator_state", "qpos", "qvel", "object_pose", "robot_observation",
            "camera1", "camera2", "full_observation", "preprocessing",
        )
        failures = []
        for section in ("reset", "policy_input"):
            for field in strict_fields:
                values = {initial[arm][section][field] for arm in ARMS}
                if len(values) != 1:
                    failures.append(f"{section}.{field}")
        for field in ("instruction", "noise_sha256", "strong_velocity_sha256"):
            if len({initial[arm][field] for arm in ARMS}) != 1:
                failures.append(field)
        if failures:
            atomic_json(artifact / "invalid_pairs" / f"{pair_id}.json", {
                "pair_id": pair_id, "status": "invalid_not_rerun", "mismatches": failures,
                "initial": initial,
            })
            raise RuntimeError(f"paired integrity failure {pair_id}: {failures}")
        for arm in ARMS:
            records[arm]["paired_initial_integrity"] = "PASS"
            atomic_json(outputs[arm], records[arm])
        new_pairs += 1
        print(json.dumps({"pair_id": pair_id, "new_pairs": new_pairs, "success": {arm: records[arm]["success"] for arm in ARMS}}), flush=True)
        if args.max_new_pairs is not None and new_pairs >= args.max_new_pairs:
            break


if __name__ == "__main__":
    main()
