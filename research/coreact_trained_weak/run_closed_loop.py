from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import make_task_env, prepare
from research.coreact_trained_weak.runtime import load_locked_pair
from research.coreact_trained_weak.sampler import ARMS, sample_trained_weak_actions


MAX_STEPS = 280
EXECUTED_PREFIX = 10


def array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def prepared_sha256(batch) -> str:
    values = [*batch["images"], *batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"]]
    return hashlib.sha256("".join(tensor_sha256(value) for value in values).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    strength = yaml.safe_load((artifact / "guidance_strength.lock.yaml").read_text())
    lambda_value = float(strength["selected_lambda"])
    if strength["closed_loop_outcomes_used"] or lambda_value != 0.5:
        raise RuntimeError("invalid guidance strength lock")
    _, strong, weak = load_locked_pair(artifact)
    config, strong_policy, preprocessor, postprocessor = strong
    _, weak_policy, _, _ = weak
    output_dir = artifact / "trained_weak_closed_loop_episodes"
    output_dir.mkdir(exist_ok=True)

    ordinal = 0
    for task_id in args.task_ids:
        for init_state_id in range(10, 20):
            ordinal += 1
            unit_id = f"task{task_id:02d}__init{init_state_id:02d}"
            episode_id = f"{unit_id}__{args.arm}"
            output = output_dir / f"{episode_id}.json"
            if output.exists():
                continue
            reset_seed = 710_000_000 + task_id * 1000 + init_state_id * 10
            noise_seed = 720_000_000 + task_id * 1000 + init_state_id * 10
            env, env_preprocessor, env_postprocessor = make_task_env("libero_spatial", task_id, config)
            queue, actions, traces, latencies, noise_hashes = [], [], [], [], []
            success, replans = False, 0
            try:
                inner = env.envs[0]
                inner.init_state_id = init_state_id
                observation, _ = env.reset(seed=reset_seed)
                language = inner.task_description
                initial_state_hash = array_sha256(inner._env.get_sim_state())
                initial_prepared_hash = None
                for _ in range(MAX_STEPS):
                    if not queue:
                        batch = prepare(strong_policy, preprocessor, env_preprocessor, observation, language)
                        if initial_prepared_hash is None:
                            initial_prepared_hash = prepared_sha256(batch)
                        noise = torch.randn(
                            (1, config.chunk_size, config.max_action_dim),
                            generator=torch.Generator(device=batch["state"].device).manual_seed(noise_seed + replans),
                            device=batch["state"].device,
                            dtype=batch["state"].dtype,
                        )
                        noise_hashes.append(tensor_sha256(noise))
                        started = time.perf_counter()
                        chunk, trace = sample_trained_weak_actions(
                            strong_policy.model, weak_policy.model,
                            batch["images"], batch["image_masks"], batch["lang_tokens"],
                            batch["lang_masks"], batch["state"], noise,
                            arm=args.arm, lambda_value=lambda_value,
                        )
                        torch.cuda.synchronize()
                        latencies.append(time.perf_counter() - started)
                        queue.extend(chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1))
                        traces.append({"replan": replans, **trace})
                        replans += 1
                    action = queue.pop(0)
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("nonfinite action")
                    physical = postprocessor(action)
                    legal = env_postprocessor({"action": physical})["action"]
                    observation, _, terminated, _, info = env.step(legal.detach().cpu().numpy())
                    actions.append(action[0].detach().float().cpu())
                    success = bool(vector_info_value(info, "is_success"))
                    if success or bool(terminated[0]):
                        break
            finally:
                env.close()
            action_array = torch.stack(actions)
            record = {
                "episode_id": episode_id,
                "unit_id": unit_id,
                "arm": args.arm,
                "task_id": task_id,
                "init_state_id": init_state_id,
                "reset_seed": reset_seed,
                "noise_seed": noise_seed,
                "language": language,
                "status": "complete",
                "success": success,
                "control_steps": len(actions),
                "replans": replans,
                "initial_sim_state_sha256": initial_state_hash,
                "initial_prepared_input_sha256": initial_prepared_hash,
                "noise_sha256_by_replan": noise_hashes,
                "all_actions_finite": bool(torch.isfinite(action_array).all()),
                "median_replan_latency_seconds": float(np.median(latencies)),
                "lambda": lambda_value if args.arm == "G" else None,
                "replan_traces": traces,
            }
            if not record["all_actions_finite"] or not math.isfinite(record["median_replan_latency_seconds"]):
                raise RuntimeError("episode integrity failure")
            atomic_json(output, record)
            print(json.dumps({"arm": args.arm, "ordinal": ordinal, "task": task_id, "init": init_state_id, "success": success, "steps": len(actions)}), flush=True)


if __name__ == "__main__":
    main()
