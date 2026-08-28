"""Validate LIBERO simulator snapshot restore without running guided rollouts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.coreact_closed_loop.runtime import (
    load_policy_and_processors,
    make_task_env,
    prepare,
)


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def observation_hash(observation: dict) -> str:
    rows = []
    for key in sorted(observation):
        value = observation[key]
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            rows.append((key, str(value.dtype), tuple(value.shape), hashlib.sha256(value.tobytes()).hexdigest()))
        else:
            rows.append((key, repr(value)))
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def observation_field_hashes(observation: dict, prefix: str = "") -> dict[str, str]:
    result = {}
    for key, value in observation.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(observation_field_hashes(value, name))
        else:
            value = np.ascontiguousarray(value)
            result[name] = hashlib.sha256(value.tobytes()).hexdigest()
    return result


def prepared_hash(batch: dict) -> str:
    values = [*batch["images"], *batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"]]
    return hashlib.sha256("".join(tensor_hash(x) for x in values).encode()).hexdigest()


def restore_observation(inner, sim_state):
    # LiberoEnv exposes the underlying EnvWrapper as `_env`; the vector wrapper
    # itself intentionally has no public set_state method.
    inner._env.set_state(np.asarray(sim_state))
    inner._env.sim.forward()
    inner._env._post_process()
    inner._env._update_observables(force=True)
    raw_obs = inner._env.env._get_observations()
    formatted = inner._format_raw_obs(raw_obs)
    # `inner` is the scalar LiberoEnv inside a vector env; the outer reset adds
    # a leading environment dimension. Reproduce that wrapper contract here.
    def batch(value):
        if isinstance(value, dict):
            return {key: batch(item) for key, item in value.items()}
        return np.expand_dims(np.asarray(value), 0)
    return batch(formatted)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--task-id", type=int, default=4)
    p.add_argument("--init-state-id", type=int, default=0)
    p.add_argument("--language", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    workspace = args.workspace.resolve()
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    env, env_pre, env_post = make_task_env("libero_spatial", args.task_id, config)
    try:
        inner = env.envs[0]
        inner.init_state_id = args.init_state_id
        observation, _ = env.reset(seed=120000000 + args.task_id * 100000 + args.init_state_id * 10 + 1)
        original_state = np.asarray(inner._env.get_sim_state()).copy()
        original_obs = copy.deepcopy(observation)
        original_batch = prepare(policy, preprocessor, env_pre, original_obs, args.language)
        restored_obs = restore_observation(inner, original_state)
        restored_batch = prepare(policy, preprocessor, env_pre, restored_obs, args.language)
        state_a = np.ascontiguousarray(original_state)
        state_b = np.ascontiguousarray(np.asarray(inner._env.get_sim_state()))
        result = {
            "task_id": args.task_id,
            "init_state_id": args.init_state_id,
            "sim_state_hash_before": hashlib.sha256(state_a.tobytes()).hexdigest(),
            "sim_state_hash_after": hashlib.sha256(state_b.tobytes()).hexdigest(),
            "sim_state_equal": bool(np.array_equal(state_a, state_b)),
            "observation_hash_before": observation_hash(original_obs),
            "observation_hash_after": observation_hash(restored_obs),
            "observation_equal": observation_hash(original_obs) == observation_hash(restored_obs),
            "observation_field_hashes_before": observation_field_hashes(original_obs),
            "observation_field_hashes_after": observation_field_hashes(restored_obs),
            "prepared_hash_before": prepared_hash(original_batch),
            "prepared_hash_after": prepared_hash(restored_batch),
            "prepared_equal": prepared_hash(original_batch) == prepared_hash(restored_batch),
            "model_frozen_eval": bool(not policy.training and all(not x.requires_grad for x in policy.parameters())),
            "all_inputs_finite": bool(all(torch.isfinite(x).all() for x in [*original_batch["images"], original_batch["state"], original_batch["lang_tokens"].float()])),
        }
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
