#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_strong_weak_screening.sampler import ARMS, sample_strong_weak_actions


def array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def prepared_sha256(batch) -> str:
    values = [
        *batch["images"],
        *batch["image_masks"],
        batch["lang_tokens"],
        batch["lang_masks"],
        batch["state"],
    ]
    return hashlib.sha256(
        "".join(tensor_sha256(value) for value in values).encode("ascii")
    ).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def shard_rows(manifest, shard_index, shard_count):
    pairs = list(dict.fromkeys(row["pair_id"] for row in manifest))
    selected = {pair for index, pair in enumerate(pairs) if index % shard_count == shard_index}
    return [row for row in manifest if row["pair_id"] in selected]


def existing_identities(artifact: Path, rows):
    identities = {}
    for spec in rows:
        path = artifact / "episodes" / f"{spec['episode_id']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        identity = (record["initial_sim_state_sha256"], record["initial_prepared_input_sha256"])
        previous = identities.setdefault(spec["pair_id"], identity)
        if previous != identity:
            raise RuntimeError(f"existing paired identity mismatch: {spec['pair_id']}")
    return identities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if not json.loads((artifact / "integrity_report.json").read_text())["pass"]:
        raise RuntimeError("dry-run integrity must pass before rollout")
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    manifest = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines()]
    if len(manifest) != protocol["total_episodes"] or set(row["arm"] for row in manifest) != set(ARMS):
        raise RuntimeError("manifest contract mismatch")
    rows = shard_rows(manifest, args.shard_index, args.shard_count)
    identities = existing_identities(artifact, rows)
    initial_observations = {}
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)

    for ordinal, spec in enumerate(rows, 1):
        output = artifact / "episodes" / f"{spec['episode_id']}.json"
        if output.exists():
            continue
        env, env_preprocessor, env_postprocessor = make_task_env(
            spec["suite"], spec["task_id"], config
        )
        actions, queue, traces, latencies, boundaries, noise_hashes = [], [], [], [], [], []
        replans, success = 0, False
        try:
            inner = env.envs[0]
            inner.init_state_id = spec["init_state_id"]
            observation, _ = env.reset(seed=spec["reset_seed"])
            initial_state_hash = array_sha256(np.asarray(inner._env.get_sim_state()))
            observation = copy.deepcopy(
                initial_observations.setdefault(spec["pair_id"], copy.deepcopy(observation))
            )
            initial_prepared_hash = None
            torch.cuda.reset_peak_memory_stats()
            for _control_step in range(protocol["runtime"]["maximum_control_steps"]):
                if not queue:
                    batch = prepare(
                        policy, preprocessor, env_preprocessor, observation, spec["language"]
                    )
                    if initial_prepared_hash is None:
                        initial_prepared_hash = prepared_sha256(batch)
                        identity = (initial_state_hash, initial_prepared_hash)
                        expected = identities.setdefault(spec["pair_id"], identity)
                        if identity != expected:
                            raise RuntimeError(f"paired input mismatch: {spec['pair_id']}")
                    noise = torch.randn(
                        (1, config.chunk_size, config.max_action_dim),
                        generator=torch.Generator(device=batch["state"].device).manual_seed(
                            spec["action_noise_seed"] + replans
                        ),
                        device=batch["state"].device,
                        dtype=batch["state"].dtype,
                    )
                    noise_hashes.append(tensor_sha256(noise))
                    start = time.perf_counter()
                    chunk, trace = sample_strong_weak_actions(
                        policy.model,
                        batch["images"],
                        batch["image_masks"],
                        batch["lang_tokens"],
                        batch["lang_masks"],
                        batch["state"],
                        noise,
                        arm=spec["arm"],
                    )
                    torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - start)
                    prefix = chunk[:, :10, :7].transpose(0, 1)
                    if actions:
                        boundaries.append(
                            float(torch.linalg.vector_norm(prefix[0, 0].cpu() - actions[-1]))
                        )
                    queue.extend(prefix)
                    traces.append({"replan": replans, **trace})
                    replans += 1
                model_action = queue.pop(0)
                physical = postprocessor(model_action)
                legal = env_postprocessor({"action": physical})["action"]
                observation, _, terminated, _, info = env.step(
                    legal.detach().cpu().numpy()
                )
                actions.append(model_action[0].detach().float().cpu())
                success = bool(vector_info_value(info, "is_success"))
                if bool(terminated[0]) or success:
                    break
        finally:
            env.close()
        action_array = torch.stack(actions)
        differences = torch.linalg.vector_norm(action_array[1:] - action_array[:-1], dim=1)
        record = {
            **spec,
            "status": "complete",
            "success": success,
            "terminal_failure_or_timeout": not success,
            "control_steps": len(actions),
            "replans": replans,
            "initial_sim_state_sha256": initial_state_hash,
            "initial_prepared_input_sha256": initial_prepared_hash,
            "noise_sha256_by_replan": noise_hashes,
            "action_total_variation": float(differences.sum()) if len(differences) else 0.0,
            "action_total_variation_first_30_steps": float(differences[:29].sum())
            if len(differences)
            else 0.0,
            "chunk_discontinuity": float(np.mean(boundaries)) if boundaries else 0.0,
            "median_replan_latency_seconds": float(np.median(latencies)),
            "all_actions_finite": bool(torch.isfinite(action_array).all()),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "replan_traces": traces,
        }
        numeric = (
            "action_total_variation",
            "action_total_variation_first_30_steps",
            "chunk_discontinuity",
            "median_replan_latency_seconds",
        )
        if not record["all_actions_finite"] or not all(math.isfinite(record[key]) for key in numeric):
            raise RuntimeError("episode numeric integrity failure")
        atomic_json(output, record)
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "ordinal": ordinal,
                    "episode": spec["episode_id"],
                    "success": success,
                    "steps": len(actions),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
