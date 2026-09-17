"""Collect held-out seeds 100-199 after task routers have been frozen."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.collect_l11_initial_state_features import extract_features
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    get_image_from_maniskill2_obs_dict,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.l11_initial_state_router_protocol import (
    PROTOCOL,
    TASKS,
    TEST_SEEDS,
    atomic_json,
    feature_path,
    metadata_path,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import load_reference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common


def parse_seeds(specification: str) -> list[int]:
    seeds = []
    for part in specification.split(","):
        part = part.strip()
        if "-" in part:
            low, high = map(int, part.split("-", 1))
            seeds.extend(range(low, high + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(seed not in TEST_SEEDS for seed in result):
        raise ValueError("confirmation seeds must be within 100..199")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    frozen = artifact / "FROZEN_MODELS.json"
    if not frozen.exists():
        raise RuntimeError("models must be frozen before held-out feature collection")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    canonical = args.canonical.resolve()
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = OpenVLAInference(**config)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, 0.5)
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[args.task]
    policy.attention_layers = (11,)

    for seed in parse_seeds(args.seeds):
        output = feature_path(artifact, args.task, seed)
        metadata_output = metadata_path(artifact, args.task, seed)
        if output.exists() and metadata_output.exists():
            print(json.dumps({"skip": True, "task": args.task, "seed": seed}), flush=True)
            continue
        snapshot_file = canonical / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
        with snapshot_file.open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(canonical, args.task, seed)
        if snapshot_sha(snapshot) != reference["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if (state_sha, rgb_sha) != (
            reference["initial_state_sha256"], reference["initial_rgb_sha256"]
        ):
            raise RuntimeError("restored snapshot mismatch")
        instruction = env.unwrapped.get_language_instruction()
        if instruction != reference["instruction"]:
            raise RuntimeError("instruction mismatch")
        policy.reset(instruction, seed=seed)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        started = time.monotonic()
        arrays, metadata = extract_features(
            policy, image, instruction, np.asarray(obs["agent"]["eef_pos"], dtype=np.float32)
        )
        runtime = time.monotonic() - started
        metadata.update({
            "protocol_id": PROTOCOL,
            "task": args.task,
            "seed": seed,
            "environment_id": environment_id,
            "worker_id": args.worker_id,
            "gpu_id": args.gpu,
            "runtime_seconds": runtime,
            "canonical_snapshot_sha256": reference["canonical_snapshot_sha256"],
            "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
        atomic_json(metadata_output, metadata)
        print(json.dumps({"task": args.task, "seed": seed, "runtime_seconds": round(runtime, 3)}), flush=True)


if __name__ == "__main__":
    main()
