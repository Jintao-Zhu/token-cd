"""Collect frozen OpenVLA features on canonical initial states."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    get_image_from_maniskill2_obs_dict,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.l11_state_probe_protocol import (
    PROTOCOL,
    TASKS,
    atomic_json,
    feature_path,
    metadata_path,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy import (
    N_VISUAL,
    PromptAttentionSHRInference,
    prompt_query_layout,
)
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
    if not result or min(result) < 0 or max(result) > 99:
        raise ValueError("state-probe seeds must be within 0..99")
    return result


@torch.inference_mode()
def extract_features(policy, image: np.ndarray, instruction: str, proprio: np.ndarray) -> tuple[dict, dict]:
    inputs = policy.process_inputs(image, task_description=instruction)
    tokenizer = policy.processor.tokenizer
    text_indices, query_positions, query_token_ids = prompt_query_layout(
        inputs["input_ids"], set(int(value) for value in tokenizer.all_special_ids)
    )
    with projector_intervention(policy.vla) as trace:
        output = policy.vla(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    if trace.before is None:
        raise RuntimeError("projector features were not captured")
    visual = trace.before[0].detach().float().cpu()
    if visual.shape[0] != N_VISUAL or not torch.isfinite(visual).all():
        raise RuntimeError(f"invalid visual features: {tuple(visual.shape)}")
    if output.hidden_states is None or len(output.hidden_states) != 33:
        raise RuntimeError("expected embedding plus 32 language hidden-state tensors")
    hidden = output.hidden_states[-1][0].detach().float().cpu()
    attention = output.attentions[11][0, :, query_positions, 1 : 1 + N_VISUAL]
    attention = attention.detach().float().cpu().mean(dim=(0, 1))
    attention = attention / attention.sum().clamp_min(1e-12)
    prompt_hidden = hidden[query_positions]
    action_context_index = int(hidden.shape[0] - 1)

    arrays = {
        "proprio": np.asarray(proprio, dtype=np.float32).reshape(-1),
        "visual_mean": visual.mean(dim=0).numpy().astype(np.float16),
        "visual_std": visual.std(dim=0, unbiased=False).numpy().astype(np.float16),
        "visual_l11_weighted": (attention[:, None] * visual).sum(dim=0).numpy().astype(np.float16),
        "prompt_hidden_mean": prompt_hidden.mean(dim=0).numpy().astype(np.float16),
        "action_context_hidden": hidden[action_context_index].numpy().astype(np.float16),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("non-finite state-probe feature")
    metadata = {
        "protocol_id": PROTOCOL,
        "instruction": instruction,
        "prompt_text_indices": text_indices,
        "prompt_multimodal_query_indices": query_positions,
        "prompt_query_token_ids": query_token_ids,
        "action_context_index": action_context_index,
        "visual_shape": list(visual.shape),
        "hidden_shape": list(hidden.shape),
        "l11_attention_mass_before_visual_normalization": float(
            output.attentions[11][0, :, query_positions, 1 : 1 + N_VISUAL]
            .detach().float().mean(dim=(0, 1)).sum().item()
        ),
        "feature_sha256": {
            name: hashlib.sha256(value.tobytes()).hexdigest()
            for name, value in arrays.items()
        },
    }
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    canonical = args.canonical.resolve()
    seeds = parse_seeds(args.seeds)
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = OpenVLAInference(**config)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, 0.5)
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[args.task]
    policy.attention_layers = (11,)

    for seed in seeds:
        output_path = feature_path(artifact, args.task, seed)
        meta_path = metadata_path(artifact, args.task, seed)
        if output_path.exists() and meta_path.exists():
            print(json.dumps({"skip": True, "task": args.task, "seed": seed}), flush=True)
            continue
        snapshot_path = canonical / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(canonical, args.task, seed)
        if snapshot_sha(snapshot) != reference["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if state_sha != reference["initial_state_sha256"] or rgb_sha != reference["initial_rgb_sha256"]:
            raise RuntimeError("restored snapshot mismatch")
        instruction = env.unwrapped.get_language_instruction()
        if instruction != reference["instruction"]:
            raise RuntimeError("instruction mismatch")
        policy.reset(instruction, seed=seed)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        policy.reset(instruction)
        started = time.monotonic()
        arrays, metadata = extract_features(
            policy, image, instruction, np.asarray(obs["agent"]["eef_pos"], dtype=np.float32)
        )
        runtime = time.monotonic() - started
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output_path)
        metadata.update({
            "task": args.task,
            "seed": seed,
            "environment_id": environment_id,
            "worker_id": args.worker_id,
            "physical_gpu": args.gpu,
            "runtime_seconds": runtime,
            "canonical_snapshot_sha256": reference["canonical_snapshot_sha256"],
            "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha,
        })
        atomic_json(meta_path, metadata)
        print(json.dumps({
            "task": args.task,
            "seed": seed,
            "runtime_seconds": round(runtime, 3),
            "output": str(output_path),
        }), flush=True)


if __name__ == "__main__":
    main()
