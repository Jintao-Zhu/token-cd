"""Recover five close-drawer trajectories and extract 15 offline L11 states.

The environment is stepped only with actions saved by the completed canonical
SHR rollout.  The freshly evaluated policy never controls the environment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, get_image_from_maniskill2_obs_dict, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy import (
    PromptAttentionSHRInference, extract_prompt_attention_per_layer,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import load_reference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common


TASK = "google_robot_close_drawer"
SEEDS = (0, 1, 9, 21, 49)
PHASES = ("early", "middle", "late")


def step_indices(length: int) -> list[int]:
    return [int(round((length - 1) * fraction)) for fraction in (.1, .5, .9)]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    canonical = args.canonical.resolve()
    env, _ = make_environment(TASK, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, TASK, {}, False)
    policy = copy.copy(OpenVLAInference(**config))
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, .5)
    policy.beta = 0.0
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[TASK]
    policy.attention_layers = (11,)

    for seed in SEEDS:
        source_root = canonical / "episodes" / TASK / "shr_harmonic"
        summary = json.loads((source_root / f"episode_{seed:03d}_summary.json").read_text())
        rollout = np.load(source_root / f"episode_{seed:03d}_arrays.npz")
        actions = rollout["executed_actions"]
        targets = step_indices(len(actions))
        target_map = dict(zip(targets, PHASES))
        with (canonical / "snapshots" / TASK / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        reference = load_reference(canonical, TASK, seed)
        if snapshot_sha(snapshot) != reference["canonical_snapshot_sha256"]:
            raise RuntimeError("snapshot hash mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if (state_sha, rgb_sha) != (reference["initial_state_sha256"], reference["initial_rgb_sha256"]):
            raise RuntimeError("initial state/RGB mismatch")
        instruction = env.unwrapped.get_language_instruction()
        if instruction != reference["instruction"]:
            raise RuntimeError("instruction mismatch")

        for step, action in enumerate(actions):
            if step in target_map:
                phase = target_map[step]
                out = artifact / "states" / TASK / f"seed_{seed:03d}"
                npz_path = out / f"step_{step:03d}.npz"
                json_path = out / f"step_{step:03d}.json"
                if not npz_path.exists() or not json_path.exists():
                    raw_image = get_image_from_maniskill2_obs_dict(env, obs)
                    image = np.asarray(
                        raw_image.convert("RGB") if hasattr(raw_image, "convert") else raw_image,
                        dtype=np.uint8,
                    )
                    policy.reset(instruction, seed=seed)
                    inputs = policy.process_inputs(image, task_description=instruction)
                    with projector_intervention(policy.vla) as trace:
                        clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
                    if trace.before is None or clean_scores.shape[0] != 7:
                        raise RuntimeError("offline clean forward failed")
                    attention, meta = extract_prompt_attention_per_layer(policy, inputs, instruction, trace.before)
                    standard_ids = summary["selector_trace"][step]["selected_token_ids"]
                    standard_mask = np.isin(np.arange(256), standard_ids).astype(np.uint8)
                    out.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(npz_path, original=attention, image=image, standard_mask=standard_mask)
                    atomic_json(json_path, {
                        "task": TASK, "seed": seed, "step": step, "phase": phase,
                        "instruction": instruction, "arrays_file": npz_path.name,
                        "trajectory_source": "canonical SHR saved executed_actions",
                        "policy_did_not_control_environment": True,
                        "attention_layers_available": list(range(32)),
                        "attention_meta": meta,
                        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "matched_count": int(standard_mask.sum()),
                    })
                    Image.fromarray(image).save(out / f"step_{step:03d}.png")
                    print(json.dumps({"seed": seed, "step": step, "phase": phase}), flush=True)
            obs, _reward, _success, _truncated, _info = env.step(np.asarray(action))


if __name__ == "__main__":
    main()
