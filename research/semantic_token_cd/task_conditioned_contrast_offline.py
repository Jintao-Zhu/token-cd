#!/usr/bin/env python3
"""Offline 300-state mechanism analysis for task-conditioned contrast + APC."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.task_conditioned_contrast_protocol import (
    APC_BETA,
    ARTIFACT,
    ATTENTION_LAYERS,
    CONTROL_INSTRUCTIONS,
    CONTROL_INSTRUCTIONS_ALT,
    KMEANS_K,
    KMEANS_SEED,
    LAMBDA0,
    OFFLINE_SEEDS_PER_TASK,
    PCD_SOURCE,
    PROTOCOL,
    TASKS,
)


STATE_ROOT = (
    Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts")
    / "prompt_attn_instr_swap_v1/runs/emitted_states"
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def state_pairs(task: str, limit: int = OFFLINE_SEEDS_PER_TASK) -> list[tuple[Path, Path]]:
    pairs = []
    for json_path in sorted((STATE_ROOT / task).glob("seed_*/step_*.json")):
        npz_path = json_path.with_suffix(".npz")
        if npz_path.exists():
            pairs.append((json_path, npz_path))
    if len(pairs) < limit:
        raise FileNotFoundError(f"{task}: only {len(pairs)} RGB states, need {limit}")
    # Deterministic, pre-registered: evenly spaced over sorted trajectory states.
    indices = np.linspace(0, len(pairs) - 1, num=limit, dtype=int).tolist()
    if len(set(indices)) != limit:
        raise RuntimeError("offline sampling produced duplicate indices")
    return [pairs[index] for index in indices]


def build_policy(base, task: str, arm: str, control_instruction: str):
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
    from research.semantic_token_cd.task_conditioned_contrast_policy import TaskConditionedContrastInference

    policy = copy.copy(base)
    policy.__class__ = TaskConditionedContrastInference
    _init_common(policy, LAMBDA0)
    policy.selector_mode = "prompt_attention"
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = ATTENTION_LAYERS
    policy.selection_count = None
    policy.selection_top_p = None
    policy.selection_budget_schedule = None
    policy.selection_budget_scale = 1.0
    policy.selection_budget_label = None
    policy.selection_budget_source = "matched"
    policy.selection_budget_entities = None
    policy.save_prompt_attention = False
    policy.arm = arm
    policy.apc_beta = APC_BETA
    policy.control_instruction = control_instruction
    policy.task_key = task
    policy.save_condition_logits = True
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--limit", type=int, default=OFFLINE_SEEDS_PER_TASK)
    parser.add_argument("--alt-control", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    repo = Path("/home/leju-suzhou/zjt_ws/token-cd")
    for path in (str(repo / "task1/shim_site"), str(repo), str(PCD_SOURCE)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    pairs = state_pairs(args.task, args.limit)
    control = (CONTROL_INSTRUCTIONS_ALT if args.alt_control else CONTROL_INSTRUCTIONS)[args.task]
    output_dir = artifact / ("offline_alt_control" if args.alt_control else "offline") / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**config)

    # T computes all four conditions; C_APC/T_APC metrics are derived from the
    # same frozen logits without another forward pass.
    policy = build_policy(base, args.task, "T", control)
    for index, (json_path, npz_path) in enumerate(pairs):
        metadata = json.loads(json_path.read_text())
        image = np.asarray(np.load(npz_path)["image"])
        instruction = metadata["instruction"]
        seed = int(metadata["seed"])
        step = int(metadata["control_step"])
        record_path = output_dir / f"seed_{seed:03d}_step_{step:04d}.json"
        logits_path = output_dir / f"seed_{seed:03d}_step_{step:04d}.npz"
        if record_path.exists() and logits_path.exists():
            continue

        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        _raw, _actions, meta = policy.step(image, None, instruction, proprio=None)
        condition = policy._episode_logits[-1]
        from research.semantic_token_cd.task_conditioned_contrast_policy import (
            _action_start,
            _apc_apply,
        )
        action_start, _action_bins = _action_start(policy)
        LA = torch.from_numpy(condition["LA"]).float()
        LB = torch.from_numpy(condition["LB"]).float()
        LC = torch.from_numpy(condition["LC"]).float()
        LD = torch.from_numpy(condition["LD"]).float()
        S_visual = LA - LB
        S_control = LC - LD
        S_task = S_visual - S_control
        clean_global = torch.tensor(meta["clean_action_tokens"], dtype=torch.long)
        clean_local = clean_global - action_start

        def decode_arm(scores: torch.Tensor, apc: bool) -> dict:
            apc_meta = None
            final_scores = scores
            if apc:
                final_scores, apc_meta = _apc_apply(scores, LA, APC_BETA)
            local = final_scores.argmax(dim=-1)
            local[6] = clean_local[6]
            glob = local + action_start
            glob[6] = clean_global[6]
            decoded = np.asarray(policy._decode_actions(glob, policy.unnorm_key), dtype=float).reshape(-1)
            return {
                "executed_action": decoded.tolist(),
                "final_action_tokens": [int(value) for value in glob.tolist()],
                "changed_dims_vs_clean": int((local[:6] != clean_local[:6]).sum().item()),
                "APC_keep_count_per_dim": None if apc_meta is None else [int(v) for v in apc_meta["keep"].sum(dim=-1).tolist()],
                "APC_blocked": None if apc_meta is None else [bool(v) for v in apc_meta["blocked"].tolist()],
                "blocked_winner_clean_rank": None if apc_meta is None else [int(v) for v in apc_meta["blocked_clean_rank"].tolist()],
                "blocked_winner_clean_relative_prob": None if apc_meta is None else [float(v) for v in apc_meta["blocked_clean_relative_prob"].tolist()],
            }

        offline_arms = {
            "C": decode_arm(LA + LAMBDA0 * S_visual, False),
            "C_APC": decode_arm(LA + LAMBDA0 * S_visual, True),
            "T": decode_arm(LA + LAMBDA0 * S_task, False),
            "T_APC": decode_arm(LA + LAMBDA0 * S_task, True),
        }
        record = {
            "protocol_id": PROTOCOL,
            "task": args.task,
            "seed": seed,
            "episode_step": step,
            "state_json": str(json_path),
            "state_npz": str(npz_path),
            "rgb_sha256": metadata["rgb_sha256"],
            "original_instruction": instruction,
            "control_instruction": control,
            "m_t": meta["m_t"],
            "selected_token_ids": meta["selected_token_ids"],
            "matched_clusters": meta["matched_cluster_ids"],
            "S_visual_rms": meta["S_visual_rms"],
            "S_control_rms": meta["S_control_rms"],
            "S_task_rms": meta["S_task_rms"],
            "cos_visual_task": meta["cos_visual_task"],
            "clean_action_tokens": meta["clean_action_tokens"],
            "final_action_tokens": meta["final_action_tokens"],
            "changed_dims_vs_clean": meta["changed_dims_vs_clean"],
            "APC_keep_count_per_dim": meta["APC_keep_count_per_dim"],
            "APC_blocked": meta["APC_blocked"],
            "blocked_winner_clean_rank": meta["blocked_winner_clean_rank"],
            "blocked_winner_clean_relative_prob": meta["blocked_winner_clean_relative_prob"],
            "arms": offline_arms,
            "C_to_T_action_change": int(
                any(
                    a != b
                    for a, b in zip(
                        offline_arms["C"]["final_action_tokens"],
                        offline_arms["T"]["final_action_tokens"],
                    )
                )
            ),
            "C_APC_to_T_APC_action_change": int(
                any(
                    a != b
                    for a, b in zip(
                        offline_arms["C_APC"]["final_action_tokens"],
                        offline_arms["T_APC"]["final_action_tokens"],
                    )
                )
            ),
        }
        atomic_json(record_path, record)
        np.savez_compressed(
            logits_path,
            LA=condition["LA"],
            LB=condition["LB"],
            LC=condition["LC"],
            LD=condition["LD"],
            S_visual=condition["S_visual"],
            S_control=condition["S_control"],
            S_task=condition["S_task"],
        )
        print(json.dumps({
            "task": args.task,
            "index": index,
            "seed": seed,
            "step": step,
            "m_t": meta["m_t"],
            "S_visual_rms": meta["S_visual_rms"],
            "S_control_rms": meta["S_control_rms"],
            "S_task_rms": meta["S_task_rms"],
        }), flush=True)


if __name__ == "__main__":
    main()
