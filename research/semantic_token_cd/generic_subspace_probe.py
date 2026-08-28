"""K-dimensional Generic-Subspace probe for Orthogonal Dual-CD.

Mechanism diagnostic (NOT a replacement for the closed-loop method). For a set
of initial states it records, per state, the clean logits, the semantic-mask
negative logits, and M different generic-mask negative logits. The residual
vectors r_i = pos - masked_i then span a candidate "generic correction
subspace"; the analysis script decides whether semantic residuals live outside
it.

This script only RUNS forward passes and saves raw action-vocabulary logits.
It never computes success, never touches the closed-loop rollout, and reuses the
audited masking machinery (block_action_to_visual_attention) plus the existing
OrthogonalDualCDInference._negative_scores path.

Generic masks (all size-matched to 64 tokens, so only the spatial ARRANGEMENT
varies, not the amount of information removed):
  1. lattice phase (0,0)  == strided_2x2          (the current uniform basis)
  2. lattice phase (1,1)  == shifted_strided_2x2
  3. lattice phase (0,1)  (even rows, odd columns)
  4. lattice phase (1,0)  (odd rows, even columns)
  5..R+4. size-matched random masks (deterministic seeds)

The four 2x2 phase lattices are mutually disjoint and tile the 16x16 grid.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    get_image_from_maniskill2_obs_dict,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.distractor_policy import _action_logits
from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.orthogonal_dual_rollout import build_policy
from research.semantic_token_cd.spatial_grid_policy import GRID_SIZE, get_grid_mask_indices
from research.semantic_token_cd.spatial_grid_rollout import make_environment


PROTOCOL = "GENERIC_SUBSPACE_PROBE_V1"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
)
GRID = GRID_SIZE  # 16


def lattice_phase(row_parity: int, col_parity: int) -> list[int]:
    """64-token strided 2x2 lattice at the given (row_parity, col_parity) phase."""
    indices = [
        r * GRID + c
        for r in range(row_parity, GRID, 2)
        for c in range(col_parity, GRID, 2)
    ]
    assert len(indices) == 64 and len(set(indices)) == 64
    return indices


def random_mask_64(seed: int) -> list[int]:
    """Deterministic size-64 random mask (fixed seed per mask id)."""
    rng = np.random.default_rng(seed)
    indices = sorted(int(i) for i in rng.choice(256, size=64, replace=False))
    assert len(indices) == 64 and len(set(indices)) == 64
    return indices


def build_mask_set(n_random: int) -> tuple[list[list[int]], list[str]]:
    """Return the generic mask list and matching human-readable descriptors.

    Order is fixed and documented so the analysis script can reproduce the
    train/held-out split deterministically.
    """
    masks: list[list[int]] = []
    labels: list[str] = []
    masks.append(get_grid_mask_indices("strided_2x2"))
    labels.append("lattice_phase_00")
    masks.append(get_grid_mask_indices("shifted_strided_2x2"))
    labels.append("lattice_phase_11")
    masks.append(lattice_phase(0, 1))
    labels.append("lattice_phase_01")
    masks.append(lattice_phase(1, 0))
    labels.append("lattice_phase_10")
    for i in range(n_random):
        masks.append(random_mask_64(0x6000_0000 + i))
        labels.append(f"random_64_{i:02d}")
    return masks, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,5,6,7,8,9")
    parser.add_argument("--n-random", type=int, default=12, help="random generic masks")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    masks, mask_labels = build_mask_set(args.n_random)
    n_masks = len(masks)

    artifact.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": PROTOCOL,
        "task": args.task,
        "seeds": seeds,
        "n_masks": n_masks,
        "mask_labels": mask_labels,
        "mask_token_counts": [len(m) for m in masks],
        "n_random": args.n_random,
        "note": "residuals are raw logit differences pos - masked_i, action vocab [7,256]",
    }
    (artifact / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    env, _ = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config), 0.5, 0.35)

    all_positive: list[np.ndarray] = []
    all_semantic: list[np.ndarray] = []
    per_mask: list[list[np.ndarray]] = [[] for _ in range(n_masks)]

    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        obs, _, _ = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        policy.reset(instruction, seed=seed)

        # One step: positive + M generic masks + semantic mask, all on the same image.
        image = get_image_from_maniskill2_obs_dict(env, obs)
        inputs = policy.process_inputs(image, task_description=instruction)

        with projector_intervention(policy.vla) as positive_trace:
            positive_scores = policy._forward_scores(
                inputs, policy.unnorm_key, do_sample=False
            )
        if positive_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError(f"seed {seed}: positive branch invalid")
        positive_features = positive_trace.before
        positive = _action_logits(policy, positive_scores)
        all_positive.append(positive)

        input_ids = inputs["input_ids"]
        appended_empty_token = int(not bool(torch.all(input_ids[:, -1] == 29871).item()))
        action_query_start = (
            positive_features.shape[1] + input_ids.shape[1] + appended_empty_token - 1
        )

        semantic_indices, _ = policy._select(positive_features[0].numpy())
        semantic_scores, _, _, _ = policy._negative_scores(
            inputs, semantic_indices, action_query_start, positive_features
        )
        semantic_scores = policy._align_scores(semantic_scores, positive_scores)
        all_semantic.append(_action_logits(policy, semantic_scores))

        for mi, mask in enumerate(masks):
            scores, _, _, _ = policy._negative_scores(
                inputs, mask, action_query_start, positive_features
            )
            scores = policy._align_scores(scores, positive_scores)
            per_mask[mi].append(_action_logits(policy, scores))

        print(json.dumps({"task": args.task, "seed": seed, "done": True}), flush=True)

    generic = np.stack([np.stack(per_mask[mi]) for mi in range(n_masks)])  # [M, N, 7, 256]
    np.savez_compressed(
        artifact / "residuals.npz",
        positive=np.stack(all_positive),       # [N, 7, 256]
        generic=generic,                        # [M, N, 7, 256]
        semantic=np.stack(all_semantic),        # [N, 7, 256]
        mask_labels=np.asarray(mask_labels),
    )
    print(json.dumps({"task": args.task, "DONE": True, "n_states": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
