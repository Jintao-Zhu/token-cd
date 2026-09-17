"""Offline four-branch decode on emitted vanilla states.

For every emitted frame the same four decodes are run:
  region = top tokens or middle tokens (identical across instructions)
  instruction = "open top drawer" or "open middle drawer"

Each branch records the clean greedy action, the SHR-guided action
(guided = clean prefix; final = clean + 0.5*(clean-negative) on dims 0..5),
the guided/clean action L2, changed action dims and reconstruction strength.
The stored masks come from the same frame's simulator geometry + segmentation,
so the *same* region reconstruction is applied under both instructions.

This is a mechanism probe only (not an SHR performance evaluation).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.xdrawer_protocol import (
    DRAWERS, INSTRUCTION_BY_DRAWER, PROTOCOL, TASK_BY_DRAWER, atomic_json,
)
from research.semantic_token_cd.semantic_recon_rollout import LAMBDA, _init_common
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.distractor_rollout import jsonable

ARMS_BRANCH = ("target_region", "other_region")


def build_policy(base, instruction: str):
    p = copy.copy(base)
    p.__class__ = PromptAttentionSHRInference
    _init_common(p, LAMBDA)
    p.beta = 0.0
    p.selector_mode = "sim_region"
    p.attention_layers = (11,)
    p.save_prompt_attention = False
    p.region_token_ids = None
    p.region_label = None
    p.region_aux = None
    p.task_index = 0
    p.reset(instruction, seed=0)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emitted", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drawer", choices=DRAWERS, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    emitted = args.emitted.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_root = emitted / args.drawer
    if not state_root.exists():
        raise FileNotFoundError(state_root)
    frames = sorted(state_root.glob("seed_*/step_*.json"))

    checkpoint = "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source/pretrained/openvla-7b"
    config = get_policy_config("openvla", checkpoint, TASK_BY_DRAWER[args.drawer], {}, False)
    base = OpenVLAInference(**config)
    policies = {d: build_policy(base, INSTRUCTION_BY_DRAWER[d]) for d in DRAWERS}

    records = []
    for meta_path in frames:
        meta = json.loads(meta_path.read_text())
        stem = meta_path.stem
        np_path = meta_path.with_suffix(".npz")
        image = np.asarray(np.load(np_path)["image"], dtype=np.uint8)
        branch = {}
        for instr_d in DRAWERS:
            mask_labels = {
                "target_region": instr_d,
                "other_region": "middle" if instr_d == "top" else "top",
            }
            for branch_name, mask_d in mask_labels.items():
                pol = policies[instr_d]
                pol._episode_trace = []
                pol._episode_logits = []
                mask_tokens = meta["regions"][mask_d]["token_ids"]
                pol.region_token_ids = list(mask_tokens)
                pol.region_label = mask_d
                regs_sizes = meta.get("sizes", {})
                pol.region_aux = {
                    "region_size_top": regs_sizes.get("top", 0),
                    "region_size_middle": regs_sizes.get("middle", 0),
                    "region_overlap_top_middle": meta.get("overlap", 0),
                    "region_label": mask_d,
                    "row_span": meta["regions"][mask_d]["row_span"],
                    "col_span": meta["regions"][mask_d]["col_span"],
                }
                _raw, _actions, step_meta = pol.step(
                    image, None, INSTRUCTION_BY_DRAWER[instr_d],
                    proprio=np.zeros(8, dtype=np.float64),
                )
                rec = {k: step_meta[k] for k in (
                    "clean_action", "guided_action", "guided_clean_action_l2",
                    "guided_changed_dims", "guided_change_ratio",
                    "feature_perturbation_norm", "feature_perturbation_relative",
                    "lambda", "beta", "coverage_mode",
                    "centered_logit_residual_norm_per_dim",
                )}
                rec["mask_drawer"] = mask_d
                rec["instruction_drawer"] = instr_d
                rec["region_size"] = int(step_meta["num_tokens"])
                rec["region_token_ids"] = step_meta["selected_token_ids"]
                branch[f"{instr_d}/{branch_name}"] = rec
        row = {
            "drawer_origin": args.drawer,
            "seed": meta["seed"],
            "control_step": meta["control_step"],
            "rgb_sha256": meta["rgb_sha"],
            "regions": {d: meta["regions"][d]["token_ids"] for d in DRAWERS},
            "sizes": meta["sizes"],
            "overlap": meta["overlap"],
            "branches": branch,
        }
        records.append(row)
        print(json.dumps({"frame": f"{args.drawer}/{meta['seed']}/{stem}",
                          "instr": INSTRUCTION_BY_DRAWER[args.drawer]}), flush=True)

    summary = {
        "protocol_id": PROTOCOL,
        "purpose": "offline four-branch: same frame x top/middle instruction x target/other region",
        "drawer_origin": args.drawer,
        "n_frames": len(records),
        "config": {"lambda": LAMBDA, "guided_dims": [0, 1, 2, 3, 4, 5],
                   "gripper": "clean dim 6", "prefix": "clean greedy",
                   "sampling": False, "harmonic": "16x16 4-neighbor Dirichlet beta=0"},
    }
    atomic_json(out / f"offline_branches_{args.drawer}.json", jsonable({"summary": summary, "records": records}))
    print(json.dumps({"drawer": args.drawer, "frames": len(records), "out": str(out / f"offline_branches_{args.drawer}.json")}))


if __name__ == "__main__":
    main()
