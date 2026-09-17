"""XSWAP-V1 offline same-state decode worker.

Reads emitted vanilla state samples (image npz + meta json) and computes four
same-state branches on the identical image and shared clean greedy prefix:
  correct / paraphrase / swapped selector attention + matched-coverage random.
The actual task instruction never changes.  Emits per-state JSON scalars plus an
npz holding each branch's 256-d attention vector and 16x16 mask for analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
from research.semantic_token_cd.xswap_protocol import (
    ARTIFACT, N_VISUAL, atomic_json, instruction_set, resolve_present_phrases,
    swap_for_scene,
)
from research.semantic_token_cd.xswap_rollout import build_policy, make_environment

BRANCHES = ("correct", "paraphrase", "swapped", "random")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(__import__(
        "research.semantic_token_cd.xswap_protocol", fromlist=["TASKS"]).TASKS))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", default="manual")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from research.semantic_token_cd.xswap_protocol import TASKS
    task = args.task
    emitted_root = ARTIFACT / "runs" / "emitted_states" / task
    if not emitted_root.exists():
        raise FileNotFoundError(emitted_root)
    out_root = ARTIFACT / "runs" / "offline" / task
    out_root.mkdir(parents=True, exist_ok=True)

    env, _environment_id = make_environment(task, args.gpu)
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, task, {}, False))
    policies = {b: build_policy(base, task, b) for b in BRANCHES}

    records = []
    for npz_path in sorted(emitted_root.glob("seed_*/step_*.npz")):
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        meta = json.loads(json_path.read_text())
        seed = meta["seed"]
        image = np.asarray(np.load(npz_path)["image"], dtype=np.uint8)
        instruction = meta["instruction"]
        # present phrases are re-derived from a fresh restore so the swapped
        # mapping equals the closed-loop one for this scene.
        present = []
        try:
            env.unwrapped.reset(seed=seed)
            present = resolve_present_phrases(env)
        except Exception:
            present = []
        plan = instruction_set(instruction)
        if plan["kind"] in ("pick_object", "move_near"):
            plan = swap_for_scene(instruction, present)

        row = {"task": task, "seed": seed, "control_step": meta["control_step"],
               "rgb_sha256": meta["rgb_sha256"], "instruction": instruction,
               "present": present, "plan_kind": plan.get("kind"),
               "swapped_kind": plan.get("swapped_kind"),
               "branches": {}, "clean_equal_across_branches": True,
               "m_equal_across_branches": True}
        out_stem = out_root / f"seed_{seed:03d}" / f"step_{meta['control_step']:04d}"
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        attn_arrays = {}
        masks_arrays = {}
        clean_ref = None
        m_ref = None
        for branch in BRANCHES:
            pol = policies[branch]
            if branch == "correct":
                pol.selector_instruction = None
            elif branch == "paraphrase":
                pol.selector_instruction = plan["paraphrase"]
            elif branch == "swapped":
                pol.selector_instruction = plan["swapped"]
            pol._episode_trace = []
            pol._episode_logits = []
            pol._episode_seed = seed
            pol._selector_step = 0
            pol.reset(instruction, seed=seed)
            _raw, _actions, _aux = pol.step(
                image, None, instruction, proprio=np.zeros(8, dtype=np.float64)
            )
            step = pol._episode_trace[-1]
            record = pol._episode_logits[-1]
            clean = np.asarray(step["clean_action"], dtype=np.float64)
            guided = np.asarray(step["guided_action"], dtype=np.float64)
            if clean_ref is None:
                clean_ref = clean
            elif not np.allclose(clean, clean_ref, atol=1e-6):
                row["clean_equal_across_branches"] = False
            m = int(step["num_tokens"])
            if m_ref is None:
                m_ref = m
            elif m != m_ref:
                row["m_equal_across_branches"] = False
            attn = np.asarray(record.get("prompt_attention"), dtype=np.float32) if branch != "random" else None
            mask = np.zeros(N_VISUAL, dtype=np.uint8)
            mask[np.asarray(step["selected_token_ids"], dtype=np.int64)] = 1
            attn_arrays[branch] = attn if attn is not None else np.zeros(N_VISUAL, dtype=np.float32)
            masks_arrays[branch] = mask
            delta = guided[:6] - clean[:6]
            row["branches"][branch] = {
                "m": m,
                "selected_token_ids": [int(i) for i in np.flatnonzero(mask)],
                "selector_instruction": step.get("selector_instruction"),
                "selector_matches_task": step.get("selector_matches_task"),
                "clean_action": clean.tolist(),
                "guided_action": guided.tolist(),
                "effect_l2": float(np.linalg.norm(delta)),
                "effect_translation_norm": float(np.linalg.norm(delta[:3])),
                "feature_perturbation_norm": float(step["feature_perturbation_norm"]),
                "feature_perturbation_relative": float(step["feature_perturbation_relative"]),
                "centered_logit_residual_norm_per_dim": step["centered_logit_residual_norm_per_dim"],
                "guided_changed_dims": int(step["guided_changed_dims"]),
                "guided_clean_action_l2": float(step["guided_clean_action_l2"]),
                "attention_sha256": step.get("attention_sha256"),
                "coverage_mode": step.get("coverage_mode"),
                "lambda": float(step["lambda"]), "beta": float(step["beta"]),
            }
        sel = row["branches"]
        for b in BRANCHES:
            sa = set(sel["correct"]["selected_token_ids"])
            sb = set(sel[b]["selected_token_ids"])
            row["branches"][b]["jaccard_vs_correct"] = round(len(sa & sb) / max(1, len(sa | sb)), 4)
        records.append(row)
        np.savez_compressed(out_stem.with_suffix(".npz"), **attn_arrays, **{
            f"mask_{b}": masks_arrays[b] for b in BRANCHES
        })
        atomic_json(out_stem.with_suffix(".json"), row)
        print(json.dumps({"task": task, "seed": seed, "step": meta["control_step"],
                          "m": m_ref}), flush=True)
    summary = {
        "task": task,
        "n_states": len(records),
        "clean_equal_all_states": all(r["clean_equal_across_branches"] for r in records),
        "m_equal_all_states": all(r["m_equal_across_branches"] for r in records),
        "mean_m": float(np.mean([list(r["branches"].values())[0]["m"] for r in records])) if records else None,
    }
    atomic_json(out_root / "offline_summary.json", summary)
    env.close()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
