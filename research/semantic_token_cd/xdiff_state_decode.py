"""Stage 2 identical-state action propagation for four pre-specified selectors."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.xdiff_protocol import (
    ARTIFACT, ATTENTION_LAYERS, EPSILON, LAMBDA, SOURCE_ARTIFACT, TASKS, atomic_json,
)

CONFIGS = ("correct", "semantic_0p5", "semantic_1p0", "paraphrase_0p5")


def build(base, task, name, plan):
    p = copy.copy(base); p.__class__ = PromptAttentionSHRInference
    _init_common(p, LAMBDA)
    p.beta = 0.0; p.selector_mode = "prompt_attention"; p.task_index = TASK_INDEX[task]
    p.attention_layers = tuple(ATTENTION_LAYERS); p.selection_count = None; p.selection_top_p = None
    p.save_prompt_attention = True; p.selector_instruction = None
    p.selector_contrast_instruction = None; p.selector_difference_eta = None
    if name != "correct":
        p.selector_difference_eta = 1.0 if name == "semantic_1p0" else 0.5
        p.selector_difference_kind = "paraphrase" if name == "paraphrase_0p5" else "semantic"
        p.selector_contrast_instruction = (plan["paraphrase"] if name == "paraphrase_0p5"
                                           else plan["semantic"])
        p.selector_difference_epsilon = EPSILON
    return p


def centered(record):
    r = np.asarray(record["positive"][:6], dtype=np.float64) - np.asarray(record["negative"][:6], dtype=np.float64)
    return r - r.mean(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS); ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard-index", type=int, required=True); ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    manifest = json.loads((ARTIFACT / "STAGE2_STATE_MANIFEST.json").read_text())["states"][args.task]
    items = [x for i, x in enumerate(manifest) if i % args.num_shards == args.shard_index]
    for rel in items:
        image_path = SOURCE_ARTIFACT / rel
        old_json = (SOURCE_ARTIFACT / "runs/offline" / args.task /
                    image_path.parent.name / image_path.name).with_suffix(".json")
        old = json.loads(old_json.read_text())
        plan = {"semantic": old["branches"]["swapped"]["selector_instruction"],
                "paraphrase": old["branches"]["paraphrase"]["selector_instruction"]}
        image = np.asarray(np.load(image_path)["image"], dtype=np.uint8)
        instruction = old["instruction"]; seed = int(old["seed"]); step_no = int(old["control_step"])
        out = ARTIFACT / "stage2_action" / "states" / args.task / f"seed_{seed:03d}" / f"step_{step_no:04d}"
        if out.with_suffix(".json").exists() and out.with_suffix(".npz").exists():
            continue
        metrics = {}; arrays = {}; clean_ref = None; m_ref = None; residual_ref = None
        for name in CONFIGS:
            policy = build(base, args.task, name, plan)
            policy._episode_trace = []; policy._episode_logits = []; policy._episode_seed = seed; policy._selector_step = step_no
            policy.reset(instruction, seed=seed)
            policy.step(image, None, instruction, proprio=np.zeros(8, dtype=np.float64))
            trace = policy._episode_trace[-1]; record = policy._episode_logits[-1]
            clean = np.asarray(trace["clean_action"], dtype=np.float64)
            guided = np.asarray(trace["guided_action"], dtype=np.float64)
            if clean_ref is None: clean_ref = clean
            elif not np.array_equal(clean, clean_ref): raise RuntimeError("clean action differs across masks")
            m = int(trace["num_tokens"])
            if m_ref is None: m_ref = m
            elif m != m_ref: raise RuntimeError("matched count differs on identical state")
            res = centered(record)
            if residual_ref is None: residual_ref = res
            denom = np.linalg.norm(residual_ref) * np.linalg.norm(res)
            metrics[name] = {
                "m": m, "selected_token_ids": trace["selected_token_ids"],
                "guided_action": guided.tolist(), "guided_changed_dims": trace["guided_changed_dims"],
                "guided_clean_action_l2": trace["guided_clean_action_l2"],
                "feature_perturbation_norm": trace["feature_perturbation_norm"],
                "centered_residual_norm": float(np.linalg.norm(res)),
                "residual_cosine_vs_correct": float(np.sum(residual_ref * res) / denom) if denom else 1.0,
                "final_token_ids": trace["final_token_ids"],
                "prompt_query_token_ids": trace.get("prompt_query_token_ids"),
                "contrast_prompt_query_token_ids": trace.get("contrast_prompt_query_token_ids"),
            }
            for key, value in record.items(): arrays[f"{name}__{key}"] = value
        metrics["audit"] = {"clean_exact": True, "m_exact": True,
                            "all_masks_exact_m": all(len(set(metrics[n]["selected_token_ids"])) == m_ref for n in CONFIGS)}
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.with_suffix(".npz"), **arrays)
        atomic_json(out.with_suffix(".json"), {
            "task": args.task, "seed": seed, "control_step": step_no, "instruction": instruction,
            "semantic_contrast": plan["semantic"], "paraphrase": plan["paraphrase"],
            "metrics": metrics, "source_rgb_sha256": old["rgb_sha256"],
        })
        print(json.dumps({"task": args.task, "seed": seed, "step": step_no}), flush=True)


if __name__ == "__main__":
    main()
