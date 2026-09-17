"""Stage A worker: extract and audit grouping on the existing 918 states."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.semantic_token_cd.prompt_action_complement_protocol import (
    ARMS, ARTIFACT, PCD_SOURCE, STATE_SOURCE, TASKS, atomic_json,
)
from research.semantic_token_cd.prompt_action_complement_state import build_policy, extract_state


def jaccard(a, b):
    a, b = set(a), set(b); return len(a & b) / max(1, len(a | b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--gpu", required=True, type=int, choices=(2, 3))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    policy = build_policy(base, args.task)
    items = sorted((STATE_SOURCE / "runs/emitted_states" / args.task).glob("seed_*/*.npz"))
    items = [item for index, item in enumerate(items) if index % args.num_shards == args.shard_index]
    for image_path in items:
        rel = image_path.relative_to(STATE_SOURCE / "runs/emitted_states" / args.task)
        out = ARTIFACT / "stage_a/states" / args.task / rel
        if out.with_suffix(".json").exists() and out.exists(): continue
        old_json = STATE_SOURCE / "runs/offline" / args.task / rel.with_suffix(".json")
        old = json.loads(old_json.read_text())
        image = np.asarray(np.load(image_path)["image"], dtype=np.uint8)
        seed = int(old["seed"]); step = int(old["control_step"]); instruction = old["instruction"]
        state = extract_state(policy, image, instruction, seed, step)
        original = state["masks"]["original"]["selected"]
        old_original = sorted(int(x) for x in old["branches"]["correct"]["selected_token_ids"])
        if original != old_original: raise RuntimeError("Original mask does not reproduce saved L11 mainline")
        arrays = {
            "image": image, "prompt_attention": state["prompt"],
            **state["action_arrays"],
        }
        for arm in ARMS:
            mask = np.zeros(256, dtype=np.uint8); mask[state["masks"][arm]["selected"]] = 1
            arrays[f"{arm}__mask"] = mask
        out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, **arrays)
        pairwise = {f"{a}__{b}": jaccard(state["masks"][a]["selected"], state["masks"][b]["selected"])
                    for i, a in enumerate(ARMS) for b in ARMS[i + 1:]}
        atomic_json(out.with_suffix(".json"), {
            "task": args.task, "seed": seed, "control_step": step, "instruction": instruction,
            "m": state["m"], "r": state["masks"]["original"]["supplement_count_r"],
            "original_mask_exact": True, "prompt_meta": state["prompt_meta"],
            "action_meta": state["action_meta"], "pairwise_jaccard": pairwise,
            "arms": state["masks"], "arrays_file": str(out.relative_to(ARTIFACT)),
        })
        print(json.dumps({"task": args.task, "seed": seed, "step": step}), flush=True)


if __name__ == "__main__": main()

