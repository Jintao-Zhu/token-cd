"""Stage B worker: four-mask action response on the locked 240 states."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.semantic_token_cd.prompt_action_complement_protocol import (
    ARMS, ARTIFACT, PCD_SOURCE, STAGE2_SOURCE_MANIFEST, STATE_SOURCE, TASKS, atomic_json,
)
from research.semantic_token_cd.prompt_action_complement_state import (
    build_policy, evaluate_mask, extract_state,
)


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
    manifest = json.loads(STAGE2_SOURCE_MANIFEST.read_text())["states"][args.task]
    items = [relative for index, relative in enumerate(manifest)
             if index % args.num_shards == args.shard_index]
    for relative in items:
        image_path = STATE_SOURCE / relative
        rel = image_path.relative_to(STATE_SOURCE / "runs/emitted_states" / args.task)
        out = ARTIFACT / "stage_b/states" / args.task / rel
        if out.with_suffix(".json").exists() and out.exists(): continue
        old = json.loads((STATE_SOURCE / "runs/offline" / args.task / rel.with_suffix(".json")).read_text())
        image = np.asarray(np.load(image_path)["image"], dtype=np.uint8)
        seed = int(old["seed"]); step = int(old["control_step"]); instruction = old["instruction"]
        state = extract_state(policy, image, instruction, seed, step)
        metrics, arrays, residuals = {}, {}, {}
        for arm in ARMS:
            evaluation = evaluate_mask(policy, state, state["masks"][arm]["selected"])
            metrics[arm] = {**state["masks"][arm], **evaluation["metrics"]}
            residuals[arm] = evaluation["arrays"]["centered_residual"].astype(np.float64)
            for key, value in evaluation["arrays"].items(): arrays[f"{arm}__{key}"] = value
            mask = np.zeros(256, dtype=np.uint8); mask[state["masks"][arm]["selected"]] = 1
            arrays[f"{arm}__mask"] = mask
        reference = residuals["original"]
        for arm in ARMS:
            denominator = np.linalg.norm(reference) * np.linalg.norm(residuals[arm])
            metrics[arm]["residual_cosine_vs_original"] = float(
                np.sum(reference * residuals[arm]) / denominator if denominator else 1.0
            )
        arrays.update({"prompt_attention": state["prompt"], **state["action_arrays"]})
        out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, **arrays)
        atomic_json(out.with_suffix(".json"), {
            "task": args.task, "seed": seed, "control_step": step, "instruction": instruction,
            "m": state["m"], "metrics": metrics, "prompt_meta": state["prompt_meta"],
            "action_meta": state["action_meta"], "arrays_file": str(out.relative_to(ARTIFACT)),
            "audit": {"clean_shared": True, "m_shared": True,
                      "masks_exact": all(len(set(metrics[a]["selected"])) == state["m"] for a in ARMS)},
        })
        print(json.dumps({"task": args.task, "seed": seed, "step": step}), flush=True)


if __name__ == "__main__": main()
