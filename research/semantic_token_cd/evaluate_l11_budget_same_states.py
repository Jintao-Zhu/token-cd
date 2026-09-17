"""Evaluate Matched, TopP80 and TopP85 on the same 60 saved observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, jsonable
from research.semantic_token_cd.prompt_attn_l11_count_rollout import build_policies as build_count
from research.semantic_token_cd.prompt_attn_l11_top_p_rollout import build_policies as build_top_p
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
ARMS = ("l11_matched", "l11_top_p80", "l11_top_p85")


def source_states(task: str) -> list[Path]:
    budget = Path("artifacts/prompt_attn_l11_budget_diagnostic_v1")
    manifest = json.loads((budget / "selection_manifest.json").read_text())["tasks"][task]
    roots = (budget / "states", Path("artifacts/prompt_attn_layer_selection_v1/states"))
    result = []
    for seed in manifest:
        found = []
        for root in roots:
            found = sorted((root / task / f"seed_{seed:03d}").glob("*.json"))
            if found:
                break
        if len(found) != 3:
            raise RuntimeError(f"expected three states for {task}/{seed}, got {len(found)}")
        result.extend(found)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**config)
    policies = {
        **build_count(base, args.task, ("l11_matched",)),
        **build_top_p(base, args.task, ("l11_top_p80", "l11_top_p85")),
    }
    for policy in policies.values():
        policy.save_prompt_attention = True

    for source_json in source_states(args.task):
        source = json.loads(source_json.read_text())
        source_npz = np.load(source_json.with_suffix(".npz"))
        image = source_npz["image"]
        instruction = source["instruction"]
        seed = int(source["seed"])
        step = int(source.get("source_step", source.get("step")))
        expected_m = int(source.get("m", source.get("matched_count", int(source_npz["standard_mask"].sum()))))
        state_id = source.get("state_id", f"{args.task}__seed{seed:03d}__step{step:03d}")
        out_dir = artifact / "same_state_eval" / args.task / f"seed_{seed:03d}"
        out_json = out_dir / f"step_{step:03d}.json"
        out_npz = out_json.with_suffix(".npz")
        if out_json.exists() and out_npz.exists():
            print(json.dumps({"skip": state_id}), flush=True)
            continue

        metrics = {}; arrays = {"image": image}
        for arm in ARMS:
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._selector_step = step
            policy._episode_trace = []; policy._episode_logits = []
            policy.step(image, None, instruction)
            meta = policy._episode_trace[-1]
            record = policy._episode_logits[-1]
            metrics[arm] = meta
            for key, value in record.items():
                arrays[f"{arm}__{key}"] = value

        matched_count_reproduced = metrics["l11_matched"]["actual_selected_count"] == expected_m
        positive = [arrays[f"{arm}__positive"] for arm in ARMS]
        if not all(np.array_equal(positive[0], value) for value in positive[1:]):
            raise RuntimeError(f"clean positive logits differ across arms for {state_id}")
        attention = [arrays[f"{arm}__prompt_attention"] for arm in ARMS]
        if not all(np.array_equal(attention[0], value) for value in attention[1:]):
            raise RuntimeError(f"L11 attention differs across arms for {state_id}")

        sets = {arm: set(metrics[arm]["selected_token_ids"]) for arm in ARMS}
        differences = {}
        for arm in ("l11_top_p80", "l11_top_p85"):
            differences[arm] = {
                "matched_minus_top_p": sorted(sets["l11_matched"] - sets[arm]),
                "top_p_minus_matched": sorted(sets[arm] - sets["l11_matched"]),
                "intersection_count": len(sets["l11_matched"] & sets[arm]),
                "nested": sets["l11_matched"] <= sets[arm] or sets[arm] <= sets["l11_matched"],
            }
            if not differences[arm]["nested"]:
                raise RuntimeError(f"shared L11 ranking failed nesting for {state_id}/{arm}")

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **arrays)
        atomic_json(out_json, {
            "protocol_id": "L11_MATCHED_TOP_P_SAME_STATE_DIAGNOSTIC_V1",
            "state_id": state_id, "task": args.task, "seed": seed, "source_step": step,
            "phase": source.get("phase"), "instruction": instruction,
            "source_json": str(source_json.resolve()), "expected_matched_count": expected_m,
            "matched_count_reproduced_from_legacy_state": matched_count_reproduced,
            "metrics": jsonable(metrics), "differences": differences,
            "clean_positive_bit_equal": True, "l11_attention_bit_equal": True,
            "source_rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
            "arrays_file": out_npz.name,
        })
        print(json.dumps({"state": state_id, "counts": {
            arm: metrics[arm]["actual_selected_count"] for arm in ARMS
        }}), flush=True)


if __name__ == "__main__":
    main()
