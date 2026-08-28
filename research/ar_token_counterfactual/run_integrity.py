"""Run pre-baseline integrity checks on one real LIBERO-Spatial state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .intervention import clean_action_token_ids, decode_action_ids, teacher_forced_forward, tensor_sha256
from .libero_runtime import load_policy, predict_action, prepare_agentview, set_determinism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    code_dir = workspace / "third_party/openvla/prismatic/extern/hf"
    set_determinism(7)
    model, processor = load_policy(checkpoint, code_dir)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        _, image = prepare_agentview(obs)
    finally:
        env.close()

    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    input_ids_hash = tensor_sha256(inputs["input_ids"])
    attention_mask_hash = tensor_sha256(inputs["attention_mask"])
    pixel_hash = tensor_sha256(inputs["pixel_values"])

    official_action_1 = predict_action(model, processor, image, task.language)
    official_action_2 = predict_action(model, processor, image, task.language)
    clean_ids_1 = clean_action_token_ids(model, inputs)
    clean_ids_2 = clean_action_token_ids(model, inputs)
    clean_1 = teacher_forced_forward(model, inputs, clean_ids_1, record_attention=True)
    clean_2 = teacher_forced_forward(model, inputs, clean_ids_1, record_attention=False)
    noop = teacher_forced_forward(model, inputs, clean_ids_1, selected_indices=(), replacement_mean=None)

    zero_mean = torch.zeros(clean_1.visual_token_count, clean_1.trace.before.shape[-1])
    masked = teacher_forced_forward(model, inputs, clean_ids_1, selected_indices=[0], replacement_mean=zero_mean)
    decoded = decode_action_ids(model, clean_ids_1)
    clean_repeat_diff = float(torch.max(torch.abs(clean_1.logits - clean_2.logits)).item())
    noop_diff = float(torch.max(torch.abs(clean_1.logits - noop.logits)).item())
    attention_finite = clean_1.attention_scores is not None and bool(torch.isfinite(clean_1.attention_scores).all())

    checks = {
        "frozen_eval_model": not model.training and all(not p.requires_grad for p in model.parameters()),
        "official_action_determinism": bool(np.array_equal(official_action_1, official_action_2)),
        "action_token_determinism": bool(torch.equal(clean_ids_1, clean_ids_2)),
        "teacher_forced_logit_determinism": clean_repeat_diff == 0.0,
        "empty_hook_logit_parity": noop_diff == 0.0,
        "changed_visual_index_exactness": masked.trace.changed_indices == (0,),
        "input_ids_unchanged": tensor_sha256(inputs["input_ids"]) == input_ids_hash,
        "attention_mask_unchanged": tensor_sha256(inputs["attention_mask"]) == attention_mask_hash,
        "pixel_values_unchanged": tensor_sha256(inputs["pixel_values"]) == pixel_hash,
        "official_decode_parity": bool(np.array_equal(decoded, official_action_1)),
        "all_logits_and_attention_finite": bool(torch.isfinite(clean_1.logits).all() and torch.isfinite(masked.logits).all() and attention_finite),
        "seven_action_positions": clean_1.logits.shape[1] == 7 and clean_ids_1.shape[1] == 7,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "task_id": 0,
        "init_state_index": 0,
        "task_description": task.language,
        "visual_token_count": clean_1.visual_token_count,
        "projector_shape": list(clean_1.trace.before.shape),
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "input_hashes": {"input_ids": input_ids_hash, "attention_mask": attention_mask_hash, "pixel_values": pixel_hash},
        "clean_projector_hash": tensor_sha256(clean_1.trace.before),
        "masked_projector_hash": tensor_sha256(masked.trace.after),
        "clean_repeat_max_abs_logit_diff": clean_repeat_diff,
        "noop_max_abs_logit_diff": noop_diff,
        "clean_action_token_ids": clean_ids_1[0].tolist(),
        "official_action": official_action_1.tolist(),
        "manual_decoded_action": decoded.tolist(),
        "note": "Changed-index mechanism check uses zero replacement; the formal phase repeats it with the locked calibration mean.",
    }
    path = artifact / "integrity_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "path": str(path)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
