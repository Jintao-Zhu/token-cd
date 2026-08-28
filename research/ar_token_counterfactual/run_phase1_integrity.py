"""Validate the locked calibration-mean intervention on one real Phase-1 state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .intervention import clean_action_token_ids, masked_action_token_ids, teacher_forced_forward, tensor_sha256
from .libero_runtime import build_prompt, load_policy, set_determinism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if not (artifact / "phase1_protocol.lock.yaml").is_file():
        raise RuntimeError("Phase-1 protocol must be locked first")
    state = json.loads((artifact / "phase1_state_manifest.jsonl").read_text().splitlines()[0])
    mean = torch.load(artifact / "position_conditioned_visual_mean.pt", map_location="cpu", weights_only=True)["mean"]
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)
    image = Image.open(artifact / state["image"]).convert("RGB")
    inputs = processor(build_prompt(state["task_description"]), image).to(model.device, dtype=torch.bfloat16)
    protected = {key: tensor_sha256(inputs[key]) for key in ("input_ids", "attention_mask", "pixel_values")}
    clean_ids_1 = clean_action_token_ids(model, inputs)
    clean_ids_hash = tensor_sha256(clean_ids_1)
    clean_ids_2 = clean_action_token_ids(model, inputs)
    clean_1 = teacher_forced_forward(model, inputs, clean_ids_1, record_attention=True)
    clean_2 = teacher_forced_forward(model, inputs, clean_ids_1)
    noop = teacher_forced_forward(model, inputs, clean_ids_1, [], None)
    masked_1 = teacher_forced_forward(model, inputs, clean_ids_1, [0], mean)
    masked_2 = teacher_forced_forward(model, inputs, clean_ids_1, [0], mean)
    free_ids_1, free_trace_1 = masked_action_token_ids(model, inputs, [0], mean)
    free_ids_2, free_trace_2 = masked_action_token_ids(model, inputs, [0], mean)
    checks = {
        "clean_action_token_determinism": bool(torch.equal(clean_ids_1, clean_ids_2)),
        "clean_teacher_forced_determinism": bool(torch.equal(clean_1.logits, clean_2.logits)),
        "masked_teacher_forced_determinism": bool(torch.equal(masked_1.logits, masked_2.logits)),
        "masked_free_running_determinism": bool(torch.equal(free_ids_1, free_ids_2)),
        "empty_intervention_parity": bool(torch.equal(clean_1.logits, noop.logits)),
        "teacher_changed_index_exactness": masked_1.trace.changed_indices == (0,),
        "free_changed_index_exactness": free_trace_1.changed_indices == (0,) and free_trace_2.changed_indices == (0,),
        "protected_inputs_unchanged": all(tensor_sha256(inputs[key]) == value for key, value in protected.items()),
        "teacher_clean_action_prefix_not_mutated": tensor_sha256(clean_ids_1) == clean_ids_hash,
        "all_finite": bool(
            torch.isfinite(clean_1.logits).all() and torch.isfinite(masked_1.logits).all()
            and torch.isfinite(clean_1.attention_scores).all()
        ),
        "visual_shape_matches_mean": tuple(clean_1.trace.before.shape[1:]) == tuple(mean.shape),
        "single_token_intervention": len(masked_1.trace.changed_indices) == 1,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "snapshot_id": state["snapshot_id"],
        "visual_token_count": clean_1.visual_token_count,
        "projector_shape": list(clean_1.trace.before.shape),
        "clean_projector_hash": tensor_sha256(clean_1.trace.before),
        "masked_projector_hash": tensor_sha256(masked_1.trace.after),
        "clean_action_token_ids": clean_ids_1[0].tolist(),
        "masked_free_action_token_ids": free_ids_1[0].tolist(),
        "clean_masked_teacher_max_abs_logit_delta": float(torch.max(torch.abs(clean_1.logits - masked_1.logits)).item()),
        "protected_hashes": protected,
    }
    path = artifact / "phase1_integrity_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "path": str(path)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
