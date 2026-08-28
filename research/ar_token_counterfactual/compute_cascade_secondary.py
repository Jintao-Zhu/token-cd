"""Complete the preregistered cascade secondary metric without changing primary results."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .intervention import decode_action_ids, teacher_forced_forward
from .libero_runtime import build_prompt, load_policy, set_determinism


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if not (artifact / "amendment_001_cascade_secondary.yaml").is_file():
        raise RuntimeError("Cascade schema amendment must be written before this secondary rerun")
    raw = [json.loads(line) for line in (artifact / "token_effects.jsonl").read_text().splitlines() if line]
    by_token = defaultdict(list)
    for row in raw:
        by_token[(row["snapshot_id"], row["visual_token_idx"])].append(row)
    if len(by_token) != 2400 or any(len(rows) != 7 for rows in by_token.values()):
        raise RuntimeError("Primary raw rows are incomplete")
    states = {row["snapshot_id"]: row for row in (
        json.loads(line) for line in (artifact / "phase1_state_manifest.jsonl").read_text().splitlines() if line
    )}
    mean = torch.load(artifact / "position_conditioned_visual_mean.pt", map_location="cpu", weights_only=True)["mean"]
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)
    if not (artifact / "amendment_002_cascade_teacher_argmax.yaml").is_file():
        raise RuntimeError("Teacher-argmax correction amendment must exist")
    output = artifact / "cascade_effects_v2.jsonl"
    completed = set()
    if output.exists():
        completed = {(row["snapshot_id"], row["visual_token_idx"]) for row in (
            json.loads(line) for line in output.read_text().splitlines() if line
        )}
    tokens_by_state = defaultdict(list)
    for key in by_token:
        tokens_by_state[key[0]].append(key[1])
    for snapshot_id, token_indices in tokens_by_state.items():
        state = states[snapshot_id]
        image = Image.open(artifact / state["image"]).convert("RGB")
        inputs = processor(build_prompt(state["task_description"]), image).to(model.device, dtype=torch.bfloat16)
        first_rows = sorted(by_token[(snapshot_id, token_indices[0])], key=lambda row: row["action_position"])
        clean_ids = torch.tensor([[row["clean_action_token_id"] for row in first_rows]], device=model.device, dtype=torch.long)
        clean_tf = teacher_forced_forward(model, inputs, clean_ids)
        clean_tf_ids = clean_tf.logits.argmax(-1).to(model.device)
        clean_action = decode_action_ids(model, clean_tf_ids)
        for token_index in sorted(token_indices):
            key = (snapshot_id, token_index)
            if key in completed:
                continue
            masked_tf = teacher_forced_forward(model, inputs, clean_ids, [token_index], mean)
            masked_tf_ids = masked_tf.logits.argmax(-1).to(model.device)
            masked_tf_action = decode_action_ids(model, masked_tf_ids)
            teacher_l2 = float(np.linalg.norm(clean_action - masked_tf_action))
            primary_rows = by_token[key]
            free_l2_values = {float(row["free_running_action_l2"]) for row in primary_rows}
            if len(free_l2_values) != 1:
                raise RuntimeError("Inconsistent free-running effect within token rows")
            free_l2 = free_l2_values.pop()
            append_jsonl(output, {
                "snapshot_id": snapshot_id,
                "task_id": state["task_id"],
                "phase": state["phase"],
                "visual_token_idx": token_index,
                "selection_category": primary_rows[0]["selection_category"],
                "clean_generated_action_token_ids": clean_ids[0].tolist(),
                "clean_teacher_argmax_token_ids": clean_tf_ids[0].tolist(),
                "masked_teacher_argmax_token_ids": masked_tf_ids[0].tolist(),
                "clean_teacher_vs_generated_mismatch": bool(not torch.equal(clean_tf_ids, clean_ids)),
                "teacher_forced_argmax_action_l2": teacher_l2,
                "free_running_action_l2": free_l2,
                "cascade_amplification": free_l2 / (teacher_l2 + 1.0e-8),
                "epsilon": 1.0e-8,
                "all_finite": bool(np.isfinite([teacher_l2, free_l2, free_l2 / (teacher_l2 + 1.0e-8)]).all()),
            })
        print(json.dumps({"snapshot_id": snapshot_id, "completed_tokens": 16}), flush=True)


if __name__ == "__main__":
    main()
