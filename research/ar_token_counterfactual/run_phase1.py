"""Run locked teacher-forced and free-running AR token counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .intervention import (
    action_logit_metrics,
    clean_action_token_ids,
    decode_action_ids,
    masked_action_token_ids,
    select_visual_tokens,
    teacher_forced_forward,
    tensor_sha256,
)
from .libero_runtime import build_prompt, load_policy, set_determinism


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if not (artifact / "phase1_protocol.lock.yaml").is_file():
        raise RuntimeError("phase1_protocol.lock.yaml must exist before intervention outcomes are computed")
    states = [json.loads(line) for line in (artifact / "phase1_state_manifest.jsonl").read_text().splitlines() if line]
    mean_payload = torch.load(artifact / "position_conditioned_visual_mean.pt", map_location="cpu", weights_only=True)
    replacement_mean = mean_payload["mean"]
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)
    output_path = artifact / "token_effects.jsonl"
    completed = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            row = json.loads(line)
            completed.add((row["snapshot_id"], row["visual_token_idx"]))

    for state in states:
        image = Image.open(artifact / state["image"]).convert("RGB")
        inputs = processor(build_prompt(state["task_description"]), image).to(model.device, dtype=torch.bfloat16)
        protected_hashes = {key: tensor_sha256(inputs[key]) for key in ("input_ids", "attention_mask", "pixel_values")}
        clean_ids = clean_action_token_ids(model, inputs)
        clean_tf = teacher_forced_forward(model, inputs, clean_ids, record_attention=True)
        if clean_tf.visual_token_count != replacement_mean.shape[0]:
            raise RuntimeError("Replacement mean visual-token dimension mismatch")
        seed = 20260808 + int(hashlib.sha256(state["snapshot_id"].encode()).hexdigest()[:8], 16)
        categories = select_visual_tokens(clean_tf.attention_scores, seed)
        rank_order = torch.argsort(clean_tf.attention_scores, descending=True)
        ranks = torch.empty_like(rank_order)
        ranks[rank_order] = torch.arange(rank_order.numel())
        clean_action = decode_action_ids(model, clean_ids)
        category_by_token = {token: category for category, tokens in categories.items() for token in tokens}

        for token_index in sum(categories.values(), []):
            if (state["snapshot_id"], token_index) in completed:
                continue
            started = time.perf_counter()
            masked_tf = teacher_forced_forward(model, inputs, clean_ids, [token_index], replacement_mean)
            masked_ids, free_trace = masked_action_token_ids(model, inputs, [token_index], replacement_mean)
            masked_action = decode_action_ids(model, masked_ids)
            metrics = action_logit_metrics(clean_tf.logits, masked_tf.logits)
            numeric_arrays = [clean_tf.logits.numpy(), masked_tf.logits.numpy(), clean_action, masked_action]
            numeric_arrays.extend(value for key, value in metrics.items() if key != "argmax_flip")
            if not all(np.isfinite(value).all() for value in numeric_arrays):
                raise RuntimeError("Nonfinite logit, action, or metric detected")
            if masked_tf.trace.changed_indices != (token_index,) or free_trace.changed_indices != (token_index,):
                raise RuntimeError("Changed-index audit failed with locked calibration mean")
            if any(tensor_sha256(inputs[key]) != value for key, value in protected_hashes.items()):
                raise RuntimeError("Protected model input mutated during intervention")
            runtime_ms = (time.perf_counter() - started) * 1000.0
            free_hamming = int(torch.sum(clean_ids != masked_ids).item())
            free_l2 = float(np.linalg.norm(clean_action - masked_action))
            rows = []
            for action_position in range(7):
                rows.append({
                    **state,
                    "visual_token_idx": token_index,
                    "selection_category": category_by_token[token_index],
                    "attention_rank": int(ranks[token_index].item()),
                    "attention_score": float(clean_tf.attention_scores[token_index].item()),
                    "action_position": action_position,
                    "js_div": float(metrics["js_div"][0, action_position]),
                    "kl_clean_mask": float(metrics["kl_clean_mask"][0, action_position]),
                    "argmax_flip": bool(metrics["argmax_flip"][0, action_position]),
                    "clean_margin": float(metrics["clean_margin"][0, action_position]),
                    "masked_margin": float(metrics["masked_margin"][0, action_position]),
                    "clean_action_token_id": int(clean_ids[0, action_position].item()),
                    "masked_free_action_token_id": int(masked_ids[0, action_position].item()),
                    "decoded_action_delta": float(abs(clean_action[action_position] - masked_action[action_position])),
                    "free_running_sequence_hamming": free_hamming,
                    "free_running_action_l2": free_l2,
                    "clean_projector_hash": tensor_sha256(clean_tf.trace.before),
                    "masked_projector_hash": tensor_sha256(masked_tf.trace.after),
                    "runtime_ms_per_token": runtime_ms,
                    "all_finite": True,
                })
            append_jsonl(output_path, rows)
            completed.add((state["snapshot_id"], token_index))
        print(json.dumps({"snapshot_id": state["snapshot_id"], "completed_tokens": 16}), flush=True)


if __name__ == "__main__":
    main()
