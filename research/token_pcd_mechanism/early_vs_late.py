from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from research.token_pcd_stage_a.core import array_sha256, cosine, direction_metrics, read_jsonl, teacher_forced_logits, tensor_sha256


@contextmanager
def replace_patch_tokens(model, selected, means):
    selected = sorted(int(i) for i in selected)
    handles = []
    modules = [model.vision_backbone.featurizer.patch_embed]
    if model.config.use_fused_vision_backbone:
        modules.append(model.vision_backbone.fused_featurizer.patch_embed)
    for branch, module in enumerate(modules):
        def hook(_module, _inputs, output, branch=branch):
            if not selected:
                return output
            modified = output.clone()
            modified[:, selected, :] = means[branch].to(device=output.device, dtype=output.dtype)[selected]
            return modified
        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_inputs(processor, image_path, instruction, model):
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return processor(instruction, Image.fromarray(image)).to(model.device, dtype=torch.bfloat16)


def patch_means(model, processor, pcd_root, rows):
    totals = [None, None]; count = 0
    with torch.inference_mode():
        for row in rows:
            inputs = load_inputs(processor, pcd_root / row["image_path"], row["instruction"], model)
            pixels = inputs["pixel_values"]
            images = torch.split(pixels, [3, 3], dim=1)
            for branch, image in enumerate(images):
                patches = model.vision_backbone.featurizer.patch_embed(image) if branch == 0 else model.vision_backbone.fused_featurizer.patch_embed(image)
                value = patches[0].detach().float().cpu()
                totals[branch] = value if totals[branch] is None else totals[branch] + value
            count += 1
    return [value / count for value in totals]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pcd_root, stage_a, artifact = args.pcd_root.resolve(), args.stage_a.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint = pcd_root / "source/PCD/pretrained/openvla-7b"
    sys.path.insert(0, str(pcd_root / "source/PCD"))
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16,
                                                    low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True).cuda().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    states = read_jsonl(stage_a / "states.lock.jsonl")
    calibration = read_jsonl(stage_a / "calibration.lock.jsonl")
    means = patch_means(model, processor, pcd_root, calibration)
    torch.save({"means": means, "count": len(calibration), "sha256": [tensor_sha256(x) for x in means]}, artifact / "early_patch_means.pt")
    rows = states[:args.limit] if args.limit else states
    stage_results = {item["state_id"]: item for item in read_jsonl(stage_a / "results.jsonl")}
    output = artifact / "results.jsonl"
    with output.open("w") as handle:
        for ordinal, row in enumerate(rows, 1):
            mask = json.loads((stage_a / "masks" / f"{row['state_id']}.json").read_text())
            selected = mask["object_token_ids_25"]
            clean_inputs = load_inputs(processor, pcd_root / row["clean_path"], row["instruction"], model)
            clean_ids = torch.tensor([stage_results[row["state_id"]]["clean_action_tokens"]], device=model.device, dtype=clean_inputs["input_ids"].dtype)
            logits_np = np.load(stage_a / "logits" / f"{row['state_id']}.npz")
            clean_logits = logits_np["clean"]
            pixel_logits = logits_np["pixel_cf"]
            clean_t, pixel_t = torch.from_numpy(clean_logits), torch.from_numpy(pixel_logits)
            with replace_patch_tokens(model, selected, means):
                early_logits, trace = teacher_forced_logits(model, clean_inputs, clean_ids)
            early_delta = direction_metrics(model, clean_t, early_logits)
            pixel_delta = direction_metrics(model, clean_t, pixel_t)
            late_logits = torch.from_numpy(logits_np["object_token_cf"])
            late_delta = direction_metrics(model, clean_t, late_logits)
            result = {"state_id": row["state_id"], "task": row["task"], "object_token_ids": selected,
                      "early_pixel_cosine": cosine(early_delta, pixel_delta), "late_pixel_cosine": cosine(late_delta, pixel_delta),
                      "early_late_cosine": cosine(early_delta, late_delta),
                      "early_pixel_norm_ratio": float(torch.linalg.vector_norm(early_delta.double()) / torch.linalg.vector_norm(pixel_delta.double())) if torch.linalg.vector_norm(pixel_delta) else float("nan"),
                      "late_pixel_norm_ratio": float(torch.linalg.vector_norm(late_delta.double()) / torch.linalg.vector_norm(pixel_delta.double())) if torch.linalg.vector_norm(pixel_delta) else float("nan"),
                      "changed_indices": selected, "early_trace_shape": list(trace.before.shape) if trace.before is not None else None}
            handle.write(json.dumps(result, sort_keys=True) + "\n"); handle.flush()
            print(json.dumps({"ordinal": ordinal, "total": len(rows), **result}), flush=True)
    print(json.dumps({"status": "COMPLETE", "states": len(rows), "artifact": str(artifact)}))


if __name__ == "__main__": main()
