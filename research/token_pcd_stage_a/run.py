from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from research.ar_token_counterfactual.intervention import projector_intervention
from .core import (ALPHA, append_jsonl, array_sha256, clean_action_ids, cosine, direction_metrics,
                   decode_action_ids, file_sha256, matched_random_ids, pcd_ids, read_jsonl, teacher_forced_logits,
                   tensor_sha256, write_json, ensure_empty_action_token)


def load_model(checkpoint: Path):
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def inputs_for(processor, model, image_path: Path, instruction: str):
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return processor(instruction, Image.fromarray(resized)).to(model.device, dtype=torch.bfloat16)


def compute_mean(model, processor, pcd_root: Path, artifact: Path) -> torch.Tensor:
    output = artifact / "position_conditioned_visual_mean.pt"
    if output.exists():
        return torch.load(output, map_location="cpu", weights_only=True)["mean"]
    rows = read_jsonl(artifact / "calibration.lock.jsonl")
    total = None
    for ordinal, row in enumerate(rows, 1):
        inputs = inputs_for(processor, model, pcd_root / row["image_path"], row["instruction"])
        with torch.inference_mode(), projector_intervention(model) as trace:
            model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                  pixel_values=inputs["pixel_values"], use_cache=False, return_dict=True)
        if trace.before is None or trace.before.shape[1] != 256:
            raise RuntimeError("Calibration projector shape is not [1,256,D]")
        features = trace.before[0].double()
        total = features if total is None else total + features
        print(json.dumps({"calibration": ordinal, "total": len(rows)}), flush=True)
    mean = (total / len(rows)).float()
    torch.save({"mean": mean, "count": len(rows), "shape": list(mean.shape)}, output)
    write_json(artifact / "position_mean.json", {"count": len(rows), "shape": list(mean.shape),
                                                  "tensor_sha256": tensor_sha256(mean),
                                                  "file_sha256": file_sha256(output)})
    return mean


def branch_record(model, clean_logits, branch_logits):
    delta = direction_metrics(model, clean_logits, branch_logits)
    return delta, float(torch.linalg.vector_norm(delta.double()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--integrity-only", action="store_true")
    args = parser.parse_args()
    pcd_root, artifact = args.pcd_root.resolve(), args.artifact.resolve()
    if not (artifact / "protocol.lock.json").is_file() or not (artifact / "mask_capture_report.json").is_file():
        raise RuntimeError("Protocol lock and complete SAM mask capture are required")
    torch.manual_seed(20260814); torch.cuda.manual_seed_all(20260814)
    checkpoint = pcd_root / "source/PCD/pretrained/openvla-7b"
    sys.path.insert(0, str(pcd_root / "source/PCD"))
    model, processor = load_model(checkpoint)
    if args.preflight_only:
        row = read_jsonl(artifact / "states.lock.jsonl")[0]
        inputs = inputs_for(processor, model, pcd_root / row["clean_path"], row["instruction"])
        ids1, ids2 = clean_action_ids(model, inputs), clean_action_ids(model, inputs)
        unnorm_key = "fractal20220817_data" if row["task"].startswith("google_robot") else "bridge_orig"
        manual_action = decode_action_ids(model, ids1[0], unnorm_key)
        official_ids, official_mask = ensure_empty_action_token(inputs)
        official_action = model.predict_action(input_ids=official_ids, attention_mask=official_mask,
                                               pixel_values=inputs["pixel_values"], unnorm_key=unnorm_key,
                                               do_sample=False)
        report = {"status": "PASS", "state_id": row["state_id"],
                  "frozen_eval_model": (not model.training) and all(not p.requires_grad for p in model.parameters()),
                  "clean_generation_deterministic": bool(torch.equal(ids1, ids2)),
                  "official_manual_decode_exact_parity": bool(np.array_equal(manual_action, official_action)),
                  "clean_action_tokens": ids1[0].cpu().tolist(), "manual_action": manual_action.tolist(),
                  "official_predict_action": official_action.tolist()}
        if not all(value for key, value in report.items() if key in ("frozen_eval_model", "clean_generation_deterministic", "official_manual_decode_exact_parity")):
            report["status"] = "FAIL"
        write_json(artifact / "model_preflight.json", report)
        print(json.dumps(report, sort_keys=True)); return
    if not (artifact / "model_preflight.json").is_file() or json.loads((artifact / "model_preflight.json").read_text())["status"] != "PASS":
        raise RuntimeError("Passing model preflight is required before calibration/intervention")
    mean = compute_mean(model, processor, pcd_root, artifact)
    states = read_jsonl(artifact / "states.lock.jsonl")
    completed = {row["state_id"] for row in read_jsonl(artifact / "results.jsonl")} if (artifact / "results.jsonl").exists() else set()
    if args.integrity_only:
        states = states[:1]
    elif args.limit is not None:
        states = states[:args.limit]
    for row in states:
        if row["state_id"] in completed:
            continue
        mask_record = json.loads((artifact / "masks" / f"{row['state_id']}.json").read_text())
        object_ids = mask_record["object_token_ids_25"]
        object_ids50 = mask_record["object_token_ids_50"]
        random_seed = 20260814 + int(hashlib.sha256(row["state_id"].encode()).hexdigest()[:8], 16)
        random_ids = matched_random_ids(object_ids, random_seed)
        random_ids50 = matched_random_ids(object_ids50, random_seed + 1)
        clean_inputs = inputs_for(processor, model, pcd_root / row["clean_path"], row["instruction"])
        pixel_inputs = inputs_for(processor, model, pcd_root / row["pixel_path"], row["instruction"])
        if not torch.equal(clean_inputs["input_ids"], pixel_inputs["input_ids"]):
            raise RuntimeError("Clean and Pixel-CF language tokens differ")
        protected = {name: tensor_sha256(clean_inputs[name]) for name in ("input_ids", "attention_mask", "pixel_values")}
        clean_ids = clean_action_ids(model, clean_inputs)
        clean_logits, clean_trace = teacher_forced_logits(model, clean_inputs, clean_ids)
        pixel_logits, pixel_trace = teacher_forced_logits(model, pixel_inputs, clean_ids)
        object_logits, object_trace = teacher_forced_logits(model, clean_inputs, clean_ids, object_ids, mean)
        random_logits, random_trace = teacher_forced_logits(model, clean_inputs, clean_ids, random_ids, mean)
        object_logits50, object_trace50 = teacher_forced_logits(model, clean_inputs, clean_ids, object_ids50, mean)
        random_logits50, random_trace50 = teacher_forced_logits(model, clean_inputs, clean_ids, random_ids50, mean)
        pixel_delta, pixel_norm = branch_record(model, clean_logits, pixel_logits)
        object_delta, object_norm = branch_record(model, clean_logits, object_logits)
        random_delta, random_norm = branch_record(model, clean_logits, random_logits)
        object_delta50, object_norm50 = branch_record(model, clean_logits, object_logits50)
        random_delta50, random_norm50 = branch_record(model, clean_logits, random_logits50)
        pixel_pcd, token_pcd = pcd_ids(clean_logits, pixel_logits), pcd_ids(clean_logits, object_logits)
        logits_path = artifact / "logits" / f"{row['state_id']}.npz"
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(logits_path, clean=clean_logits.numpy(), pixel_cf=pixel_logits.numpy(),
                            object_token_cf=object_logits.numpy(), random_token_cf=random_logits.numpy(),
                            object_token_cf_50=object_logits50.numpy(), random_token_cf_50=random_logits50.numpy())
        checks = {
            "original_256_token_shape": list(clean_trace.before.shape),
            "modified_256_token_shape": list(object_trace.after.shape),
            "actual_changed_indices": list(object_trace.changed_indices),
            "random_actual_changed_indices": list(random_trace.changed_indices),
            "object_50_actual_changed_indices": list(object_trace50.changed_indices),
            "random_50_actual_changed_indices": list(random_trace50.changed_indices),
            "protected_inputs_unchanged": all(tensor_sha256(clean_inputs[name]) == digest for name, digest in protected.items()),
            "random_count_matched": len(random_ids) == len(object_ids),
            "object_indices_exact": list(object_trace.changed_indices) == object_ids,
            "random_indices_exact": list(random_trace.changed_indices) == random_ids,
            "object_50_indices_exact": list(object_trace50.changed_indices) == object_ids50,
            "random_50_indices_exact": list(random_trace50.changed_indices) == random_ids50,
            "all_finite": all(torch.isfinite(value).all().item() for value in (clean_logits, pixel_logits, object_logits, random_logits, object_logits50, random_logits50)),
        }
        if not all(value is True or isinstance(value, list) for value in checks.values()) or not all(checks[key] for key in checks if isinstance(checks[key], bool)):
            raise RuntimeError(f"Integrity failure for {row['state_id']}: {checks}")
        result = {"state_id": row["state_id"], "task": row["task"], "seed": row["seed"],
                  "rgb_hash": row["rgb_array_sha256"], "sam_mask_hash": mask_record["sam_target_mask_sha256"],
                  "preprocessed_mask": mask_record["token_overlap_ratio"], "object_token_ids": object_ids,
                  "random_token_ids": random_ids, "object_token_ids_50": object_ids50, "random_token_ids_50": random_ids50,
                  "token_overlap_ratio": [mask_record["token_overlap_ratio"][i] for i in object_ids],
                  "clean_action_tokens": clean_ids[0].cpu().tolist(), "pixel_pcd_action_tokens": pixel_pcd.tolist(),
                  "token_pcd_action_tokens": token_pcd.tolist(), "random_pcd_action_tokens": pcd_ids(clean_logits, random_logits).tolist(),
                  "object_pixel_cosine": cosine(object_delta, pixel_delta),
                  "random_pixel_cosine": cosine(random_delta, pixel_delta),
                  "object_pixel_norm_ratio": object_norm / pixel_norm if pixel_norm else float("nan"),
                  "random_pixel_norm_ratio": random_norm / pixel_norm if pixel_norm else float("nan"),
                  "object_pixel_cosine_50": cosine(object_delta50, pixel_delta),
                  "random_pixel_cosine_50": cosine(random_delta50, pixel_delta),
                  "object_pixel_norm_ratio_50": object_norm50 / pixel_norm if pixel_norm else float("nan"),
                  "random_pixel_norm_ratio_50": random_norm50 / pixel_norm if pixel_norm else float("nan"),
                  "pcd_position_agreement": float((pixel_pcd == token_pcd).float().mean()),
                  "pcd_full_sequence_match": bool(torch.equal(pixel_pcd, token_pcd)),
                  "logits_path": str(logits_path.relative_to(artifact)), "logits_sha256": file_sha256(logits_path),
                  "logits_storage_dtype": "float32", "alpha": ALPHA, "checks": checks}
        if args.integrity_only:
            empty_logits, empty_trace = teacher_forced_logits(model, clean_inputs, clean_ids, (), mean)
            clean_logits_2, _ = teacher_forced_logits(model, clean_inputs, clean_ids)
            integrity = {"hook_off_teacher_forced_exact_parity": bool(torch.equal(clean_logits, clean_logits_2)),
                         "empty_replacement_exact_parity": bool(torch.equal(clean_logits, empty_logits)),
                         "empty_changed_indices": list(empty_trace.changed_indices),
                         "alpha_zero_exact_parity": bool(torch.equal(pcd_ids(clean_logits, object_logits, 0.0), clean_logits.argmax(-1))),
                         **checks}
            write_json(artifact / "integrity_sentinel.json", {"status": "PASS" if all(v for v in integrity.values() if isinstance(v, bool)) else "FAIL",
                                                               "state_id": row["state_id"], "checks": integrity})
            print(json.dumps(integrity, sort_keys=True)); return
        append_jsonl(artifact / "results.jsonl", result)
        print(json.dumps({"state_id": row["state_id"], "object_cos": result["object_pixel_cosine"],
                          "random_cos": result["random_pixel_cosine"]}), flush=True)


if __name__ == "__main__":
    main()
