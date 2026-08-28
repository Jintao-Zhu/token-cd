from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import ensure_empty_action_token
from research.token_pcd_stage_a.core import action_token_slice, append_jsonl, file_sha256, read_jsonl
from research.token_pcd_stage_a.run import inputs_for, load_model


LLM_LAYERS = (8, 16, 24)
OBJECT_THRESHOLD = 0.25
VISUAL_TOKENS = 256


def output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def spatial_patches(output):
    value = output_tensor(output)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < VISUAL_TOKENS:
        raise RuntimeError(f"Cannot extract 256 spatial patches from {tuple(value.shape)}")
    return value[:, -VISUAL_TOKENS:, :]


@contextmanager
def vision_capture(model):
    traces = {}
    handles = []
    branches = (("dino", model.vision_backbone.featurizer),
                ("siglip", model.vision_backbone.fused_featurizer))

    def save(name, spatial=False):
        def hook(_module, _inputs, output):
            value = spatial_patches(output) if spatial else output_tensor(output)
            traces[name] = value.detach().float().cpu().clone()
        return hook

    for name, branch in branches:
        middle = len(branch.blocks) // 2
        handles.append(branch.patch_embed.register_forward_hook(save(f"{name}_patch")))
        handles.append(branch.blocks[middle - 1].register_forward_hook(save(f"{name}_middle", spatial=True)))
        handles.append(branch.blocks[-1].register_forward_hook(save(f"{name}_last", spatial=True)))
    handles.append(model.projector.register_forward_hook(save("projector")))
    try:
        yield traces
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def captured_forward(model, inputs, clean_ids):
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    teacher_ids = torch.cat((base_ids, clean_ids[:, :-1]), dim=1)
    teacher_mask = torch.cat((base_mask, torch.ones_like(clean_ids[:, :-1], dtype=base_mask.dtype)), dim=1)
    with vision_capture(model) as vision:
        output = model(input_ids=teacher_ids, attention_mask=teacher_mask, pixel_values=inputs["pixel_values"],
                       use_cache=False, output_hidden_states=True, return_dict=True)
    missing = {"dino_patch", "siglip_patch", "dino_middle", "siglip_middle", "dino_last",
               "siglip_last", "projector"} - set(vision)
    if missing:
        raise RuntimeError(f"Capture hooks not invoked: {sorted(missing)}")
    if vision["projector"].shape[1:] != (VISUAL_TOKENS, model.config.text_config.hidden_size):
        raise RuntimeError(f"Unexpected projector shape {tuple(vision['projector'].shape)}")
    query = [VISUAL_TOKENS + base_ids.shape[1] - 1 + offset for offset in range(clean_ids.shape[1])]
    logits = output.logits[0, query, action_token_slice(model)].detach().float().cpu()
    llm = {f"llm_{layer}": output.hidden_states[layer][0, 1:1 + VISUAL_TOKENS].detach().float().cpu().clone()
           for layer in LLM_LAYERS}
    return vision, llm, logits, teacher_ids, teacher_mask


def fused(vision, stage):
    return torch.cat((vision[f"dino_{stage}"][0], vision[f"siglip_{stage}"][0]), dim=-1)


def energy_record(delta, object_ids):
    energy = delta.double().square().sum(dim=-1)
    total = energy.sum()
    object_energy = energy[object_ids].sum() if object_ids else torch.tensor(0.0, dtype=energy.dtype)
    return {"object_energy": float(object_energy), "total_energy": float(total),
            "object_energy_ratio": float(object_energy / total) if total > 0 else float("nan")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pcd_root, stage_a, artifact = args.pcd_root.resolve(), args.stage_a.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True); (artifact / "states").mkdir(exist_ok=True)
    checkpoint = pcd_root / "source/PCD/pretrained/openvla-7b"
    model, processor = load_model(checkpoint)
    stage_results = {row["state_id"]: row for row in read_jsonl(stage_a / "results.jsonl")}
    states = read_jsonl(stage_a / "states.lock.jsonl")
    if args.limit is not None:
        states = states[:args.limit]
    output = artifact / "extraction.jsonl"
    completed = {row["state_id"] for row in read_jsonl(output)} if output.exists() else set()
    for ordinal, row in enumerate(states, 1):
        if row["state_id"] in completed:
            continue
        mask = json.loads((stage_a / "masks" / f"{row['state_id']}.json").read_text())
        overlaps = np.asarray(mask["token_overlap_ratio"], dtype=np.float32)
        object_ids = np.flatnonzero(overlaps >= OBJECT_THRESHOLD).astype(int).tolist()
        clean_inputs = inputs_for(processor, model, pcd_root / row["clean_path"], row["instruction"])
        pixel_inputs = inputs_for(processor, model, pcd_root / row["pixel_path"], row["instruction"])
        if not torch.equal(clean_inputs["input_ids"], pixel_inputs["input_ids"]):
            raise RuntimeError("Clean and Pixel language tokens differ")
        clean_ids = torch.tensor([stage_results[row["state_id"]]["clean_action_tokens"]],
                                 device=model.device, dtype=clean_inputs["input_ids"].dtype)
        clean_vision, clean_llm, clean_logits, teacher_ids, teacher_mask = captured_forward(model, clean_inputs, clean_ids)
        pixel_vision, pixel_llm, pixel_logits, pixel_teacher_ids, pixel_teacher_mask = captured_forward(model, pixel_inputs, clean_ids)
        if not torch.equal(teacher_ids, pixel_teacher_ids) or not torch.equal(teacher_mask, pixel_teacher_mask):
            raise RuntimeError("Teacher-forced language context differs")
        residuals = {
            "patch": fused(clean_vision, "patch") - fused(pixel_vision, "patch"),
            "vision_middle": fused(clean_vision, "middle") - fused(pixel_vision, "middle"),
            "vision_last": fused(clean_vision, "last") - fused(pixel_vision, "last"),
            "projector": clean_vision["projector"][0] - pixel_vision["projector"][0],
            **{name: clean_llm[name] - pixel_llm[name] for name in clean_llm},
        }
        if any(value.shape[0] != VISUAL_TOKENS or not torch.isfinite(value).all() for value in residuals.values()):
            raise RuntimeError(f"Invalid residual for {row['state_id']}")
        stage_npz = np.load(stage_a / "logits" / f"{row['state_id']}.npz")
        expected_clean = torch.from_numpy(stage_npz["clean"][:, action_token_slice(model)].copy())
        expected_pixel = torch.from_numpy(stage_npz["pixel_cf"][:, action_token_slice(model)].copy())
        stage_npz.close()
        clean_error = float((clean_logits - expected_clean).abs().max())
        pixel_error = float((pixel_logits - expected_pixel).abs().max())
        if clean_error != 0 or pixel_error != 0:
            raise RuntimeError(f"Stage-A logit parity failed: clean={clean_error}, pixel={pixel_error}")
        state_path = artifact / "states" / f"{row['state_id']}.npz"
        arrays = {"projector_clean": clean_vision["projector"][0].numpy().astype(np.float32),
                  "projector_delta": residuals["projector"].numpy().astype(np.float32),
                  "clean_logits": clean_logits.numpy().astype(np.float32),
                  "pixel_logits": pixel_logits.numpy().astype(np.float32),
                  "overlaps": overlaps}
        arrays.update({f"delta_{name}": value.numpy().astype(np.float16)
                       for name, value in residuals.items() if name != "projector"})
        np.savez(state_path, **arrays)
        record = {"state_id": row["state_id"], "task": row["task"], "seed": row["seed"],
                  "object_threshold": OBJECT_THRESHOLD, "object_token_ids": object_ids,
                  "object_token_count": len(object_ids), "teacher_ids": teacher_ids[0].cpu().tolist(),
                  "clean_action_tokens": clean_ids[0].cpu().tolist(),
                  "locations": {name: {"shape": list(value.shape), **energy_record(value, object_ids)}
                                for name, value in residuals.items()},
                  "stage_a_clean_logit_max_abs_error": clean_error,
                  "stage_a_pixel_logit_max_abs_error": pixel_error,
                  "state_file": str(state_path.relative_to(artifact)), "state_file_sha256": file_sha256(state_path)}
        append_jsonl(output, record)
        print(json.dumps({"ordinal": ordinal, "total": len(states), "state_id": row["state_id"],
                          "projector_object_energy_ratio": record["locations"]["projector"]["object_energy_ratio"]}), flush=True)


if __name__ == "__main__":
    main()
