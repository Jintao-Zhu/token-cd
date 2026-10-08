"""Offline prompt-to-image attention layer scan for the INTACT Pi0 BridgeV2 model.

Uses the same original/synonym/target-switch diagnostic as the existing
SpatialVLA and pi0-FAST layer scans. It reads fixed SIMPLER states and does not
launch an environment or modify the deployed LeRobot runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
SOURCE = REPO / "artifacts/spatialvla_layer_selection_v1b/states"
OUT = REPO / "artifacts/pi0_intact_layer_selection_v1"
CHECKPOINT = ROOT / "checkpoints/pi0_intact_bridge_simpler"
TASKS = (
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
SEEDS = (100, 110, 120, 130, 150, 160)
N_LAYERS = 18

SYNONYMS = {
    "widowx_carrot_on_plate": "place the carrot onto the plate",
    "widowx_put_eggplant_in_basket": "place the eggplant inside the yellow basket",
    "widowx_spoon_on_towel": "place the spoon onto the towel",
    "widowx_stack_cube": "put the green block on top of the yellow block",
}


def _find_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("empty prompt token sequence")
    for i in range(len(values) - len(needle) + 1):
        if values[i : i + len(needle)] == needle:
            return i
    raise RuntimeError("could not find task phrase tokens in prepared Pi0 prompt")


def _patch_mask(seg: np.ndarray, actor_id: int) -> np.ndarray:
    """Map source segmentation to Pi0's exact 224 square, top/left padded view."""
    import cv2

    height, width = seg.shape
    ratio = max(width / 224.0, height / 224.0)
    resized_h, resized_w = int(height / ratio), int(width / ratio)
    mask = cv2.resize(
        (seg == actor_id).astype(np.uint8),
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )
    pad_h, pad_w = 224 - resized_h, 224 - resized_w
    # LeRobot Pi0 resize_with_pad pads on the top and left.
    padded = np.pad(mask, ((pad_h, 0), (pad_w, 0)), constant_values=0)
    if padded.shape != (224, 224):
        raise RuntimeError(f"unexpected padded segmentation shape {padded.shape}")
    return padded.reshape(16, 14, 16, 14).any(axis=(1, 3)).reshape(256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 scans all 24 states")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "pi0_intact_runtime/lerobot_intact"))
    sys.path.insert(0, str(REPO))

    import torch
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy, make_att_2d_masks
    from lerobot.common.constants import OBS_ROBOT

    torch.set_num_threads(1)
    # This host's installed cuDNN SDPA backend rejects the SigLIP attention
    # graph used by this checkpoint. Keep CUDA inference enabled and let
    # PyTorch dispatch to its supported non-cuDNN SDPA implementation.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    device = torch.device("cuda:0")
    policy = PI0Policy.from_pretrained(str(CHECKPOINT), strict=False).to(device).eval()
    model = policy.model
    layer_count = int(model.paligemma_with_expert.config.paligemma_config.text_config.num_hidden_layers)
    if layer_count != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} PaliGemma language layers, found {layer_count}")
    if policy.config.attention_implementation != "eager":
        raise RuntimeError("attention capture requires the checkpoint's eager Pi0 attention implementation")

    # Record averaged prompt-query -> 256 image-key attention for each layer.
    pg = model.paligemma_with_expert
    original_attention = pg.eager_attention_forward
    captured: list[np.ndarray] = []
    query_positions: list[int] = []

    def capture_attention(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        num_heads = self.config.paligemma_config.text_config.num_attention_heads
        kv_heads = self.config.paligemma_config.text_config.num_key_value_heads
        groups = num_heads // kv_heads
        seq_len = key_states.shape[1]
        keys = key_states[:, :, :, None, :].expand(batch_size, seq_len, kv_heads, groups, head_dim)
        keys = keys.reshape(batch_size, seq_len, num_heads, head_dim)
        q = query_states.float().transpose(1, 2)
        k = keys.float().transpose(1, 2)
        weights = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
        big_neg = -2.3819763e38
        allowed = attention_mask
        if allowed.ndim == 3:
            allowed = allowed[:, None, :, :]
        probs = torch.softmax(torch.where(allowed, weights, big_neg), dim=-1)
        qidx = torch.as_tensor(query_positions, device=probs.device, dtype=torch.long) + 256
        score = probs[0, :, qidx, :256].float().mean(dim=(0, 1)).detach().cpu().numpy()
        captured.append(score)
        # Match the underlying implementation's output exactly.
        values = value_states[:, :, :, None, :].expand(batch_size, seq_len, kv_heads, groups, head_dim)
        values = values.reshape(batch_size, seq_len, num_heads, head_dim)
        probs_for_output = probs.to(dtype=values.dtype)
        output = torch.matmul(probs_for_output, values.permute(0, 2, 1, 3))
        output = output.permute(0, 2, 1, 3).reshape(batch_size, -1, kv_heads * groups * head_dim)
        return output

    pg.eager_attention_forward = capture_attention.__get__(pg, type(pg))

    def get_attention(image: np.ndarray, instruction: str) -> tuple[np.ndarray, dict]:
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None].to(device) / 255.0
        # This matches the LeRobot conversion path: channel-first float pixels in [0, 1],
        # followed by the checkpoint's own resize_with_pad preprocessing.
        batch = {
            "observation.images.top": image_tensor,
            OBS_ROBOT: torch.zeros((1, 7), dtype=torch.float32, device=device),
            "task": [instruction],
        }
        normalized = policy.normalize_inputs(batch)
        images, img_masks = policy.prepare_images(normalized)
        lang_tokens, lang_masks = policy.prepare_language(batch)
        ids = lang_tokens[0].detach().cpu().tolist()
        prompt_ids = policy.language_tokenizer(instruction, add_special_tokens=False)["input_ids"]
        qstart = _find_subsequence(ids, list(prompt_ids))
        valid_count = int(lang_masks[0].sum().item())
        if qstart + len(prompt_ids) > valid_count:
            raise RuntimeError("prompt query tokens overlap padding")
        query_positions[:] = list(range(qstart, qstart + len(prompt_ids)))
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        captured.clear()
        with torch.inference_mode():
            model.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                fill_kv_cache=False,
            )
        if len(captured) != N_LAYERS:
            raise RuntimeError(f"captured {len(captured)} layers, expected {N_LAYERS}")
        rows = np.asarray(captured, dtype=np.float32)
        if rows.shape != (N_LAYERS, 256) or not np.isfinite(rows).all():
            raise RuntimeError(f"invalid attention shape/data: {rows.shape}")
        meta = {
            "query_token_ids": list(prompt_ids),
            "query_positions_in_language": list(query_positions),
            "visual_positions_in_prefix": list(range(256)),
            "per_layer_visual_attention_mass": rows.sum(axis=1).astype(float).tolist(),
            "image_preprocessing": "RGB uint8 -> CHW float32 /255 -> PI0Policy.prepare_images resize_with_pad(224,224)",
        }
        return rows, meta

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SCAN_CONFIG.json").write_text(json.dumps({
        "protocol_id": "PI0_INTACT_BRIDGE_PROMPT_ATTENTION_LAYER_SCAN_V1",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_type": "INTACT Pi0 BridgeV2 fine-tune; intended for WidowX Bridge tasks",
        "layers": N_LAYERS,
        "tasks": TASKS,
        "seeds": SEEDS,
        "prompt_variants": "original / frozen synonym / target-switched",
        "source_states": str(SOURCE),
        "selected_layer_cut": "to be written by analyzer after extraction",
        "gpus_visible_to_process": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_note": "Offline candidate scan only; does not establish closed-loop success benefit.",
    }, indent=2) + "\n")

    from research.semantic_token_cd.spatialvla_layer_selection_offline import get_controls

    processed = 0
    for task in TASKS:
        for seed in SEEDS:
            src = SOURCE / task / f"seed_{seed:03d}"
            meta = json.loads((src / "meta.json").read_text())
            arrays = np.load(src / "input.npz")
            image = np.asarray(arrays["image"])
            segmentation = np.asarray(arrays["seg_actor"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise RuntimeError(f"expected RGB uint8 HWC image, got {image.shape} {image.dtype}")
            instruction = meta["instruction"]
            target_switch, old_actor, new_actor = get_controls(task, instruction, meta)
            prompts = (instruction, SYNONYMS[task], target_switch)
            scores, capture_meta = [], []
            for prompt in prompts:
                score, cap = get_attention(image, prompt)
                scores.append(score)
                capture_meta.append(cap)
            old_mask = _patch_mask(segmentation, old_actor).astype(np.uint8)
            new_mask = _patch_mask(segmentation, new_actor).astype(np.uint8)
            if not old_mask.any() or not new_mask.any():
                raise RuntimeError(f"empty actor patch mask for {task} seed {seed}: {old_actor}/{new_actor}")

            out_dir = OUT / "states" / task / f"seed_{seed:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / "layers.npz",
                original=scores[0], synonym=scores[1], target_switch=scores[2],
                old_target_mask=old_mask, new_target_mask=new_mask,
            )
            result = {
                "protocol_id": "PI0_INTACT_BRIDGE_PROMPT_ATTENTION_LAYER_SCAN_V1",
                "task": task, "seed": seed, "checkpoint": str(CHECKPOINT),
                "n_language_layers": N_LAYERS,
                "instruction": instruction,
                "synonym_instruction": SYNONYMS[task],
                "target_switch_instruction": target_switch,
                "old_target_actor_id": int(old_actor), "new_target_actor_id": int(new_actor),
                "old_target_patch_count": int(old_mask.sum()),
                "new_target_patch_count": int(new_mask.sum()),
                "capture": capture_meta,
                "arrays_file": "layers.npz",
            }
            (out_dir / "meta.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"task": task, "seed": seed, "old_patches": int(old_mask.sum()),
                              "new_patches": int(new_mask.sum()), "done": True}), flush=True)
            processed += 1
            if args.limit and processed >= args.limit:
                print("DEBUG_LIMIT_REACHED", flush=True)
                return
    print("EXTRACTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
