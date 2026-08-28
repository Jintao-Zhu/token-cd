"""CW-LPCD core: layer-aware OpenVLA forward with transient causal-route knockout.

Causal-Window Latent Policy Contrastive Decoding (CW-LPCD) — the negative branch
blocks the action-query -> object-visual-token attention edges ONLY within a
contiguous decoder layer window W, then restores normal attention.

This module replicates the proven SID layer-loop infrastructure
(analysis/ar_sid_layer_loop_sentinel.py) but (a) supports a layer *window* rather
than "from layer L onward", and (b) blocks only the (action-query, object-key)
edges rather than all queries to a key set.

Checkpoint: SIMPLER OpenVLA-7B base (`source/PCD/pretrained/openvla-7b`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image

PCD_ROOT = Path("/data/docker/dev_zjt/data/code/official-reproductions/pcd_openvla_simpler_box_31b027e")
CHECKPOINT = PCD_ROOT / "source/PCD/pretrained/openvla-7b"

TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_open_drawer",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "google_robot_place_apple_in_closed_top_drawer",
)
N_ACTION_TOKENS = 7
N_CONTINUOUS_DIMS = 6
N_VISUAL = 256
PATCH_GRID = 16
EMPTY_ACTION_TOKEN = 29871
ALPHA_PCD = 0.8

# STEP 8 pre-registered window enumeration: 7 centers x 3 sizes = 21 windows.
WINDOW_CENTERS = (3, 7, 11, 15, 19, 23, 27)
WINDOW_SIZES = (3, 5, 7)


def enumerate_windows() -> list[dict]:
    """Return the 21 pre-registered (center, size, start, end) windows, 0-indexed
    inclusive layer range [start, end] within the 32 decoder layers."""
    windows: list[dict] = []
    for center in WINDOW_CENTERS:
        for size in WINDOW_SIZES:
            half = (size - 1) // 2
            start, end = center - half, center + half
            assert 0 <= start <= end <= 31, (center, size, start, end)
            windows.append({"center": center, "size": size, "start": start, "end": end})
    assert len(windows) == 21, len(windows)
    return windows


def visual_key_positions(object_ids: Sequence[int]) -> list[int]:
    """Patch indices (0..255) -> multimodal sequence positions (visual tokens occupy
    positions 1..256, so patch p is at position 1+p)."""
    return [1 + int(p) for p in object_ids]


def load_model(checkpoint: Path = CHECKPOINT):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(checkpoint), attn_implementation="eager", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def inputs_for(processor, model, image: np.ndarray, instruction: str) -> dict[str, torch.Tensor]:
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return processor(instruction, Image.fromarray(resized)).to(model.device, dtype=torch.bfloat16)


def build_multimodal(model, input_ids, attention_mask, pixel_values):
    """Replicate Prismatic forward's multimodal embedding build exactly."""
    patch_features = model.vision_backbone(pixel_values)
    projected = model.projector(patch_features)
    input_embeds = model.get_input_embeddings()(input_ids)
    n_visual = projected.shape[1]
    mm_embeds = torch.cat([input_embeds[:, :1], projected, input_embeds[:, 1:]], dim=1)
    mm_mask = torch.cat(
        [attention_mask[:, :1],
         torch.full((1, n_visual), True, device=attention_mask.device, dtype=attention_mask.dtype),
         attention_mask[:, 1:]], dim=1)
    return mm_embeds, mm_mask, n_visual


def ensure_empty_action_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    if torch.all(input_ids[:, -1] == EMPTY_ACTION_TOKEN):
        return input_ids, attention_mask
    empty = torch.full((input_ids.shape[0], 1), EMPTY_ACTION_TOKEN, dtype=input_ids.dtype, device=input_ids.device)
    visible = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([input_ids, empty], dim=1), torch.cat([attention_mask, visible], dim=1)


@torch.inference_mode()
def custom_layer_loop(model, inputs_embeds, attention_mask_2d, key_mask=None,
                      key_mask_window=None, output_attentions=False):
    """Layer-by-layer forward replicating LlamaModel.forward (no cache) exactly.

    key_mask: 4D additive mask [1, 1, seq, seq]; 0 = allowed, min_dtype = blocked.
    key_mask_window: (start, end) inclusive 0-indexed layer range; None = all layers.
    """
    lm = model.language_model
    decoder = lm.model
    seq_len = inputs_embeds.shape[1]
    dtype = inputs_embeds.dtype
    cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    # transformers 4.44.2 signature: (attention_mask, input_tensor, cache_position, past_key_values, output_attentions)
    causal_mask = decoder._update_causal_mask(attention_mask_2d, inputs_embeds, cache_position, None, output_attentions)
    position_embeddings = decoder.rotary_emb(inputs_embeds, position_ids)

    hidden_states = inputs_embeds
    all_attns = () if output_attentions else None
    for idx, layer in enumerate(decoder.layers):
        layer_mask = causal_mask
        if key_mask is not None:
            if key_mask_window is None:
                layer_mask = causal_mask + key_mask
            else:
                w_start, w_end = key_mask_window
                if w_start <= idx <= w_end:
                    layer_mask = causal_mask + key_mask
        layer_outputs = layer(
            hidden_states,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_attns += (layer_outputs[1],)

    hidden_states = decoder.norm(hidden_states)
    logits = lm.lm_head(hidden_states)
    return logits, all_attns


def make_cw_mask(seq_len: int, query_positions: Sequence[int], key_positions: Sequence[int],
                 dtype, device) -> torch.Tensor:
    """4D additive mask blocking ONLY (query in query_positions) -> (key in key_positions)."""
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    if query_positions and key_positions:
        q = torch.as_tensor(list(query_positions), device=device, dtype=torch.long)
        k = torch.as_tensor(list(key_positions), device=device, dtype=torch.long)
        mask[0, 0, q[:, None], k[None, :]] = torch.finfo(dtype).min
    return mask


def action_token_slice(model) -> slice:
    return slice(int(model.vocab_size) - 256, int(model.vocab_size))


@torch.inference_mode()
def generate_clean_ids(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids, attention_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    generated = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=inputs["pixel_values"],
        max_new_tokens=N_ACTION_TOKENS, min_new_tokens=None, eos_token_id=None, do_sample=False)
    return generated[:, -N_ACTION_TOKENS:]


@torch.inference_mode()
def teacher_forced_logits(model, inputs: dict[str, torch.Tensor], clean_ids: torch.Tensor,
                          cw_query_positions: Sequence[int] = (),
                          cw_key_positions: Sequence[int] = (),
                          cw_window: tuple[int, int] | None = None) -> torch.Tensor:
    """Teacher-forced logits at the 7 action query positions, using the shared clean
    prefix. Applies the CW-LPCD transient window knockout when object keys + window are
    provided. Returns [7, vocab]."""
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    teacher_ids = torch.cat([base_ids, clean_ids[:, :-1]], dim=1)
    teacher_mask = torch.cat([base_mask, torch.ones_like(clean_ids[:, :-1], dtype=base_mask.dtype)], dim=1)
    mm_embeds, mm_mask, n_visual = build_multimodal(model, teacher_ids, teacher_mask, inputs["pixel_values"])

    key_mask = None
    if cw_query_positions and cw_key_positions and cw_window is not None:
        seq_len = mm_embeds.shape[1]
        key_mask = make_cw_mask(seq_len, cw_query_positions, cw_key_positions, torch.bfloat16, mm_embeds.device)

    logits, _ = custom_layer_loop(model, mm_embeds, mm_mask, key_mask=key_mask, key_mask_window=cw_window)
    query_indices = [n_visual + base_ids.shape[1] - 1 + offset for offset in range(N_ACTION_TOKENS)]
    return logits[0, query_indices].detach().float().cpu()  # [7, vocab]


@torch.inference_mode()
def teacher_forced_logits_persistent(model, inputs: dict[str, torch.Tensor], clean_ids: torch.Tensor,
                                     object_ids: Sequence[int], replacement_mean: torch.Tensor):
    """Old Persistent Token-PCD (STEP 6): replace the object visual-token projector rows
    with the position-conditioned mean for ALL 32 layers (forward hook), then teacher-force
    with the shared clean prefix. Same decoder path as the CW branches (custom_layer_loop),
    so the only difference vs CW-LPCD is the intervention *mechanism* (persistent row
    replacement vs transient attention knockout). Returns (logits [7, vocab], trace)."""
    from research.ar_token_counterfactual.intervention import projector_intervention

    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    teacher_ids = torch.cat([base_ids, clean_ids[:, :-1]], dim=1)
    teacher_mask = torch.cat([base_mask, torch.ones_like(clean_ids[:, :-1], dtype=base_mask.dtype)], dim=1)
    with projector_intervention(model, list(object_ids), replacement_mean) as trace:
        mm_embeds, mm_mask, n_visual = build_multimodal(model, teacher_ids, teacher_mask, inputs["pixel_values"])
    logits, _ = custom_layer_loop(model, mm_embeds, mm_mask)
    query_indices = [n_visual + base_ids.shape[1] - 1 + offset for offset in range(N_ACTION_TOKENS)]
    return logits[0, query_indices].detach().float().cpu(), trace


def log_softmax_residual(clean: torch.Tensor, branch: torch.Tensor, token_slice: slice) -> torch.Tensor:
    """r = log_softmax(z_clean[action_vocab]) - log_softmax(z_branch[action_vocab]); [7, 256]."""
    lp_clean = torch.log_softmax(clean[:, token_slice].float(), dim=-1)
    lp_branch = torch.log_softmax(branch[:, token_slice].float(), dim=-1)
    return lp_clean - lp_branch


def mask_patch_overlaps(mask: np.ndarray, size: int = 224) -> np.ndarray:
    """Project a 2D mask (values in [0,1]) onto the 16x16 patch grid -> 256 overlaps."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    if size % PATCH_GRID:
        raise ValueError(f"Image size {size} is not divisible by {PATCH_GRID}")
    patch = size // PATCH_GRID
    return resized.reshape(PATCH_GRID, patch, PATCH_GRID, patch).mean(axis=(1, 3)).reshape(-1)


def token_ids_from_mask(mask: np.ndarray, threshold: float) -> tuple[list[int], np.ndarray]:
    overlaps = mask_patch_overlaps(mask)
    return np.flatnonzero(overlaps >= threshold).astype(int).tolist(), overlaps


def matched_random_ids(object_ids: Sequence[int], seed: int, token_count: int = N_VISUAL) -> list[int]:
    """Matched-random control: same number of visual tokens, sampled outside the object set."""
    object_set = set(int(index) for index in object_ids)
    candidates = np.asarray([index for index in range(token_count) if index not in object_set])
    if len(object_ids) > len(candidates):
        raise ValueError("Not enough non-object tokens for matched random control")
    return sorted(np.random.default_rng(seed).choice(candidates, size=len(object_ids), replace=False).astype(int).tolist())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left.flatten().double(), right.flatten().double()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denom) if denom > 0 else float("nan")


def decode_action_ids(model, action_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    token_ids = action_ids.detach().cpu().numpy().reshape(-1)
    discretized = np.clip(model.vocab_size - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
