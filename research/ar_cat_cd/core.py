"""CAT-CD Phase-0 forward primitives (base OpenVLA-7B + SIMPLER frozen states).

Uses the HF `model()` forward (the same version-stable path token_pcd_stage_a
validated on this exact checkpoint), plus projector_intervention masking:
  - clean action-vocab logits + action-query->visual attention scores
  - pixel (contrast-image) action-vocab logits -> r_PCD
  - leave-one-out causal attribution D_v over all 256 visual tokens
  - joint top-k masked branches (CAT / attention / random) at k in {4,8,16}
"""
from __future__ import annotations

from typing import Sequence

import cv2
import torch
from PIL import Image

from research.cw_lpcd.core import N_ACTION_TOKENS, N_VISUAL, ensure_empty_action_token
from research.ar_token_counterfactual.intervention import projector_intervention


def _preprocess(image) -> dict:
    """RGB numpy (any size) -> processor inputs (224x224 INTER_AREA, bf16), matching
    token_pcd_stage_a inputs_for exactly."""
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return {"pil": Image.fromarray(resized)}


def _teacher_pairs(base_ids: torch.Tensor, base_mask: torch.Tensor, clean_ids: torch.Tensor):
    teacher_ids = torch.cat([base_ids, clean_ids[:, :-1]], dim=1)
    teacher_mask = torch.cat(
        [base_mask, torch.ones_like(clean_ids[:, :-1], dtype=base_mask.dtype, device=base_mask.device)], dim=1
    )
    return teacher_ids, teacher_mask


def _action_query_positions(base_len: int) -> list[int]:
    """Visual tokens occupy multimodal positions 1..256; text follows. Returns the
    7 action query positions in the model() logits sequence."""
    return [N_VISUAL + base_len - 1 + offset for offset in range(N_ACTION_TOKENS)]


def attention_over_visual(attentions, base_len: int) -> torch.Tensor:
    """Mean action-query -> visual-key attention over late-half layers + heads + queries.

    attentions: tuple of 32 tensors [1, heads, seq, seq]. Returns [256].
    Matches teacher_forced_forward's late-half pooling (layers 16..31).
    """
    late = attentions[len(attentions) // 2:]
    q = _action_query_positions(base_len)
    stacked = torch.stack([a[:, :, q, 1:1 + N_VISUAL] for a in late])  # [16, 1, heads, 7, 256]
    return stacked.mean(dim=(0, 1, 2, 3)).detach().float().cpu()  # [256]


def forward_clean(model, processor, image, instruction: str, clean_ids: torch.Tensor,
                  record_attention: bool = True):
    """Clean branch: teacher-forced action logits (and optionally attention)."""
    inputs = processor(instruction, _preprocess(image)["pil"]).to(model.device, dtype=torch.bfloat16)
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    teacher_ids, teacher_mask = _teacher_pairs(base_ids, base_mask, clean_ids)
    output = model(
        input_ids=teacher_ids, attention_mask=teacher_mask,
        pixel_values=inputs["pixel_values"], use_cache=False,
        output_attentions=record_attention, return_dict=True,
    )
    q = _action_query_positions(base_ids.shape[1])
    logits_a = output.logits[0, q].detach().float().cpu()  # [7, vocab]
    attn = attention_over_visual(output.attentions, base_ids.shape[1]) if record_attention else None
    return logits_a, attn


def forward_masked(model, processor, image, instruction: str, clean_ids: torch.Tensor,
                   selected_indices: Sequence[int], mean: torch.Tensor):
    """Masked branch: replace selected visual-token projector rows with position mean."""
    inputs = processor(instruction, _preprocess(image)["pil"]).to(model.device, dtype=torch.bfloat16)
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    teacher_ids, teacher_mask = _teacher_pairs(base_ids, base_mask, clean_ids)
    with projector_intervention(model, list(selected_indices), mean) as trace:
        output = model(
            input_ids=teacher_ids, attention_mask=teacher_mask,
            pixel_values=inputs["pixel_values"], use_cache=False, return_dict=True,
        )
    q = _action_query_positions(base_ids.shape[1])
    logits_a = output.logits[0, q].detach().float().cpu()
    return logits_a, trace


def residual(clean_a: torch.Tensor, branch_a: torch.Tensor, token_slice: slice) -> torch.Tensor:
    """r = log_softmax(clean[action]) - log_softmax(branch[action]); [7, 256] (float64)."""
    lp_c = torch.log_softmax(clean_a[:, token_slice].double(), dim=-1)
    lp_b = torch.log_softmax(branch_a[:, token_slice].double(), dim=-1)
    return lp_c - lp_b


def residual_norm(clean_a: torch.Tensor, branch_a: torch.Tensor, token_slice: slice) -> float:
    return float(torch.linalg.vector_norm(residual(clean_a, branch_a, token_slice)))
