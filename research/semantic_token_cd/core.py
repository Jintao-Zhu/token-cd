"""SEMANTIC_TOKEN_CD Phase-0 forward primitives.

Extends the CAT-CD forward path (research.ar_cat_cd.core) with feature
extraction: projector-output visual tokens h_i [256, 4096] and the visual
self-attention graph A [256, 256] (late-half layers, mean over heads).
"""
from __future__ import annotations

import torch

from research.cw_lpcd.core import ensure_empty_action_token
from research.ar_token_counterfactual.intervention import projector_intervention
from research.ar_cat_cd.core import _preprocess, _teacher_pairs, N_VISUAL


def visual_attention_matrix(attentions) -> torch.Tensor:
    """Visual self-attention [256,256]: mean over late-half layers + heads of the
    visual->visual sub-block (positions 1..256)."""
    late = attentions[len(attentions) // 2:]
    mats = torch.stack([a[0, :, 1:1 + N_VISUAL, 1:1 + N_VISUAL] for a in late])  # [16, heads, 256, 256]
    return mats.mean(dim=(0, 1)).detach().float().cpu()  # [256, 256]


def extract_features(model, processor, image, instruction: str, clean_ids: torch.Tensor):
    """One clean forward -> (h_i [256,4096] projector tokens, A [256,256] attention graph)."""
    inputs = processor(instruction, _preprocess(image)["pil"]).to(model.device, dtype=torch.bfloat16)
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    teacher_ids, teacher_mask = _teacher_pairs(base_ids, base_mask, clean_ids)
    with projector_intervention(model) as trace:
        output = model(
            input_ids=teacher_ids, attention_mask=teacher_mask,
            pixel_values=inputs["pixel_values"], use_cache=False,
            output_attentions=True, return_dict=True,
        )
    if trace.before is None or trace.before.shape[1] != N_VISUAL:
        raise RuntimeError(f"Expected [1,256,D] projector output, got {None if trace.before is None else trace.before.shape}")
    h_i = trace.before[0].detach().float().cpu()          # [256, 4096]
    A = visual_attention_matrix(output.attentions)         # [256, 256]
    return h_i, A
