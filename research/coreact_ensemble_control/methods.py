from __future__ import annotations

from typing import Sequence

import torch

from research.coreact_closed_loop.guidance import tensor_sha256


@torch.no_grad()
def sample_dual_seed_average(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    first_noise: torch.Tensor,
    second_noise: torch.Tensor,
    *,
    action_dim: int = 7,
    trust_region_kappa: float = 0.25,
    matched_step: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Average two native chunks, optionally clipping the offset from the first."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("dual-seed averaging requires a frozen eval model")
    if first_noise.shape != second_noise.shape:
        raise ValueError("dual-seed noise shapes differ")
    first = model.sample_actions(images, image_masks, lang_tokens, lang_masks, state, noise=first_noise)
    second = model.sample_actions(images, image_masks, lang_tokens, lang_masks, state, noise=second_noise)
    raw_offset = 0.5 * (second - first)
    raw_offset[..., action_dim:] = 0
    raw_norm = torch.linalg.vector_norm(raw_offset[..., :action_dim])
    reference_norm = torch.linalg.vector_norm(first[..., :action_dim])
    limit = trust_region_kappa * reference_norm
    clip_scale = (
        torch.clamp(limit / (raw_norm + 1e-12), max=1.0)
        if matched_step
        else torch.ones((), device=raw_norm.device, dtype=raw_norm.dtype)
    )
    applied = raw_offset * clip_scale
    output = first + applied
    if not torch.isfinite(output).all():
        raise RuntimeError("nonfinite dual-seed averaged action chunk")
    trace = {
        "method": "dual_seed_chunk_average",
        "matched_step": matched_step,
        "masked_token_count": 0,
        "selected_indices": [],
        "changed_indices": [],
        "first_noise_sha256": tensor_sha256(first_noise),
        "second_noise_sha256": tensor_sha256(second_noise),
        "first_chunk_sha256": tensor_sha256(first),
        "second_chunk_sha256": tensor_sha256(second),
        "output_chunk_sha256": tensor_sha256(output),
        "clean_minus_masked_l2_norm_pre_clip": float(raw_norm),
        "applied_delta_l2_norm_post_clip": float(torch.linalg.vector_norm(applied[..., :action_dim])),
        "reference_chunk_l2_norm": float(reference_norm),
        "clip_scale": float(clip_scale),
        "trust_region_clipping_active": bool(float(clip_scale) < 1.0 - 1e-12),
        "all_output_finite": True,
    }
    return output, trace
