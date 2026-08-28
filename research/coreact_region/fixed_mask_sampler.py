"""Teacher-free first-chunk sampler for externally selected visual token groups."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import torch

from research.coreact_closed_loop.guidance import (
    GuidanceConfig,
    _full_velocity,
    _prefix_cache,
    replace_visual_tokens,
    tensor_sha256,
)
from research.coreact_exploration.instrumentation import (
    attention_ranking_scores,
    build_prefix_span_map,
    validate_intervention_group,
)


@torch.no_grad()
def prepare_ranked_prefix(
    model, images, image_masks, lang_tokens, lang_masks, state, noise,
    *, camera_ids: Sequence[str] = ("camera1", "camera2"),
) -> dict:
    prefix, pad_masks, att_masks = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
    span_map = build_prefix_span_map(
        model, images, image_masks, lang_tokens, lang_masks, pad_masks, camera_ids=camera_ids
    )[0]
    timestep = torch.ones((1,), dtype=torch.float32, device=state.device)
    _, traces = _full_velocity(
        model, prefix, pad_masks, att_masks, noise, timestep, record_attention=True
    )
    scores = attention_ranking_scores(traces, prefix.shape[1])[
        "late_half_action_to_context_attention"
    ]
    return {
        "prefix": prefix, "prefix_pad_masks": pad_masks, "prefix_att_masks": att_masks,
        "span_map": span_map, "scores": scores, "attention_layer_count": len(traces),
    }


@torch.no_grad()
def sample_fixed_mask_actions(
    model,
    ranked: dict,
    noise: torch.Tensor,
    visual_position_mean: torch.Tensor,
    selected_indices: Sequence[int],
    *,
    camera_ids: Sequence[str] = ("camera1", "camera2"),
    config: GuidanceConfig = GuidanceConfig(),
    replacement_type: str = "position_mean",
) -> tuple[torch.Tensor, dict]:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("fixed-mask sampling requires a frozen model in eval mode")
    validate_intervention_group(ranked["span_map"], selected_indices)
    if replacement_type == "position_mean":
        masked = replace_visual_tokens(
            ranked["prefix"], ranked["span_map"], selected_indices, visual_position_mean, camera_ids
        )
    elif replacement_type == "zero":
        masked = ranked["prefix"].clone()
        masked[:, list(selected_indices)] = 0
    else:
        raise ValueError(f"unsupported replacement type: {replacement_type}")
    changed = (ranked["prefix"] != masked).any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
    if sorted(changed) != sorted(selected_indices):
        raise RuntimeError(f"masked prefix changed {changed}, expected {list(selected_indices)}")
    cache = _prefix_cache(model, masked, ranked["prefix_pad_masks"], ranked["prefix_att_masks"])
    dt = -1.0 / config.num_steps
    x_t = noise.clone()
    for step in range(config.num_steps):
        timestep = torch.full((1,), 1.0 + step * dt, dtype=torch.float32, device=noise.device)
        velocity = model.denoise_step(ranked["prefix_pad_masks"], cache, x_t, timestep)
        if not bool(torch.isfinite(velocity).all()):
            raise RuntimeError("nonfinite fixed-mask velocity")
        x_t = x_t + dt * velocity
    return x_t, {
        "selected_indices": list(selected_indices), "changed_indices": changed,
        "selected_tokens": [asdict(ranked["span_map"][index]) for index in selected_indices],
        "selected_scores": [float(ranked["scores"][index]) for index in selected_indices],
        "prefix_sha256": tensor_sha256(ranked["prefix"]),
        "masked_prefix_sha256": tensor_sha256(masked),
        "replacement_type": replacement_type,
        "attention_layer_count": ranked["attention_layer_count"],
        "all_output_finite": bool(torch.isfinite(x_t).all()),
    }
