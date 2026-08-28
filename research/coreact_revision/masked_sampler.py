"""Frozen SmolVLA sampler that executes a selected-token masked prefix directly."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import torch

from research.coreact_closed_loop.guidance import (
    GuidanceConfig,
    _full_velocity,
    _prefix_cache,
    replace_visual_tokens,
    select_visual_tokens,
    tensor_sha256,
)
from research.coreact_exploration.instrumentation import (
    attention_ranking_scores,
    build_prefix_span_map,
)


@torch.no_grad()
def sample_masked_actions(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    visual_position_mean: torch.Tensor,
    *,
    camera_ids: Sequence[str] = ("camera1", "camera2"),
    config: GuidanceConfig,
    selection_seed: int,
) -> tuple[torch.Tensor, dict]:
    """Rank on the native prefix, then integrate only the selected-token masked branch."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("masked sampling requires a frozen model in eval mode")
    expected = (state.shape[0], model.config.chunk_size, model.config.max_action_dim)
    if noise.shape != expected:
        raise ValueError(f"unexpected noise shape {tuple(noise.shape)}, expected {expected}")
    if state.shape[0] != 1:
        raise ValueError("mask diagnostic requires batch size one")
    if config.num_steps != model.config.num_steps:
        raise ValueError("flow-step count must equal the checkpoint configuration")

    prefix, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks, state=state
    )
    span_map = build_prefix_span_map(
        model,
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        prefix_pad_masks,
        camera_ids=camera_ids,
    )[0]
    first_time = torch.ones((1,), dtype=torch.float32, device=state.device)
    _, traces = _full_velocity(
        model,
        prefix,
        prefix_pad_masks,
        prefix_att_masks,
        noise,
        first_time,
        record_attention=True,
    )
    scores = attention_ranking_scores(traces, prefix.shape[1])[
        "late_half_action_to_context_attention"
    ]
    selected = select_visual_tokens(
        span_map,
        scores,
        method=config.selection,
        count=config.group_count,
        random_seed=selection_seed,
    )
    masked_prefix = replace_visual_tokens(
        prefix, span_map, selected, visual_position_mean, camera_ids
    )
    changed = (prefix != masked_prefix).any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
    if sorted(changed) != sorted(selected):
        raise RuntimeError(f"masked prefix changed {changed}, expected {selected}")

    masked_cache = _prefix_cache(model, masked_prefix, prefix_pad_masks, prefix_att_masks)
    dt = -1.0 / config.num_steps
    x_t = noise.clone()
    step_traces = []
    for step in range(config.num_steps):
        flow_time = 1.0 + step * dt
        timestep = torch.full((1,), flow_time, dtype=torch.float32, device=state.device)
        velocity = model.denoise_step(prefix_pad_masks, masked_cache, x_t, timestep)
        if not bool(torch.isfinite(velocity).all()):
            raise RuntimeError("nonfinite masked velocity")
        x_t = x_t + dt * velocity
        step_traces.append(
            {
                "step": step,
                "flow_time": flow_time,
                "masked_velocity_norm": float(
                    torch.linalg.vector_norm(velocity[..., : config.action_dim])
                ),
                "virtual_velocity_norm": float(
                    torch.linalg.vector_norm(velocity[..., config.action_dim :])
                ),
                "finite": True,
            }
        )

    selected_tokens = [asdict(span_map[index]) for index in selected]
    return x_t, {
        "selection": config.selection,
        "selected_indices": selected,
        "changed_indices": changed,
        "selected_tokens": selected_tokens,
        "selected_scores": [float(scores[index]) for index in selected],
        "prefix_sha256": tensor_sha256(prefix),
        "masked_prefix_sha256": tensor_sha256(masked_prefix),
        "attention_layer_count": len(traces),
        "step_traces": step_traces,
        "fallback_count": 0,
        "all_output_finite": bool(torch.isfinite(x_t).all()),
        "protected_tokens_untouched": all(
            span_map[index].modality == "visual" and span_map[index].intervention_allowed
            for index in changed
        ),
    }
