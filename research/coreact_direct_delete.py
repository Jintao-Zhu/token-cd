from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import torch

from research.coreact_closed_loop.guidance import GuidanceConfig, _full_velocity, _prefix_cache, select_visual_tokens, tensor_sha256
from research.coreact_exploration.instrumentation import attention_ranking_scores, build_prefix_span_map


def remove_prefix_tokens(prefix: torch.Tensor, pad_masks: torch.Tensor, att_masks: torch.Tensor, indices: Sequence[int]):
    selected = tuple(sorted(set(int(i) for i in indices)))
    if len(selected) != len(indices):
        raise ValueError("selected indices must be unique")
    keep = torch.ones(prefix.shape[1], dtype=torch.bool, device=prefix.device)
    keep[list(selected)] = False
    return prefix[:, keep], pad_masks[:, keep], att_masks[:, keep]


@torch.no_grad()
def sample_deleted_actions(model, images, image_masks, lang_tokens, lang_masks, state, noise, *, config: GuidanceConfig, selection_seed: int, mode: str):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("direct-delete sampler requires frozen eval model")
    prefix, prefix_pad, prefix_att = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
    span = build_prefix_span_map(model, images, image_masks, lang_tokens, lang_masks, prefix_pad, camera_ids=("camera1", "camera2"))[0]
    _, traces = _full_velocity(model, prefix, prefix_pad, prefix_att, noise, torch.ones((1,), device=state.device), record_attention=True)
    score = attention_ranking_scores(traces, prefix.shape[1])["late_half_action_to_context_attention"]
    selected = select_visual_tokens(span, score, method="top", count=config.group_count, random_seed=selection_seed)
    deleted_prefix, deleted_pad, deleted_att = remove_prefix_tokens(prefix, prefix_pad, prefix_att, selected)
    if deleted_prefix.shape[1] != prefix.shape[1] - config.group_count:
        raise RuntimeError("direct deletion did not reduce prefix length by configured group count")
    clean_cache = _prefix_cache(model, prefix, prefix_pad, prefix_att)
    deleted_cache = _prefix_cache(model, deleted_prefix, deleted_pad, deleted_att)
    dt = -1.0 / config.num_steps
    x_t = noise.clone(); step_traces = []
    for step in range(config.num_steps):
        t = torch.full((1,), 1.0 + step * dt, dtype=torch.float32, device=state.device)
        clean_v = model.denoise_step(prefix_pad, clean_cache, x_t, t)
        deleted_v = model.denoise_step(deleted_pad, deleted_cache, x_t, t)
        if mode == "mask_only":
            guided = deleted_v; signed = clean_v - deleted_v
        else:
            raw = clean_v - deleted_v
            raw[..., config.action_dim:] = 0
            limit = config.trust_region_kappa * torch.linalg.vector_norm(clean_v[..., :config.action_dim])
            norm = torch.linalg.vector_norm(raw[..., :config.action_dim])
            clipped = raw * torch.clamp(limit / (norm + 1e-12), max=1.0)
            signed = (1.0 if mode == "away" else -1.0) * config.guidance_scale * clipped
            guided = clean_v + signed
        if not torch.isfinite(guided).all():
            raise RuntimeError("nonfinite direct-delete velocity")
        x_t = x_t + dt * guided
        step_traces.append({"step": step, "flow_time": float(t.item()), "clean_norm": float(torch.linalg.vector_norm(clean_v[..., :config.action_dim])), "deleted_norm": float(torch.linalg.vector_norm(deleted_v[..., :config.action_dim])), "guidance_norm": float(torch.linalg.vector_norm(signed[..., :config.action_dim]))})
    return x_t, {"selected_indices": selected, "changed_indices": selected, "native_prefix_sha256": tensor_sha256(prefix), "deleted_prefix_sha256": tensor_sha256(deleted_prefix), "native_prefix_length": prefix.shape[1], "deleted_prefix_length": deleted_prefix.shape[1], "step_traces": step_traces, "all_output_finite": bool(torch.isfinite(x_t).all()), "mode": mode}
