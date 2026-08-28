"""Frozen SmolVLA self-contrast sampler used by the closed-loop pilot."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from research.coreact_exploration.instrumentation import (
    PrefixToken,
    attention_ranking_scores,
    build_prefix_span_map,
    validate_intervention_group,
)


SelectionMethod = Literal["top", "bottom", "random"]
GuidanceDirection = Literal["away", "toward"]


@dataclass(frozen=True)
class GuidanceConfig:
    branch: str = "visual"
    selection: SelectionMethod = "top"
    group_count: int = 8
    guidance_scale: float = 0.5
    trust_region_kappa: float = 0.25
    action_dim: int = 7
    num_steps: int = 10
    direction: GuidanceDirection = "away"
    record_correction_vectors: bool = False
    flow_step_start: int = 0
    flow_step_end: int = 10
    action_start: int = 0
    action_end: int = 50
    placement_multiplier: float = 1.0


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _full_velocity(
    model,
    prefix_embeddings: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    *,
    record_attention: bool,
) -> tuple[torch.Tensor, list[dict]]:
    suffix_embeddings, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, timestep)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    attention_mask = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    traces: list[dict] = []
    owner = model.vlm_with_expert
    previous = getattr(owner, "attention_trace_callback", None)
    owner.attention_trace_callback = traces.append if record_attention else None
    try:
        (_, suffix_out), _ = owner.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, suffix_embeddings],
            use_cache=False,
            fill_kv_cache=False,
        )
    finally:
        owner.attention_trace_callback = previous
    suffix_out = suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out), traces


def _prefix_cache(
    model,
    prefix_embeddings: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
):
    attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, cache = model.vlm_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )
    return cache


def select_visual_tokens(
    span_map: Sequence[PrefixToken],
    score: torch.Tensor,
    *,
    method: SelectionMethod,
    count: int,
    random_seed: int,
) -> list[int]:
    candidates = [token.index for token in span_map if token.modality == "visual" and token.intervention_allowed]
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible visual tokens for requested count {count}")
    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
    if method == "random":
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        order = torch.randperm(len(candidates), generator=generator)[:count]
        selected = candidate_tensor[order]
    else:
        candidate_scores = score.detach().cpu()[candidate_tensor]
        descending = method == "top"
        order = torch.argsort(candidate_scores, descending=descending, stable=True)[:count]
        selected = candidate_tensor[order]
    output = selected.tolist()
    validate_intervention_group(span_map, output)
    return output


def replace_visual_tokens(
    prefix_embeddings: torch.Tensor,
    span_map: Sequence[PrefixToken],
    indices: Sequence[int],
    visual_position_mean: torch.Tensor,
    camera_ids: Sequence[str],
) -> torch.Tensor:
    validate_intervention_group(span_map, indices)
    camera_to_index = {camera: index for index, camera in enumerate(camera_ids)}
    output = prefix_embeddings.clone()
    for index in indices:
        token = span_map[index]
        if token.modality != "visual" or token.camera_id not in camera_to_index:
            raise ValueError(f"visual replacement received nonvisual token {asdict(token)}")
        if token.visual_token_index is None:
            raise ValueError(f"visual token lacks connector position: {asdict(token)}")
        replacement = visual_position_mean[camera_to_index[token.camera_id], token.visual_token_index]
        output[:, index] = replacement.to(device=output.device, dtype=output.dtype)
    return output


@torch.no_grad()
def sample_coreact_actions(
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
    config: GuidanceConfig = GuidanceConfig(),
    selection_seed: int,
) -> tuple[torch.Tensor, dict]:
    """Sample with one matched negative prefix while leaving the frozen model unchanged."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CoreAct sampling requires a frozen model in eval mode")
    if noise.shape != (state.shape[0], model.config.chunk_size, model.config.max_action_dim):
        raise ValueError(f"unexpected noise shape {tuple(noise.shape)}")
    if state.shape[0] != 1:
        raise ValueError("pilot sampler requires batch size one")
    if config.num_steps != model.config.num_steps:
        raise ValueError("guidance flow-step count must equal the checkpoint configuration")
    if not 0 <= config.guidance_scale:
        raise ValueError("guidance scale must be nonnegative")
    if config.direction not in ("away", "toward"):
        raise ValueError(f"unknown guidance direction {config.direction}")
    if not 0 < config.trust_region_kappa <= 1:
        raise ValueError("trust-region kappa must lie in (0, 1]")
    if not (0 <= config.flow_step_start <= config.flow_step_end <= config.num_steps):
        raise ValueError("invalid flow-step placement")
    if not (0 <= config.action_start <= config.action_end <= model.config.chunk_size):
        raise ValueError("invalid action placement")
    if config.placement_multiplier < 0:
        raise ValueError("placement multiplier must be nonnegative")

    prefix, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks, state=state
    )
    if config.branch == "acg":
        span_map, traces, selected, attention_top = [], [], [], []
        scores = torch.empty(prefix.shape[1], device=prefix.device)
        negative_prefix = prefix
    else:
        span_map = build_prefix_span_map(
            model, images, image_masks, lang_tokens, lang_masks, prefix_pad_masks,
            camera_ids=camera_ids,
        )[0]
        first_time = torch.ones((1,), dtype=torch.float32, device=state.device)
        _, traces = _full_velocity(
            model, prefix, prefix_pad_masks, prefix_att_masks, noise, first_time,
            record_attention=True,
        )
        scores = attention_ranking_scores(traces, prefix.shape[1])["late_half_action_to_context_attention"]
        selected = select_visual_tokens(
            span_map, scores, method=config.selection, count=config.group_count,
            random_seed=selection_seed,
        )
        attention_top = select_visual_tokens(
            span_map, scores, method="top", count=config.group_count,
            random_seed=selection_seed,
        )
        negative_prefix = replace_visual_tokens(
            prefix, span_map, selected, visual_position_mean, camera_ids
        )
    changed = (prefix != negative_prefix).any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
    if config.branch != "acg" and sorted(changed) != sorted(selected):
        raise RuntimeError(f"negative prefix changed {changed}, expected {selected}")

    positive_cache = _prefix_cache(model, prefix, prefix_pad_masks, prefix_att_masks)
    negative_cache = _prefix_cache(model, negative_prefix, prefix_pad_masks, prefix_att_masks)
    dt = -1.0 / config.num_steps
    x_t = noise.clone()
    step_traces = []
    fallback_count = 0
    for step in range(config.num_steps):
        flow_time = 1.0 + step * dt
        timestep = torch.full((1,), flow_time, dtype=torch.float32, device=state.device)
        positive_velocity = model.denoise_step(prefix_pad_masks, positive_cache, x_t, timestep)
        negative_velocity = model.denoise_step(
            prefix_pad_masks,
            negative_cache,
            x_t,
            timestep,
            incoherent_action_attention=(config.branch == "acg"),
        )
        raw_guidance = positive_velocity - negative_velocity
        raw_guidance[..., config.action_dim :] = 0
        positive_norm = torch.linalg.vector_norm(positive_velocity[..., : config.action_dim])
        raw_norm = torch.linalg.vector_norm(raw_guidance[..., : config.action_dim])
        clip_limit = config.trust_region_kappa * positive_norm
        clip_scale = torch.clamp(clip_limit / (raw_norm + 1e-12), max=1.0)
        clipped = raw_guidance * clip_scale
        direction_sign = 1.0 if config.direction == "away" else -1.0
        placement = torch.zeros_like(clipped)
        if config.flow_step_start <= step < config.flow_step_end:
            placement[:, config.action_start:config.action_end, : config.action_dim] = 1.0
        signed_guidance = (
            direction_sign * config.guidance_scale * config.placement_multiplier * clipped * placement
        )
        guided_velocity = positive_velocity + signed_guidance
        finite = bool(torch.isfinite(guided_velocity).all())
        if not finite:
            guided_velocity = positive_velocity
            fallback_count += 1
        x_t = x_t + dt * guided_velocity
        step_trace = {
                "step": step,
                "flow_time": flow_time,
                "positive_velocity_norm": float(positive_norm),
                "negative_delta_norm": float(raw_norm),
                "clip_scale": float(clip_scale),
                "applied_guidance_norm": float(
                    torch.linalg.vector_norm(signed_guidance[..., : config.action_dim])
                ),
                "raw_guidance_sum": float(raw_guidance[..., : config.action_dim].sum()),
                "signed_guidance_sum": float(signed_guidance[..., : config.action_dim].sum()),
                "positive_velocity_sha256": tensor_sha256(positive_velocity),
                "raw_guidance_sha256": tensor_sha256(raw_guidance),
                "virtual_guidance_norm": float(
                    torch.linalg.vector_norm(signed_guidance[..., config.action_dim :])
                ),
                "finite": finite,
            }
        if config.record_correction_vectors:
            step_trace["_correction_vector"] = raw_guidance[..., : config.action_dim].detach().float().cpu()
            step_trace["_applied_correction_vector"] = signed_guidance[..., : config.action_dim].detach().float().cpu()
        step_traces.append(step_trace)

    selected_tokens = [asdict(span_map[index]) for index in selected]
    trace = {
        "selection": config.selection,
        "guidance_direction": config.direction,
        "guidance_branch": config.branch,
        "flow_step_range": [config.flow_step_start, config.flow_step_end],
        "action_range": [config.action_start, config.action_end],
        "placement_multiplier": config.placement_multiplier,
        "selected_indices": selected,
        "changed_indices": changed,
        "selected_tokens": selected_tokens,
        "selected_scores": [float(scores[index]) for index in selected],
        "attention_top_indices": attention_top,
        "overlap_with_attention_top": len(set(selected) & set(attention_top)),
        "prefix_sha256": tensor_sha256(prefix),
        "negative_prefix_sha256": tensor_sha256(negative_prefix),
        "attention_layer_count": len(traces),
        "step_traces": step_traces,
        "fallback_count": fallback_count,
        "all_output_finite": bool(torch.isfinite(x_t).all()),
        "protected_tokens_untouched": all(
            span_map[index].modality == "visual" and span_map[index].intervention_allowed
            for index in changed
        ),
    }
    return x_t, trace
