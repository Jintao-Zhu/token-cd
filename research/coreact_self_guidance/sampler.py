from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from research.coreact_closed_loop.guidance import (
    GuidanceConfig,
    _prefix_cache,
    _full_velocity,
    replace_visual_tokens,
    select_visual_tokens,
    tensor_sha256,
)
from research.coreact_exploration.instrumentation import attention_ranking_scores, build_prefix_span_map


def _native_prefix(model, images, image_masks, lang_tokens, lang_masks, state):
    prefix, pad_masks, att_masks = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
    span_map = build_prefix_span_map(model, images, image_masks, lang_tokens, lang_masks, pad_masks, camera_ids=("camera1", "camera2"))[0]
    cache = _prefix_cache(model, prefix, pad_masks, att_masks)
    return prefix, pad_masks, att_masks, span_map, cache


def _velocity(model, pad_masks, cache, x_t, tau):
    timestep = torch.full((x_t.shape[0],), float(tau), dtype=torch.float32, device=x_t.device)
    return model.denoise_step(pad_masks, cache, x_t, timestep).to(dtype=torch.float32)


def apply_self_guidance_velocity(clean_velocity, negative_velocity, *, w, skipped, pure_negative, action_dim, trust_region_kappa):
    raw_delta = (clean_velocity - negative_velocity).clone()
    raw_delta[..., action_dim:] = 0
    clean_norm = torch.linalg.vector_norm(clean_velocity[..., :action_dim])
    raw_norm = torch.linalg.vector_norm(raw_delta[..., :action_dim])
    limit = trust_region_kappa * clean_norm
    clip_scale = torch.clamp(limit / (raw_norm + 1e-12), max=1.0)
    clipped_delta = raw_delta * clip_scale
    if skipped:
        guided_velocity = clean_velocity
        applied = torch.zeros_like(raw_delta)
    elif pure_negative:
        guided_velocity = negative_velocity
        applied = negative_velocity - clean_velocity
    else:
        applied = (w - 1.0) * clipped_delta
        guided_velocity = clean_velocity + applied
    return guided_velocity, raw_norm, applied, clip_scale


@torch.no_grad()
def reference_trajectory(model, pad_masks, cache, noise, *, num_steps: int = 10):
    dt = -1.0 / num_steps
    x_t = noise.clone()
    states, velocities, times = [], [], []
    for step in range(num_steps):
        tau = 1.0 + step * dt
        states.append(x_t.clone())
        velocity = _velocity(model, pad_masks, cache, x_t, tau)
        velocities.append(velocity)
        times.append(tau)
        x_t = x_t + dt * velocity
    return states, velocities, times, x_t


@torch.no_grad()
def sample_self_guided_actions(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    *,
    delta: float,
    w: float,
    pure_negative: bool = False,
    action_dim: int = 7,
    trust_region_kappa: float = 0.25,
    num_steps: int = 10,
) -> tuple[torch.Tensor, dict]:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("self-guidance requires frozen eval model")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 <= w <= 2.0:
        raise ValueError("w must be in [0, 2]")
    if abs(delta * num_steps - round(delta * num_steps)) > 1e-6:
        raise ValueError("delta must align to the locked flow grid")
    lag = int(round(delta * num_steps))
    prefix, pad_masks, att_masks, span_map, cache = _native_prefix(model, images, image_masks, lang_tokens, lang_masks, state)
    ref_states, ref_velocities, ref_times, reference_final = reference_trajectory(model, pad_masks, cache, noise, num_steps=num_steps)
    native_final = reference_final
    # w=1 is intentionally returned from the native trajectory so the identity gate is exact.
    if w == 1.0 and not pure_negative:
        return native_final, {
            "method": "self_guidance",
            "w": w,
            "delta": delta,
            "lag_steps": lag,
            "selected_indices": [],
            "changed_indices": [],
            "masked_token_count": 0,
            "prefix_sha256": tensor_sha256(prefix),
            "negative_prefix_sha256": tensor_sha256(prefix),
            "reference_final_sha256": tensor_sha256(reference_final),
            "noise_sha256": tensor_sha256(noise),
            "step_traces": [{"step": i, "skipped_step": False, "finite": True, "effective_delta_used": 0.0} for i in range(num_steps)],
            "all_output_finite": True,
            "protected_tokens_untouched": True,
        }

    dt = -1.0 / num_steps
    x_t = noise.clone()
    step_traces = []
    for step in range(num_steps):
        tau = ref_times[step]
        clean_velocity = _velocity(model, pad_masks, cache, x_t, tau)
        skipped = step < lag
        if skipped:
            negative_velocity = clean_velocity
        else:
            # Protocol time is progress s=1-tau. s-delta is the earlier/coarser
            # point, represented by the native reference at step-delta_steps.
            negative_velocity = ref_velocities[step - lag]
        guided_velocity, raw_norm, applied, clip_scale = apply_self_guidance_velocity(
            clean_velocity,
            negative_velocity,
            w=w,
            skipped=skipped,
            pure_negative=pure_negative,
            action_dim=action_dim,
            trust_region_kappa=trust_region_kappa,
        )
        if not torch.isfinite(guided_velocity).all():
            raise RuntimeError("nonfinite self-guided velocity")
        x_t = x_t + dt * guided_velocity
        step_traces.append(
            {
                "step": step,
                "flow_time_tau": tau,
                "progress_time": 1.0 - tau,
                "clean_minus_v_neg_l2_norm_pre_clip": float(raw_norm),
                "applied_delta_l2_norm_post_clip": float(torch.linalg.vector_norm(applied[..., :action_dim])),
                "clip_scale": float(clip_scale),
                "trust_region_clipping_active_bool": bool(float(clip_scale) < 1.0 - 1e-12),
                "skipped_step_due_to_boundary_bool": skipped,
                "effective_delta_used": 0.0 if skipped else delta,
                "pure_negative": pure_negative,
                "finite": True,
            }
        )
    return x_t, {
        "method": "self_guidance",
        "w": w,
        "delta": delta,
        "lag_steps": lag,
        "selected_indices": [],
        "changed_indices": [],
        "masked_token_count": 0,
        "prefix_sha256": tensor_sha256(prefix),
        "negative_prefix_sha256": tensor_sha256(prefix),
        "reference_final_sha256": tensor_sha256(reference_final),
        "noise_sha256": tensor_sha256(noise),
        "step_traces": step_traces,
        "all_output_finite": bool(torch.isfinite(x_t).all()),
        "protected_tokens_untouched": True,
        "reference_trajectory_final_sha256": tensor_sha256(native_final),
    }


@torch.no_grad()
def select_ref_top8_actions(model, images, image_masks, lang_tokens, lang_masks, state, noise, means, *, selection_seed: int, record_correction_vectors: bool = False):
    config = GuidanceConfig(selection="top", group_count=8, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10, direction="toward", record_correction_vectors=record_correction_vectors)
    from research.coreact_closed_loop.guidance import sample_coreact_actions

    return sample_coreact_actions(model, images, image_masks, lang_tokens, lang_masks, state, noise, means["visual_position_mean"], config=config, selection_seed=selection_seed)
