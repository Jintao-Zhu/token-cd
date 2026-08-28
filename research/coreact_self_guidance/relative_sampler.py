from __future__ import annotations

from typing import Sequence

import torch

from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_self_guidance.sampler import (
    _native_prefix,
    _velocity,
    apply_self_guidance_velocity,
    reference_trajectory,
)


def relative_reference_coordinates(step: int, alpha: float, num_steps: int) -> tuple[float, float, int, int, float]:
    progress = step / num_steps
    negative_progress = alpha * progress
    grid_position = negative_progress * num_steps
    lower = min(int(grid_position), num_steps - 1)
    upper = min(lower + 1, num_steps - 1)
    return progress, negative_progress, lower, upper, grid_position - lower


@torch.no_grad()
def sample_relative_self_guided_actions(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    *,
    alpha: float,
    w: float,
    pure_negative: bool = False,
    action_dim: int = 7,
    trust_region_kappa: float = 0.25,
    num_steps: int = 10,
) -> tuple[torch.Tensor, dict]:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("relative self-guidance requires frozen eval model")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0.0 <= w <= 2.0:
        raise ValueError("w must be in [0, 2]")
    prefix, pad_masks, _att_masks, _span_map, cache = _native_prefix(
        model, images, image_masks, lang_tokens, lang_masks, state
    )
    ref_states, _ref_velocities, _ref_times, reference_final = reference_trajectory(
        model, pad_masks, cache, noise, num_steps=num_steps
    )
    if w == 1.0 and not pure_negative:
        return reference_final, {
            "method": "relative_self_guidance",
            "alpha": alpha,
            "w": w,
            "selected_indices": [],
            "changed_indices": [],
            "masked_token_count": 0,
            "prefix_sha256": tensor_sha256(prefix),
            "negative_prefix_sha256": tensor_sha256(prefix),
            "reference_final_sha256": tensor_sha256(reference_final),
            "noise_sha256": tensor_sha256(noise),
            "step_traces": [
                {"step": i, "active_earlier_self_bool": i > 0, "finite": True}
                for i in range(num_steps)
            ],
            "all_output_finite": True,
            "protected_tokens_untouched": True,
        }

    dt = -1.0 / num_steps
    x_t = noise.clone()
    step_traces = []
    for step in range(num_steps):
        progress, negative_progress, lower, upper, fraction = relative_reference_coordinates(step, alpha, num_steps)
        if upper == lower:
            negative_state = ref_states[lower]
        else:
            negative_state = torch.lerp(ref_states[lower], ref_states[upper], fraction)
        negative_tau = 1.0 - negative_progress
        timestep = torch.full((noise.shape[0],), negative_tau, dtype=torch.float32, device=noise.device)
        clean_velocity = _velocity(model, pad_masks, cache, x_t, 1.0 - progress)
        negative_velocity = model.denoise_step(pad_masks, cache, negative_state, timestep).to(dtype=torch.float32)
        guided_velocity, raw_norm, applied, clip_scale = apply_self_guidance_velocity(
            clean_velocity,
            negative_velocity,
            w=w,
            skipped=False,
            pure_negative=pure_negative,
            action_dim=action_dim,
            trust_region_kappa=trust_region_kappa,
        )
        if not torch.isfinite(guided_velocity).all():
            raise RuntimeError("nonfinite relative self-guided velocity")
        x_t = x_t + dt * guided_velocity
        step_traces.append(
            {
                "step": step,
                "progress_time": progress,
                "negative_progress_time": negative_progress,
                "negative_flow_time_tau": negative_tau,
                "active_earlier_self_bool": step > 0,
                "clean_minus_v_neg_l2_norm_pre_clip": float(raw_norm),
                "applied_delta_l2_norm_post_clip": float(torch.linalg.vector_norm(applied[..., :action_dim])),
                "clip_scale": float(clip_scale),
                "trust_region_clipping_active_bool": bool(float(clip_scale) < 1.0 - 1e-12),
                "pure_negative": pure_negative,
                "finite": True,
            }
        )
    return x_t, {
        "method": "relative_self_guidance",
        "alpha": alpha,
        "w": w,
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
    }
