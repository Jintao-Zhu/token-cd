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


def noisier_timestep(tau: float, shift: float) -> float:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    if not 0.0 < shift < 1.0:
        raise ValueError("shift must be in (0, 1)")
    return min(1.0, tau + shift)


@torch.no_grad()
def treatment_free_timestep_geometry(
    model, images, image_masks, lang_tokens, lang_masks, state, noise, *, shift: float,
    action_dim: int = 7, num_steps: int = 10,
):
    """Evaluate clean/shift fields along the clean flow path before any action executes."""
    _prefix, pad_masks, _att_masks, _span_map, cache = _native_prefix(
        model, images, image_masks, lang_tokens, lang_masks, state
    )
    states, clean_velocities, times, clean_final = reference_trajectory(
        model, pad_masks, cache, noise, num_steps=num_steps
    )
    shifted = [
        _velocity(model, pad_masks, cache, x_t, noisier_timestep(tau, shift))
        for x_t, tau in zip(states, times)
    ]
    return {
        "x_tau": torch.stack([x[..., :action_dim].detach().float().cpu() for x in states]),
        "v_clean": torch.stack([x[..., :action_dim].detach().float().cpu() for x in clean_velocities]),
        "v_shift": torch.stack([x[..., :action_dim].detach().float().cpu() for x in shifted]),
        "d_raw": torch.stack([(a-b)[..., :action_dim].detach().float().cpu() for a,b in zip(clean_velocities,shifted)]),
        "tau": torch.tensor(times,dtype=torch.float32),
        "shifted_tau": torch.tensor([noisier_timestep(x,shift) for x in times],dtype=torch.float32),
        "clean_final": clean_final.detach().float().cpu(),
    }


@torch.no_grad()
def sample_timestep_shift_actions(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    *,
    shift: float,
    w: float,
    pure_negative: bool = False,
    action_dim: int = 7,
    trust_region_kappa: float = 0.25,
    num_steps: int = 10,
    record_correction_vectors: bool = False,
) -> tuple[torch.Tensor, dict]:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("timestep self-guidance requires frozen eval model")
    if not 0.0 <= w <= 2.0:
        raise ValueError("w must be in [0, 2]")
    prefix, pad_masks, _att_masks, _span_map, cache = _native_prefix(
        model, images, image_masks, lang_tokens, lang_masks, state
    )
    _states, _velocities, _times, reference_final = reference_trajectory(
        model, pad_masks, cache, noise, num_steps=num_steps
    )
    if w == 1.0 and not pure_negative:
        return reference_final, {
            "method": "timestep_shift_self_guidance",
            "shift": shift,
            "w": w,
            "selected_indices": [],
            "changed_indices": [],
            "masked_token_count": 0,
            "prefix_sha256": tensor_sha256(prefix),
            "negative_prefix_sha256": tensor_sha256(prefix),
            "reference_final_sha256": tensor_sha256(reference_final),
            "noise_sha256": tensor_sha256(noise),
            "step_traces": [
                {"step": step, "active_timestep_shift_bool": step > 0, "finite": True}
                for step in range(num_steps)
            ],
            "all_output_finite": True,
            "protected_tokens_untouched": True,
        }

    dt = -1.0 / num_steps
    x_t = noise.clone()
    step_traces = []
    for step in range(num_steps):
        tau = 1.0 + step * dt
        shifted_tau = noisier_timestep(tau, shift)
        clean_velocity = _velocity(model, pad_masks, cache, x_t, tau)
        shifted_velocity = _velocity(model, pad_masks, cache, x_t, shifted_tau)
        guided_velocity, raw_norm, applied, clip_scale = apply_self_guidance_velocity(
            clean_velocity,
            shifted_velocity,
            w=w,
            skipped=False,
            pure_negative=pure_negative,
            action_dim=action_dim,
            trust_region_kappa=trust_region_kappa,
        )
        raw_delta = (clean_velocity - shifted_velocity).clone()
        raw_delta[..., action_dim:] = 0
        clipped_direction = raw_delta * clip_scale
        if not torch.isfinite(guided_velocity).all():
            raise RuntimeError("nonfinite timestep self-guided velocity")
        x_t = x_t + dt * guided_velocity
        step_trace = {
                "step": step,
                "flow_time_tau": tau,
                "shifted_noisier_tau": shifted_tau,
                "active_timestep_shift_bool": shifted_tau > tau,
                "clean_minus_shift_l2_norm_pre_clip": float(raw_norm),
                "clipped_direction_l2_norm": float(torch.linalg.vector_norm(clipped_direction[..., :action_dim])),
                "actual_applied_delta_l2_norm": float(torch.linalg.vector_norm(applied[..., :action_dim])),
                "clip_scale": float(clip_scale),
                "trust_region_clipping_active_bool": bool(float(clip_scale) < 1.0 - 1e-12),
                "pure_negative": pure_negative,
                "finite": True,
            }
        if record_correction_vectors:
            step_trace["_correction_vector"] = raw_delta[..., :action_dim].detach().float().cpu()
            step_trace["_applied_correction_vector"] = applied[..., :action_dim].detach().float().cpu()
            # Preserve both fields needed for state-level geometry analysis. These are
            # diagnostics only; the returned action and integration path are unchanged.
            step_trace["_clean_velocity"] = clean_velocity[..., :action_dim].detach().float().cpu()
            step_trace["_shifted_velocity"] = shifted_velocity[..., :action_dim].detach().float().cpu()
            step_trace["_x_tau"] = x_t[..., :action_dim].detach().float().cpu()
        step_traces.append(step_trace)
    return x_t, {
        "method": "timestep_shift_self_guidance",
        "shift": shift,
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
