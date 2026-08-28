from __future__ import annotations

from typing import Sequence

import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from research.coreact_closed_loop.guidance import _prefix_cache, tensor_sha256
from research.coreact_trained_weak.sampler import applied_correction


ARMS = ("strong", "low_w1", "full_w1", "high_w1")
ACTIVE_STEPS = {
    "strong": frozenset(),
    "low_w1": frozenset((8, 9)),
    "full_w1": frozenset(range(10)),
    "high_w1": frozenset((0, 1)),
}
LAMBDA = 0.5
TRUST_REGION_KAPPA = 0.25


def _w1_velocity(model, prefix_pad_masks, cache, x_t, timestep, scales):
    suffix, suffix_pad, suffix_att = model.embed_suffix(x_t, timestep)
    suffix_len = suffix_pad.shape[1]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_2d = prefix_pad_masks[:, None, :].expand(x_t.shape[0], suffix_len, prefix_len)
    suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
    attention_mask = torch.cat([prefix_2d, suffix_2d], dim=2)
    offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = offsets + torch.cumsum(suffix_pad, dim=1) - 1
    outputs, _ = model.vlm_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        inputs_embeds=[None, suffix],
        use_cache=model.config.use_cache,
        fill_kv_cache=False,
        expert_residual_scales=scales,
    )
    suffix_out = outputs[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


@torch.no_grad()
def sample_w1_slg_actions(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    *,
    arm: str,
    lambda_value: float = LAMBDA,
):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if lambda_value not in (-0.1, -0.05, 0.0, 0.01, 0.05, 0.1, 0.25, 0.5):
        raise ValueError("dose must be one of 0,.05,.1,.25,.5")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("W1-SLG sampler requires a frozen eval model")
    if model.config.num_steps != 10 or model.vlm_with_expert.num_expert_layers != 16:
        raise RuntimeError("frozen 10-step/16-layer contract violated")
    expected = (state.shape[0], model.config.chunk_size, model.config.max_action_dim)
    if tuple(noise.shape) != expected:
        raise ValueError(f"noise shape {tuple(noise.shape)} != {expected}")

    prefix, pads, atts = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
    cache = _prefix_cache(model, prefix, pads, atts)
    scales = torch.ones(
        state.shape[0], 16, device=state.device, dtype=prefix.dtype
    )
    scales[:, -1] = 0
    x_t = noise.clone()
    x_reference = noise.clone() if arm != "strong" else x_t
    dt = -0.1
    traces = []
    for step in range(10):
        tau = 1.0 + step * dt
        timestep = torch.full((state.shape[0],), tau, device=state.device, dtype=torch.float32)
        strong = model.denoise_step(pads, cache, x_t, timestep).to(dtype=torch.float32)
        active = step in ACTIVE_STEPS[arm]
        if active:
            weak = _w1_velocity(model, pads, cache, x_t, timestep, scales)
            correction, clip_scale = applied_correction(strong, weak, lambda_value)
            used = strong + correction
            weak_hash = tensor_sha256(weak)
        else:
            weak = None
            correction = torch.zeros_like(strong)
            clip_scale = torch.ones((), device=strong.device)
            used = strong
            weak_hash = None
        if not bool(torch.isfinite(used).all()):
            raise RuntimeError(f"nonfinite velocity at flow step {step}")
        x_t = x_t + dt * used
        if arm != "strong":
            reference_velocity = model.denoise_step(pads, cache, x_reference, timestep).to(dtype=torch.float32)
            x_reference = x_reference + dt * reference_velocity
        traces.append({
            "step": step,
            "flow_time_tau": tau,
            "active": active,
            "strong_sha256": tensor_sha256(strong),
            "weak_sha256": weak_hash,
            "strong_norm": float(torch.linalg.vector_norm(strong)),
            "direction_norm": float(torch.linalg.vector_norm(strong - weak)) if active else 0.0,
            "applied_correction_norm": float(torch.linalg.vector_norm(correction)),
            "clip_scale": float(clip_scale) if active else None,
            "native_strong_action_sha256": tensor_sha256(
                (x_t + dt * strong)[..., :7]
            ),
        })
    return x_t, {
        "method": "shared_parameter_W1_skip_last_expert_block",
        "arm": arm,
        "active_steps": sorted(ACTIVE_STEPS[arm]),
        "lambda": lambda_value,
        "trust_region_kappa": TRUST_REGION_KAPPA,
        "scale_contract": [1.0] * 15 + [0.0],
        "prefix_sha256": tensor_sha256(prefix),
        "noise_sha256": tensor_sha256(noise),
        "initial_strong_velocity_sha256": traces[0]["strong_sha256"],
        "step_traces": traces,
        "all_output_finite": bool(torch.isfinite(x_t).all()),
        "reference_action_chunk": x_reference[..., :7].detach().cpu() if arm != "strong" else x_t[..., :7].detach().cpu(),
    }
