from __future__ import annotations

from typing import Sequence

import torch

from research.coreact_closed_loop.guidance import _prefix_cache, tensor_sha256


ARMS = ("V", "W", "M", "G")
TRUST_REGION_KAPPA = 0.25


def _prefix(model, images, image_masks, lang_tokens, lang_masks, state):
    embeddings, pads, atts = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks, state=state
    )
    return embeddings, pads, _prefix_cache(model, embeddings, pads, atts)


def _velocity(model, pads, cache, x_t, tau):
    timestep = torch.full((x_t.shape[0],), tau, device=x_t.device, dtype=torch.float32)
    return model.denoise_step(pads, cache, x_t, timestep).to(dtype=torch.float32)


def applied_correction(
    strong: torch.Tensor,
    weak: torch.Tensor,
    lambda_value: float,
    action_dim: int | None = 7,
):
    raw = lambda_value * (strong - weak)
    strong_physical = strong if action_dim is None else strong[..., :action_dim]
    raw_physical = raw if action_dim is None else raw[..., :action_dim]
    limit = TRUST_REGION_KAPPA * torch.linalg.vector_norm(strong_physical)
    scale = torch.clamp(limit / (torch.linalg.vector_norm(raw_physical) + 1e-12), max=1.0)
    return raw * scale, scale


@torch.no_grad()
def sample_trained_weak_actions(
    strong_model,
    weak_model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    *,
    arm: str,
    lambda_value: float,
    num_steps: int = 10,
    return_debug: bool = False,
):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    if lambda_value not in (0.1, 0.25, 0.5):
        raise ValueError("lambda must be one of the frozen offline candidates")
    for model in (strong_model, weak_model):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("trained-weak sampler requires frozen eval models")

    strong_prefix, strong_pads, strong_cache = _prefix(
        strong_model, images, image_masks, lang_tokens, lang_masks, state
    )
    weak_prefix, weak_pads, weak_cache = _prefix(
        weak_model, images, image_masks, lang_tokens, lang_masks, state
    )
    if not torch.equal(strong_pads, weak_pads):
        raise RuntimeError("Strong/Weak prefix padding mismatch")

    x_t = noise.clone()
    dt = -1.0 / num_steps
    traces = []
    debug = None
    for step in range(num_steps):
        tau = 1.0 + step * dt
        strong = _velocity(strong_model, strong_pads, strong_cache, x_t, tau)
        weak = _velocity(weak_model, weak_pads, weak_cache, x_t, tau)
        correction, clip_scale = applied_correction(strong, weak, lambda_value)
        if arm == "V":
            used = strong
        elif arm == "W":
            used = weak
        elif arm == "M":
            used = 0.5 * (strong + weak)
        else:
            used = strong + correction
        if not bool(torch.isfinite(used).all()):
            raise RuntimeError(f"nonfinite {arm} velocity at flow step {step}")
        before = x_t
        x_t = x_t + dt * used
        traces.append({
            "step": step,
            "flow_time_tau": tau,
            "strong_norm": float(torch.linalg.vector_norm(strong)),
            "weak_norm": float(torch.linalg.vector_norm(weak)),
            "direction_norm": float(torch.linalg.vector_norm(strong - weak)),
            "applied_correction_norm": float(torch.linalg.vector_norm(correction)) if arm == "G" else 0.0,
            "clip_scale": float(clip_scale) if arm == "G" else None,
            "finite": True,
        })
        if return_debug and step == 0:
            debug = {
                "before": before.detach().cpu(),
                "strong": strong.detach().cpu(),
                "weak": weak.detach().cpu(),
                "used": used.detach().cpu(),
                "after": x_t.detach().cpu(),
            }
    trace = {
        "method": "same_trajectory_trained_weak",
        "arm": arm,
        "lambda": lambda_value if arm == "G" else None,
        "trust_region_kappa": TRUST_REGION_KAPPA if arm == "G" else None,
        "strong_prefix_sha256": tensor_sha256(strong_prefix),
        "weak_prefix_sha256": tensor_sha256(weak_prefix),
        "noise_sha256": tensor_sha256(noise),
        "step_traces": traces,
        "all_output_finite": bool(torch.isfinite(x_t).all()),
        "persistent_every_flow_step": True,
    }
    if return_debug:
        trace["_debug"] = debug
    return x_t, trace
