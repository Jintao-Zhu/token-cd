from __future__ import annotations

from typing import Sequence

import torch

from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_self_guidance.sampler import _native_prefix


ARMS = ("Vanilla", "W4_only", "W3_CFG", "W4_CFG")
GUIDANCE_LAMBDA = 0.25


def make_residual_scales(model, branch: str, batch_size: int, device, dtype) -> torch.Tensor:
    if branch not in {"W3", "W4"}:
        raise ValueError(f"unknown branch: {branch}")
    scales = torch.ones(
        batch_size, model.vlm_with_expert.num_expert_layers, device=device, dtype=dtype
    )
    scales[:, -1 if branch == "W3" else -2 :] = 0.5
    return scales


def combine_velocity(strong: torch.Tensor, weak: torch.Tensor, lambda_value: float) -> torch.Tensor:
    if lambda_value != GUIDANCE_LAMBDA:
        raise ValueError(f"lambda is frozen at {GUIDANCE_LAMBDA}")
    return strong + lambda_value * (strong - weak)


def _velocity(model, pad_masks, cache, x_t, tau, scales=None):
    timestep = torch.full((x_t.shape[0],), tau, device=x_t.device, dtype=torch.float32)
    return model.denoise_step(
        pad_masks, cache, x_t, timestep, expert_residual_scales=scales
    ).to(dtype=torch.float32)


@torch.no_grad()
def sample_strong_weak_actions(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    *,
    arm: str,
    lambda_value: float = GUIDANCE_LAMBDA,
    num_steps: int = 10,
    return_debug: bool = False,
) -> tuple[torch.Tensor, dict]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Strong-Weak screening requires a frozen eval model")
    if lambda_value != GUIDANCE_LAMBDA:
        raise ValueError(f"lambda is frozen at {GUIDANCE_LAMBDA}")

    prefix, pad_masks, _att_masks, _span_map, cache = _native_prefix(
        model, images, image_masks, lang_tokens, lang_masks, state
    )
    branch = "W3" if arm == "W3_CFG" else "W4" if arm in {"W4_only", "W4_CFG"} else None
    scales = (
        make_residual_scales(model, branch, noise.shape[0], noise.device, noise.dtype)
        if branch
        else None
    )
    x_t = noise.clone()
    dt = -1.0 / num_steps
    step_traces = []
    debug = None
    for step in range(num_steps):
        tau = 1.0 + step * dt
        x_before = x_t
        strong = None
        weak = None
        if arm == "Vanilla":
            used = _velocity(model, pad_masks, cache, x_t, tau)
        elif arm == "W4_only":
            weak = _velocity(model, pad_masks, cache, x_t, tau, scales)
            used = weak
        else:
            strong = _velocity(model, pad_masks, cache, x_t, tau)
            weak = _velocity(model, pad_masks, cache, x_t, tau, scales)
            used = combine_velocity(strong, weak, lambda_value)
        if not torch.isfinite(used).all():
            raise RuntimeError(f"nonfinite velocity in {arm} at flow step {step}")
        x_t = x_t + dt * used
        correction = strong - weak if strong is not None else torch.zeros_like(used)
        step_traces.append(
            {
                "step": step,
                "flow_time_tau": tau,
                "strong_norm": float(torch.linalg.vector_norm(strong)) if strong is not None else None,
                "weak_norm": float(torch.linalg.vector_norm(weak)) if weak is not None else None,
                "strong_minus_weak_norm": float(torch.linalg.vector_norm(correction)),
                "applied_correction_norm": float(
                    torch.linalg.vector_norm(lambda_value * correction)
                ),
                "all_dimensions_guided": arm in {"W3_CFG", "W4_CFG"},
                "finite": True,
            }
        )
        if return_debug and step == 0:
            debug = {
                "x_before": x_before.detach().cpu(),
                "strong": strong.detach().cpu() if strong is not None else None,
                "weak": weak.detach().cpu() if weak is not None else None,
                "used": used.detach().cpu(),
                "x_after": x_t.detach().cpu(),
                "scales": scales.detach().cpu() if scales is not None else None,
            }
    trace = {
        "method": "structural_strong_weak_cfg",
        "arm": arm,
        "branch": branch,
        "lambda": lambda_value if arm in {"W3_CFG", "W4_CFG"} else None,
        "formula": "v_strong + 0.25 * (v_strong - v_weak)"
        if arm in {"W3_CFG", "W4_CFG"}
        else "v_weak" if arm == "W4_only" else "v_strong",
        "prefix_sha256": tensor_sha256(prefix),
        "noise_sha256": tensor_sha256(noise),
        "step_traces": step_traces,
        "all_output_finite": bool(torch.isfinite(x_t).all()),
        "no_ev_gate": True,
        "no_clipping": True,
    }
    if return_debug:
        trace["_debug"] = debug
    return x_t, trace
