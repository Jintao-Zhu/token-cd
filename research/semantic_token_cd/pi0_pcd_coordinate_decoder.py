"""Action-coordinate KDE-PM decoder matching the official PCD PiZero rule.

Inputs are positive and negative action samples with shape [N, T, D]. The
official decoder estimates a univariate density for each flattened T x D
coordinate, applies the PCD density ratio independently, and replaces only
the first six motion dimensions. Clipping is deliberately performed after
decoding, as in the official PiZero wrapper.
"""
from __future__ import annotations

import torch


def _gaussian_kde_at_samples(x: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate each coordinate's Gaussian KDE at its positive candidates.

    Args:
        x: Positive samples [N, T, D].
        reference: Density samples [N, T, D].
    Returns:
        Density [T*D, N] and per-coordinate Scott bandwidth [T*D].
    """
    if x.ndim != 3 or reference.shape != x.shape:
        raise ValueError(f"expected matching [N,T,D] samples, got {tuple(x.shape)} and {tuple(reference.shape)}")
    n, steps, dims = x.shape
    x_flat = x.reshape(n, steps * dims).transpose(0, 1)
    ref_flat = reference.reshape(n, steps * dims).transpose(0, 1)
    bandwidth = 1.06 * torch.std(ref_flat, dim=-1) * (n ** (-1.0 / 5.0))
    z = (x_flat.unsqueeze(-1) - ref_flat.unsqueeze(1)) / bandwidth[:, None, None]
    kernel = (1.0 / (2.0 * torch.pi) ** 0.5) * torch.exp(-0.5 * z.square())
    density = kernel.sum(dim=-1) / (n * bandwidth[:, None])
    return density, bandwidth


@torch.inference_mode()
def official_pcd_coordinate_decode(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    alpha: float = 0.2,
    keep_threshold: float = 0.5,
    motion_dims: int = 6,
    clip_value: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Apply the official PCD KDE-PM coordinate-wise decoding rule.

    Returns one [1,T,D] chunk and compact diagnostics. Candidate samples are
    expected to be unclipped; ``clip_value`` is applied only to the final
    decoded chunk.
    """
    if positive.ndim != 3 or negative.shape != positive.shape:
        raise ValueError("positive and negative candidates must have matching [N,T,D] shapes")
    if positive.shape[0] < 2:
        raise ValueError("KDE-PM requires at least two candidates")
    if alpha < 0 or keep_threshold < 0:
        raise ValueError("alpha and keep_threshold must be nonnegative")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise FloatingPointError("non-finite action candidate")

    n, steps, dims = positive.shape
    data = positive.reshape(n, steps * dims).transpose(0, 1)
    pos_density, pos_bw = _gaussian_kde_at_samples(positive, positive)
    neg_density, neg_bw = _gaussian_kde_at_samples(positive, negative)
    ratio = pos_density / neg_density
    final_density = pos_density * ratio.pow(float(alpha))
    threshold = float(keep_threshold) * pos_density.max(dim=-1, keepdim=True).values
    final_density = final_density.clone()
    final_density[pos_density < threshold] = 0.0
    # This rescaling is part of the reference implementation. It does not
    # alter argmax, but retain it for behavioral fidelity and diagnostics.
    final_density = final_density / final_density.max(dim=-1, keepdim=True).values * pos_density.max(dim=-1, keepdim=True).values

    coordinates = torch.arange(steps * dims, device=positive.device)
    positive_mode = data[coordinates, pos_density.argmax(dim=-1)].reshape(steps, dims)
    guided_mode = data[coordinates, final_density.argmax(dim=-1)].reshape(steps, dims)
    decoded = positive_mode.clone()
    decoded[:, : min(int(motion_dims), dims)] = guided_mode[:, : min(int(motion_dims), dims)]
    if clip_value is not None:
        decoded = decoded.clamp(-float(clip_value), float(clip_value))

    weights = final_density / final_density.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(final_density.dtype).tiny)
    diagnostics = {
        "decoder": "official_pcd_coordinatewise_kde_pm",
        "candidate_count": int(n),
        "coordinates": int(steps * dims),
        "motion_dims_replaced": int(min(int(motion_dims), dims)),
        "keep_threshold": float(keep_threshold),
        "alpha": float(alpha),
        "positive_bandwidth_mean": float(pos_bw.mean().item()),
        "negative_bandwidth_mean": float(neg_bw.mean().item()),
        "coordinate_max_weights_median": float(weights.max(dim=-1).values.median().item()),
    }
    return decoded.unsqueeze(0), diagnostics
