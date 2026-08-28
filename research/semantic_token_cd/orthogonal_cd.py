"""Gram-Schmidt composition for dual-branch contrastive decoding."""

from __future__ import annotations

from typing import Any

import torch


def compute_orthogonal_dual_logits(
    logits_pos: torch.Tensor,
    logits_uniform: torch.Tensor,
    logits_semantic: torch.Tensor,
    lambda_geom: float = 0.5,
    lambda_sem: float = 0.35,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Remove the geometric component from each semantic action-token residual.

    Inputs may have shape ``[action_tokens, vocab]`` or
    ``[batch, action_tokens, vocab]``. Projection is independent for every
    action token and always runs in float32 (float64 inputs remain float64).
    """
    if logits_pos.shape != logits_uniform.shape or logits_pos.shape != logits_semantic.shape:
        raise ValueError(
            "Positive, uniform, and semantic logits must have identical shapes, got "
            f"{tuple(logits_pos.shape)}, {tuple(logits_uniform.shape)}, and "
            f"{tuple(logits_semantic.shape)}"
        )
    if logits_pos.ndim not in (2, 3) or logits_pos.shape[-1] < 1:
        raise ValueError(f"Expected [A,V] or [B,A,V] logits, got {tuple(logits_pos.shape)}")
    if eps <= 0 or lambda_geom < 0 or lambda_sem < 0:
        raise ValueError("eps must be positive and both lambda values must be non-negative")
    if not all(torch.isfinite(value).all() for value in (logits_pos, logits_uniform, logits_semantic)):
        raise FloatingPointError("Orthogonal Dual-CD received non-finite logits")

    work_dtype = torch.float64 if logits_pos.dtype == torch.float64 else torch.float32
    pos = logits_pos.to(dtype=work_dtype)
    uniform = logits_uniform.to(dtype=work_dtype)
    semantic = logits_semantic.to(dtype=work_dtype)
    r_geom = pos - uniform
    r_sem = pos - semantic

    dot = torch.sum(r_sem * r_geom, dim=-1, keepdim=True)
    geom_norm_sq = torch.sum(r_geom.square(), dim=-1, keepdim=True)
    projection_coefficient = dot / (geom_norm_sq + eps)
    r_sem_orthogonal = r_sem - projection_coefficient * r_geom
    logits_final = pos + lambda_geom * r_geom + lambda_sem * r_sem_orthogonal
    if not torch.isfinite(logits_final).all():
        raise FloatingPointError("Orthogonal Dual-CD produced non-finite logits")

    with torch.no_grad():
        geom_norm = torch.linalg.vector_norm(r_geom, dim=-1)
        sem_norm = torch.linalg.vector_norm(r_sem, dim=-1)
        ortho_norm = torch.linalg.vector_norm(r_sem_orthogonal, dim=-1)
        cosine = dot.squeeze(-1) / (geom_norm * sem_norm + eps)
        ortho_dot = torch.sum(r_sem_orthogonal * r_geom, dim=-1)
        ortho_relative_error = torch.abs(ortho_dot) / (
            geom_norm * ortho_norm + eps
        )
        diagnostics: dict[str, Any] = {
            "norm_r_geom": float(geom_norm.mean().item()),
            "norm_r_sem": float(sem_norm.mean().item()),
            "norm_r_sem_ortho": float(ortho_norm.mean().item()),
            "cos_sim_raw": float(cosine.mean().item()),
            "ortho_ratio": float((ortho_norm / (sem_norm + eps)).mean().item()),
            "ortho_dot_max_abs": float(torch.abs(ortho_dot).max().item()),
            "ortho_relative_error_max": float(ortho_relative_error.max().item()),
            "projection_coefficient_mean": float(projection_coefficient.mean().item()),
            "per_action_cos_sim_raw": cosine.detach().cpu().tolist(),
            "per_action_ortho_ratio": (
                ortho_norm / (sem_norm + eps)
            ).detach().cpu().tolist(),
        }
    return logits_final.to(dtype=logits_pos.dtype), diagnostics
