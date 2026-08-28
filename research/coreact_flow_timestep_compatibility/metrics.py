from __future__ import annotations

import numpy as np
import torch


EPS = 1e-12
GROUPS = {"translation": (0, 1, 2), "rotation": (3, 4, 5), "gripper": (6,)}


def exact_training_pair(actions, noise, time):
    """Exact formulas from the locked training worktree forward(), lines 784-786."""
    expanded = time[:, None, None]
    return expanded * noise + (1.0 - expanded) * actions, noise - actions


def _direction_metrics(strong, weak, target, valid_mask):
    s = strong[valid_mask].float()
    w = weak[valid_mask].float()
    u = target[valid_mask].float()
    d = s - w
    g = u - s
    dot = torch.dot(d, g)
    cosine = dot / (torch.linalg.vector_norm(d) * torch.linalg.vector_norm(g) + EPS)
    return {
        "g_dot": float(dot),
        "g_positive": bool(dot > 0),
        "cosine": float(cosine),
        "cosine_positive": bool(cosine > 0),
    }


def point_metrics(strong, weak, target, valid_steps, *, lambda_value=0.5, kappa=0.25):
    if strong.shape != weak.shape or strong.shape != target.shape or strong.shape[-1] != 7:
        raise ValueError("expected matched [chunk, 7] tensors")
    valid = valid_steps[:, None].expand_as(strong)
    direction = strong - weak
    raw_correction = lambda_value * direction
    clip_limit = kappa * torch.linalg.vector_norm(strong)
    clip_scale = torch.clamp(clip_limit / (torch.linalg.vector_norm(raw_correction) + EPS), max=1.0)
    applied_correction = raw_correction * clip_scale
    raw_guided = strong + raw_correction
    applied_guided = strong + applied_correction
    s, u = strong[valid].float(), target[valid].float()
    baseline_sse = torch.sum((s - u) ** 2)
    output = _direction_metrics(strong, weak, target, valid)
    output.update({
        "delta_mse_raw": float((baseline_sse - torch.sum((raw_guided[valid].float() - u) ** 2)) / valid.sum()),
        "delta_mse_raw_positive": bool(torch.sum((raw_guided[valid].float() - u) ** 2) < baseline_sse),
        "delta_mse_applied": float((baseline_sse - torch.sum((applied_guided[valid].float() - u) ** 2)) / valid.sum()),
        "delta_mse_applied_positive": bool(torch.sum((applied_guided[valid].float() - u) ** 2) < baseline_sse),
        "direction_relative_norm": float(torch.linalg.vector_norm(direction) / (torch.linalg.vector_norm(strong) + EPS)),
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "strong_norm": float(torch.linalg.vector_norm(strong)),
        "clip_scale": float(clip_scale),
        "clipped": bool(clip_scale < 1.0 - 1e-7),
    })
    for name, dimensions in GROUPS.items():
        group_valid = valid[:, dimensions]
        group = _direction_metrics(strong[:, dimensions], weak[:, dimensions], target[:, dimensions], group_valid)
        for key, value in group.items():
            output[f"{name}_{key}"] = value
    if not all(np.isfinite(value) for value in output.values() if isinstance(value, float)):
        raise FloatingPointError("nonfinite compatibility metric")
    return output, applied_correction
