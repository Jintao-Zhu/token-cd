from __future__ import annotations

import numpy as np
import torch


EPS = 1e-12
GROUPS = {"translation": (0, 1, 2), "rotation": (3, 4, 5), "gripper": (6,)}


def exact_training_pair(actions, noise, time):
    expanded = time[:, None, None]
    return expanded * noise + (1.0 - expanded) * actions, noise - actions


def _core(strong, weak, target, valid):
    s, w, u = strong[valid].float(), weak[valid].float(), target[valid].float()
    d, g = s - w, u - s
    es, ew = s - u, w - u
    dot = torch.dot(d, g)
    return {
        "g_dot": float(dot),
        "g_positive": bool(dot > 0),
        "cosine": float(dot / (torch.linalg.vector_norm(d) * torch.linalg.vector_norm(g) + EPS)),
        "error_cosine": float(torch.dot(ew, es) / (torch.linalg.vector_norm(ew) * torch.linalg.vector_norm(es) + EPS)),
        "error_cosine_positive": bool(torch.dot(ew, es) > 0),
        "projected_severity_ratio": float(torch.dot(ew, es) / (torch.dot(es, es) + EPS)),
    }


def point_metrics(strong, weak, target, valid_steps, *, lambda_value=0.5, kappa=0.25):
    valid = valid_steps[:, None].expand_as(strong)
    direction = strong - weak
    raw_correction = lambda_value * direction
    scale = torch.clamp(kappa * torch.linalg.vector_norm(strong) / (torch.linalg.vector_norm(raw_correction) + EPS), max=1.0)
    applied_correction = raw_correction * scale
    raw, applied = strong + raw_correction, strong + applied_correction
    s, w, u = strong[valid].float(), weak[valid].float(), target[valid].float()
    strong_sse = torch.sum((s - u) ** 2)
    weak_sse = torch.sum((w - u) ** 2)
    output = _core(strong, weak, target, valid)
    output.update({
        "strong_mse": float(strong_sse / valid.sum()),
        "weak_mse": float(weak_sse / valid.sum()),
        "weak_to_strong_mse_ratio": float(weak_sse / (strong_sse + EPS)),
        "weak_inferior": bool(weak_sse > strong_sse),
        "r_gt_1": bool(output["projected_severity_ratio"] > 1.0),
        "delta_mse_raw": float((strong_sse - torch.sum((raw[valid].float() - u) ** 2)) / valid.sum()),
        "delta_mse_raw_positive": bool(torch.sum((raw[valid].float() - u) ** 2) < strong_sse),
        "delta_mse_applied": float((strong_sse - torch.sum((applied[valid].float() - u) ** 2)) / valid.sum()),
        "delta_mse_applied_positive": bool(torch.sum((applied[valid].float() - u) ** 2) < strong_sse),
        "direction_relative_norm": float(torch.linalg.vector_norm(direction) / (torch.linalg.vector_norm(strong) + EPS)),
        "clip_scale": float(scale),
        "clipped": bool(scale < 1.0 - 1e-7),
    })
    for name, dimensions in GROUPS.items():
        group_valid = valid[:, dimensions]
        group = _core(strong[:, dimensions], weak[:, dimensions], target[:, dimensions], group_valid)
        group_s = strong[:, dimensions][group_valid].float()
        group_u = target[:, dimensions][group_valid].float()
        group_guided = applied[:, dimensions][group_valid].float()
        group["delta_mse_applied"] = float(torch.mean((group_s - group_u) ** 2) - torch.mean((group_guided - group_u) ** 2))
        group["delta_mse_applied_positive"] = bool(group["delta_mse_applied"] > 0)
        for key, value in group.items():
            output[f"{name}_{key}"] = value
    if not all(np.isfinite(value) for value in output.values() if isinstance(value, float)):
        raise FloatingPointError("nonfinite capacity-Weak metric")
    return output
