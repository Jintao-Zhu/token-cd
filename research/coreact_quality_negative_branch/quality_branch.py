from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


FLOW_TIMES = tuple(round(1.0 - 0.1 * index, 1) for index in range(10))
EPSILON = 1e-12


@dataclass(frozen=True)
class BranchSpec:
    name: str
    affected_layers: int
    residual_scale: float


BRANCHES = (
    BranchSpec("W1_skip_last_1", 1, 0.0),
    BranchSpec("W2_skip_last_2", 2, 0.0),
    BranchSpec("W3_half_last_1", 1, 0.5),
    BranchSpec("W4_half_last_2", 2, 0.5),
)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def branch_scale_matrix(
    num_expert_layers: int,
    noise_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows = []
    for branch in BRANCHES:
        scale = torch.ones(num_expert_layers, device=device, dtype=dtype)
        scale[-branch.affected_layers :] = branch.residual_scale
        rows.append(scale.expand(noise_count, -1))
    return torch.cat(rows, dim=0)


def reconstruct_training_pair(model, actions: torch.Tensor, noise: torch.Tensor, time: torch.Tensor):
    """Reuse the exact helper called by SmolVLA's training forward."""
    return model.build_flow_training_pair(actions, noise, time)


@torch.no_grad()
def velocity_from_embeddings(
    model,
    prefix_embeddings: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    *,
    expert_residual_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    suffix_embeddings, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, timestep)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    attention_mask = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, suffix_embeddings],
        use_cache=False,
        fill_kv_cache=False,
        expert_residual_scales=expert_residual_scales,
    )
    suffix_out = suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def point_metrics(
    strong: torch.Tensor,
    weak: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float | bool]:
    strong_vector = strong[valid_mask].float()
    weak_vector = weak[valid_mask].float()
    target_vector = target[valid_mask].float()
    direction = strong_vector - weak_vector
    residual = target_vector - strong_vector

    strong_error = torch.mean((strong_vector - target_vector) ** 2)
    weak_error = torch.mean((weak_vector - target_vector) ** 2)
    delta_quality = weak_error - strong_error
    direction_norm = torch.linalg.vector_norm(direction)
    strong_norm = torch.linalg.vector_norm(strong_vector)
    weak_norm = torch.linalg.vector_norm(weak_vector)
    residual_norm = torch.linalg.vector_norm(residual)
    extrapolation_validity = torch.dot(direction, residual)
    ev_cosine = extrapolation_validity / (direction_norm * residual_norm + EPSILON)
    strong_weak_cosine = torch.dot(strong_vector, weak_vector) / (
        strong_norm * weak_norm + EPSILON
    )
    lambda_star = extrapolation_validity / (direction_norm**2 + EPSILON)
    relative_correction_norm = direction_norm / (strong_norm + EPSILON)

    output: dict[str, float | bool] = {
        "strong_error": float(strong_error),
        "weak_error": float(weak_error),
        "delta_quality": float(delta_quality),
        "quality_ordered": bool(delta_quality > 0),
        "extrapolation_validity": float(extrapolation_validity),
        "ev_positive": bool(extrapolation_validity > 0),
        "ev_cosine": float(ev_cosine),
        "lambda_star": float(lambda_star),
        "strong_weak_cosine": float(strong_weak_cosine),
        "relative_correction_norm": float(relative_correction_norm),
        "direction_norm": float(direction_norm),
    }
    for scale in (0.1, 0.25, 0.5):
        guided = strong_vector + scale * direction
        output[f"guided_error_delta_lambda_{scale:g}"] = float(
            torch.mean((guided - target_vector) ** 2) - strong_error
        )
    if not all(np.isfinite(value) for value in output.values() if isinstance(value, float)):
        raise FloatingPointError("nonfinite quality-branch metric")
    return output
