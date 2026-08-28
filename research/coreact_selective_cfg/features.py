from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "lerobot/src"))

from research.coreact_quality_negative_branch.quality_branch import BRANCHES, EPSILON


BRANCH_NAMES = tuple(branch.name for branch in BRANCHES)
PAIR_NAMES = tuple(
    (left, right) for left, right in combinations(range(len(BRANCH_NAMES)), 2)
)

LOCAL_FEATURES = (
    "timestep",
    "strong_weak_cosine",
    "relative_correction_norm",
    "log_strong_norm",
    "log_weak_norm",
    "log_direction_norm",
    "weak_to_strong_norm_ratio",
)

CONSENSUS_FEATURES = (
    "consensus_concentration",
    "branch_to_consensus_cosine",
    "branch_pairwise_cosine_mean",
    "branch_pairwise_cosine_min",
    "all_pairwise_cosine_mean",
    "all_pairwise_cosine_min",
    "all_pairwise_cosine_std",
    "direction_norm_cv",
    "relative_correction_norm_cv",
    *(f"direction_cosine_{BRANCH_NAMES[left]}__{BRANCH_NAMES[right]}" for left, right in PAIR_NAMES),
    *(f"relative_correction_norm_{name}" for name in BRANCH_NAMES),
)

GEOMETRY_LOCAL_FEATURES = LOCAL_FEATURES
GEOMETRY_CONSENSUS_FEATURES = LOCAL_FEATURES + CONSENSUS_FEATURES
FORBIDDEN_FEATURE_FRAGMENTS = (
    "target",
    "expert",
    "extrapolation_validity",
    "ev_",
    "delta_quality",
    "quality_ordered",
    "strong_error",
    "weak_error",
    "lambda_star",
    "guided_error",
)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.dot(left, right)
        / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right) + EPSILON)
    )


def online_geometry_features(
    strong: torch.Tensor,
    weak: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[dict[str, float]]:
    """Return target-free features for four weak branches at one matched point."""
    if weak.shape[0] != len(BRANCHES):
        raise ValueError(f"expected {len(BRANCHES)} weak branches, got {weak.shape[0]}")
    strong_vector = strong[valid_mask].float()
    weak_vectors = [value[valid_mask].float() for value in weak]
    directions = [strong_vector - value for value in weak_vectors]
    direction_norms = np.asarray(
        [float(torch.linalg.vector_norm(value)) for value in directions], dtype=np.float64
    )
    strong_norm = float(torch.linalg.vector_norm(strong_vector))
    weak_norms = np.asarray(
        [float(torch.linalg.vector_norm(value)) for value in weak_vectors], dtype=np.float64
    )
    relative_norms = direction_norms / (strong_norm + EPSILON)
    unit_directions = [value / (torch.linalg.vector_norm(value) + EPSILON) for value in directions]
    consensus = torch.stack(unit_directions).mean(dim=0)
    consensus_concentration = float(torch.linalg.vector_norm(consensus))

    pairwise: dict[tuple[int, int], float] = {
        pair: _cosine(directions[pair[0]], directions[pair[1]]) for pair in PAIR_NAMES
    }
    pair_values = np.asarray(list(pairwise.values()), dtype=np.float64)
    shared = {
        "consensus_concentration": consensus_concentration,
        "all_pairwise_cosine_mean": float(pair_values.mean()),
        "all_pairwise_cosine_min": float(pair_values.min()),
        "all_pairwise_cosine_std": float(pair_values.std()),
        "direction_norm_cv": float(direction_norms.std() / (direction_norms.mean() + EPSILON)),
        "relative_correction_norm_cv": float(
            relative_norms.std() / (relative_norms.mean() + EPSILON)
        ),
        **{
            f"direction_cosine_{BRANCH_NAMES[left]}__{BRANCH_NAMES[right]}": value
            for (left, right), value in pairwise.items()
        },
        **{
            f"relative_correction_norm_{name}": float(relative_norms[index])
            for index, name in enumerate(BRANCH_NAMES)
        },
    }
    output = []
    for index, (weak_vector, direction) in enumerate(zip(weak_vectors, directions, strict=True)):
        other_cosines = [
            _cosine(direction, directions[other])
            for other in range(len(directions))
            if other != index
        ]
        branch = {
            **shared,
            "strong_norm": strong_norm,
            "weak_norm": float(weak_norms[index]),
            "direction_norm": float(direction_norms[index]),
            "log_strong_norm": float(np.log1p(strong_norm)),
            "log_weak_norm": float(np.log1p(weak_norms[index])),
            "log_direction_norm": float(np.log1p(direction_norms[index])),
            "weak_to_strong_norm_ratio": float(weak_norms[index] / (strong_norm + EPSILON)),
            "strong_weak_cosine": _cosine(strong_vector, weak_vector),
            "relative_correction_norm": float(relative_norms[index]),
            "branch_to_consensus_cosine": _cosine(direction, consensus),
            "branch_pairwise_cosine_mean": float(np.mean(other_cosines)),
            "branch_pairwise_cosine_min": float(np.min(other_cosines)),
        }
        if not all(np.isfinite(value) for value in branch.values()):
            raise FloatingPointError("nonfinite online geometry feature")
        output.append(branch)
    return output


def assert_deployable_feature_names(names: tuple[str, ...] | list[str]) -> None:
    offenders = [
        name
        for name in names
        if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if offenders:
        raise ValueError(f"target-derived features are forbidden: {offenders}")
