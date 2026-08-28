from __future__ import annotations

import pytest
import torch

from research.semantic_token_cd.orthogonal_cd import compute_orthogonal_dual_logits


def test_orthogonal_decomposition_per_action_token() -> None:
    generator = torch.Generator().manual_seed(7)
    positive = torch.randn(7, 256, generator=generator, dtype=torch.float64)
    geometric = torch.randn(7, 256, generator=generator, dtype=torch.float64)
    independent = torch.randn(7, 256, generator=generator, dtype=torch.float64)
    independent -= (
        (independent * geometric).sum(-1, keepdim=True)
        / geometric.square().sum(-1, keepdim=True)
    ) * geometric
    uniform = positive - geometric
    semantic = positive - (0.6 * geometric + 0.2 * independent)

    final, diagnostics = compute_orthogonal_dual_logits(
        positive, uniform, semantic, lambda_geom=0.5, lambda_sem=0.35
    )
    expected = positive + 0.5 * geometric + 0.35 * 0.2 * independent

    assert final.shape == positive.shape
    assert torch.allclose(final, expected, atol=1e-9, rtol=1e-9)
    assert diagnostics["ortho_relative_error_max"] < 1e-9
    assert diagnostics["ortho_ratio"] > 0


def test_zero_geometric_residual_preserves_semantic_residual() -> None:
    positive = torch.randn(1, 7, 32)
    semantic = positive - torch.randn(1, 7, 32)
    final, diagnostics = compute_orthogonal_dual_logits(
        positive, positive, semantic, lambda_geom=0.5, lambda_sem=0.3
    )
    assert torch.allclose(final, positive + 0.3 * (positive - semantic), atol=1e-6)
    assert diagnostics["norm_r_geom"] == 0.0


def test_rejects_shape_mismatch_and_nonfinite_values() -> None:
    value = torch.zeros(7, 256)
    with pytest.raises(ValueError, match="identical shapes"):
        compute_orthogonal_dual_logits(value, value[:, :-1], value)
    bad = value.clone()
    bad[0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="non-finite"):
        compute_orthogonal_dual_logits(bad, value, value)
