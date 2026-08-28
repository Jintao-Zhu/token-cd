import torch


def test_signed_direction_formula_contract():
    clean = torch.tensor([[[3.0, 4.0, 0.0]]])
    counterfactual = torch.zeros_like(clean)
    raw = clean - counterfactual
    clean_norm = torch.linalg.vector_norm(clean)
    raw_norm = torch.linalg.vector_norm(raw)
    scale = torch.clamp(0.25 * clean_norm / (raw_norm + 1e-12), max=1.0)
    toward = clean - 0.5 * raw * scale
    away = clean + 0.5 * raw * scale
    assert torch.allclose(toward, torch.tensor([[[2.625, 3.5, 0.0]]]))
    assert torch.allclose(away, torch.tensor([[[3.375, 4.5, 0.0]]]))
    assert torch.linalg.vector_norm(0.5 * raw * scale) <= 0.5 * 0.25 * clean_norm + 1e-6
