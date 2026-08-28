import pytest
import torch

from research.coreact_strong_weak_screening.sampler import (
    GUIDANCE_LAMBDA,
    combine_velocity,
    make_residual_scales,
)


class FakeModel:
    class Expert:
        num_expert_layers = 6

    vlm_with_expert = Expert()


def test_frozen_cfg_formula():
    strong = torch.tensor([1.0, 3.0])
    weak = torch.tensor([0.0, 5.0])
    actual = combine_velocity(strong, weak, GUIDANCE_LAMBDA)
    assert torch.equal(actual, torch.tensor([1.25, 2.5]))
    with pytest.raises(ValueError):
        combine_velocity(strong, weak, 0.5)


def test_w3_w4_residual_scale_contract():
    w3 = make_residual_scales(FakeModel(), "W3", 2, "cpu", torch.float32)
    w4 = make_residual_scales(FakeModel(), "W4", 2, "cpu", torch.float32)
    assert torch.equal(w3[0], torch.tensor([1, 1, 1, 1, 1, 0.5]))
    assert torch.equal(w4[0], torch.tensor([1, 1, 1, 1, 0.5, 0.5]))
    assert torch.equal(w3[0], w3[1]) and torch.equal(w4[0], w4[1])
