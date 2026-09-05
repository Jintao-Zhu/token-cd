import numpy as np
import torch

from research.semantic_token_cd.pi0_shr_policy import (
    extract_entities,
    harmonic_reconstruct,
    shr_guided_velocity,
)


def test_shr_velocity_formula_and_gripper_lock():
    positive = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    negative = torch.tensor([[[0.0, 4.0, 30.0], [2.0, 1.0, 60.0]]])
    guided, residual = shr_guided_velocity(positive, negative, 0.5)
    assert torch.equal(guided[..., -1], positive[..., -1])
    assert torch.allclose(guided[..., :-1], positive[..., :-1] + 0.5 * residual[..., :-1])


def test_harmonic_reconstruction_is_finite():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(256, 8)).astype(np.float32)
    region = np.asarray([17, 18, 33, 34])
    solved = harmonic_reconstruct(features, region)
    assert solved.shape == (4, 8)
    assert np.isfinite(solved).all()


def test_entity_extraction():
    assert extract_entities("move the coke can near the apple") == ["coke can", "apple"]
    assert extract_entities("close the drawer") == ["drawer"]
