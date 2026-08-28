import torch

from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from research.coreact_quality_negative_branch.quality_branch import (
    branch_scale_matrix,
    point_metrics,
    reconstruct_training_pair,
)


def test_training_pair_reconstruction_uses_training_convention():
    actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    noise = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    time = torch.tensor([0.25])
    model = VLAFlowMatching
    x_t, target = reconstruct_training_pair(model, actions, noise, time)
    torch.testing.assert_close(x_t, 0.25 * noise + 0.75 * actions, rtol=0, atol=0)
    torch.testing.assert_close(target, noise - actions, rtol=0, atol=0)


def test_branch_scale_matrix_matches_preregistered_layers():
    scales = branch_scale_matrix(16, 2, device=torch.device("cpu"), dtype=torch.float32)
    assert scales.shape == (8, 16)
    assert scales[0, -1] == 0 and torch.all(scales[0, :-1] == 1)
    assert torch.all(scales[2, -2:] == 0) and torch.all(scales[2, :-2] == 1)
    assert scales[4, -1] == 0.5 and torch.all(scales[4, :-1] == 1)
    assert torch.all(scales[6, -2:] == 0.5) and torch.all(scales[6, :-2] == 1)


def test_metrics_separate_quality_ordering_and_extrapolation_validity():
    strong = torch.tensor([[[0.5]]])
    target = torch.tensor([[[1.0]]])
    valid = torch.ones_like(strong, dtype=torch.bool)
    good = point_metrics(strong, torch.tensor([[[0.0]]]), target, valid)
    bad = point_metrics(strong, torch.tensor([[[1.0]]]), target, valid)
    assert good["quality_ordered"] and good["ev_positive"]
    assert not bad["quality_ordered"] and not bad["ev_positive"]
