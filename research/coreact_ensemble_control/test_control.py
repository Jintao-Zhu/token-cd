from __future__ import annotations

import torch

from research.coreact_ensemble_control.analyze import holm
from research.coreact_ensemble_control.methods import sample_dual_seed_average


class IdentitySampler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def sample_actions(self, images, image_masks, lang_tokens, lang_masks, state, *, noise):
        return noise.clone()


def test_dual_seed_average_clips_real_dimensions_and_never_masks() -> None:
    model = IdentitySampler().eval()
    first = torch.ones((1, 2, 9))
    second = torch.full((1, 2, 9), 3.0)
    output, trace = sample_dual_seed_average(
        model, [], [], torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1), first, second,
        action_dim=7, trust_region_kappa=0.25, matched_step=True,
    )
    assert torch.allclose(output[..., 7:], first[..., 7:])
    assert torch.allclose(torch.linalg.vector_norm(output[..., :7] - first[..., :7]), 0.25 * torch.linalg.vector_norm(first[..., :7]))
    assert trace["masked_token_count"] == 0
    assert trace["selected_indices"] == []
    assert trace["changed_indices"] == []
    assert trace["trust_region_clipping_active"]


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}
