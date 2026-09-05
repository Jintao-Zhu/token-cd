from __future__ import annotations

import unittest
import torch

from research.semantic_token_cd.projected_shr_policy import projected_shr_action_logits


class ProjectedSHRTest(unittest.TestCase):
    def test_eta_one_matches_centered_original_shr_up_to_argmax(self) -> None:
        generator = torch.Generator().manual_seed(11)
        positive = torch.randn(6, 256, generator=generator)
        negative = torch.randn(6, 256, generator=generator)
        projected, diagnostics = projected_shr_action_logits(
            positive, negative, lambd=0.5, eta=1.0
        )
        original = positive + 0.5 * (positive - negative)
        self.assertTrue(torch.equal(projected.argmax(-1), original.argmax(-1)))
        self.assertTrue(torch.equal(
            projected.argmax(-1), diagnostics["original_shr"].argmax(-1)
        ))

    def test_eta_zero_is_orthogonal_per_action_dimension(self) -> None:
        generator = torch.Generator().manual_seed(23)
        positive = torch.randn(6, 256, generator=generator, dtype=torch.float64)
        negative = torch.randn(6, 256, generator=generator, dtype=torch.float64)
        _, diagnostics = projected_shr_action_logits(positive, negative, eta=0.0)
        pc = positive - positive.mean(-1, keepdim=True)
        dot = (diagnostics["orthogonal"].double() * pc).sum(-1)
        scale = diagnostics["orthogonal"].double().norm(dim=-1) * pc.norm(dim=-1)
        self.assertTrue(torch.all(dot.abs() / (scale + 1e-8) < 2e-6))

    def test_constant_logit_offsets_do_not_change_projected_argmax(self) -> None:
        positive = torch.randn(6, 256)
        negative = torch.randn(6, 256)
        first, _ = projected_shr_action_logits(positive, negative, eta=0.25)
        second, _ = projected_shr_action_logits(
            positive + 17.0, negative - 9.0, eta=0.25
        )
        self.assertTrue(torch.equal(first.argmax(-1), second.argmax(-1)))

    def test_rejects_wrong_shape_and_nonfinite(self) -> None:
        value = torch.zeros(6, 256)
        with self.assertRaises(ValueError):
            projected_shr_action_logits(value, value[:, :-1])
        bad = value.clone()
        bad[0, 0] = torch.inf
        with self.assertRaises(FloatingPointError):
            projected_shr_action_logits(bad, value)


if __name__ == "__main__":
    unittest.main()
