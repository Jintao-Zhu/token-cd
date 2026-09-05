"""Adaptive-SHR v1: action-shift-controlled guidance strength.

Everything upstream of score combination is inherited verbatim from the
locked SHR harmonic implementation.  The candidate lambda=0.5 action is
compared with vanilla on the six non-gripper action tokens.  A shift ratio
strictly greater than 0.33 selects lambda=0.25 for that control step.
"""
from __future__ import annotations

import torch

from research.semantic_token_cd.st_shr_policy import STSHRCDInference


INITIAL_LAMBDA = 0.5
REDUCED_LAMBDA = 0.25
SHIFT_THRESHOLD = 0.33
N_SHIFT_TOKENS = 6


class AdaptiveSHRCDInference(STSHRCDInference):
    """Locked SHR harmonic branch with the Adaptive-SHR v1 score controller."""

    def _forward_scores(self, inputs, unnorm_key, **kwargs):
        """Decode exactly seven action positions, even if an action ID equals EOS.

        OpenVLA action tokens share the language-model vocabulary.  On a small
        number of rollout states the greedy action ID is also configured as an
        EOS ID, causing ``generate`` to stop before all seven action dimensions
        have been scored.  EOS is not a valid stopping condition for this
        fixed-width robot-action protocol, so disable only that stopping rule.
        This leaves every score before a would-be early EOS unchanged.
        """
        # Transformers 4.40 may fall back to the model generation config when
        # an explicit ``None`` is passed.  An empty EOS set creates a stopping
        # criterion that can never match while preserving the unmodified
        # logits (unlike ``min_new_tokens``, which suppresses EOS logits).
        kwargs["eos_token_id"] = []
        scores = super()._forward_scores(inputs, unnorm_key, **kwargs)
        if scores.shape[0] != N_SHIFT_TOKENS + 1:
            raise RuntimeError(
                "Adaptive-SHR positive branch must produce exactly 7 action scores; "
                f"got {scores.shape[0]}"
            )
        return scores

    def _combine_action_scores(
        self, clean_scores: torch.Tensor, negative_scores: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        if clean_scores.shape[0] != N_SHIFT_TOKENS + 1:
            raise RuntimeError(
                f"Adaptive-SHR expects 6 action tokens plus gripper; got {clean_scores.shape[0]}"
            )
        if abs(float(self.lambd) - INITIAL_LAMBDA) > 1e-12:
            raise RuntimeError("Adaptive-SHR initial lambda must remain locked at 0.5")

        shr_scores = clean_scores.clone()
        shr_scores[:-1] = (
            (1 + INITIAL_LAMBDA) * clean_scores[:-1]
            - INITIAL_LAMBDA * negative_scores[:-1]
        )
        vanilla_ids = clean_scores.argmax(dim=-1)
        shr_ids = shr_scores.argmax(dim=-1)
        shift_count = int((shr_ids[:-1] != vanilla_ids[:-1]).sum().item())
        shift_ratio = shift_count / N_SHIFT_TOKENS
        adapt_trigger = shift_ratio > SHIFT_THRESHOLD
        final_lambda = REDUCED_LAMBDA if adapt_trigger else INITIAL_LAMBDA

        if adapt_trigger:
            final_scores = clean_scores.clone()
            final_scores[:-1] = (
                (1 + final_lambda) * clean_scores[:-1]
                - final_lambda * negative_scores[:-1]
            )
        else:
            final_scores = shr_scores
        adaptive_ids = final_scores.argmax(dim=-1)

        return final_scores, {
            "initial_lambda": INITIAL_LAMBDA,
            "final_lambda": final_lambda,
            "adapt_trigger": adapt_trigger,
            "shift_ratio": shift_ratio,
            "shift_count": shift_count,
            "shift_denominator": N_SHIFT_TOKENS,
            "shift_threshold": SHIFT_THRESHOLD,
            "vanilla_action": vanilla_ids.detach().cpu().tolist(),
            "shr_action": shr_ids.detach().cpu().tolist(),
            "adaptive_action": adaptive_ids.detach().cpu().tolist(),
            "gripper_excluded_from_shift": True,
            "gripper_token_unchanged": bool(
                adaptive_ids[-1].item() == vanilla_ids[-1].item()
            ),
        }
