"""Positive-Support Constrained SHR (PSC-SHR)."""
from __future__ import annotations

import numpy as np
import torch

from research.semantic_token_cd.st_shr_policy import STSHRCDInference


ACTION_BINS = 256


class PositiveSupportSHRInference(STSHRCDInference):
    """Canonical beta=0 SHR with Top-K positive support at logit fusion only."""

    support_top_k: int = 10

    def _combine_action_scores(self, clean_scores, negative_scores):
        if self.support_top_k not in (10, 20):
            raise ValueError("PSC rollout is locked to K=10 or K=20")
        fused = clean_scores.clone()
        fused[:-1] = 1.5 * clean_scores[:-1] - 0.5 * negative_scores[:-1]
        start = int(self.vla.vocab_size) - ACTION_BINS
        positive_action = clean_scores[:, start : start + ACTION_BINS]
        negative_action = negative_scores[:, start : start + ACTION_BINS]
        fused_action = fused[:, start : start + ACTION_BINS]
        candidate = torch.topk(positive_action[:-1], self.support_top_k, dim=-1).indices
        filtered = torch.full_like(fused, -torch.inf)
        filtered[-1] = clean_scores[-1]
        filtered[:-1].scatter_(1, candidate + start,
                               torch.gather(fused[:-1], 1, candidate + start))
        shr_ids = fused.argmax(-1)
        psc_ids = filtered.argmax(-1)
        shr_bins = shr_ids - start
        if torch.any((shr_bins[:-1] < 0) | (shr_bins[:-1] >= ACTION_BINS)):
            raise RuntimeError("raw SHR winner outside action vocabulary on a guided dimension")
        selected_positive = torch.gather(
            positive_action[:-1], 1, shr_bins[:-1, None]
        )[:, 0]
        ranks = 1 + (positive_action[:-1] > selected_positive[:, None]).sum(-1)
        support_valid = bool(torch.all(torch.any(candidate == (psc_ids[:-1] - start)[:, None], dim=-1)))
        gripper_equal = bool(psc_ids[-1] == clean_scores[-1].argmax())
        if not support_valid or not gripper_equal:
            raise RuntimeError("PSC positive-support/gripper invariant failed")
        self._episode_psc_logits.append({
            "positive_logits": positive_action.detach().to(torch.float16).cpu().numpy(),
            "negative_logits": negative_action.detach().to(torch.float16).cpu().numpy(),
            "fused_logits": fused_action.detach().to(torch.float16).cpu().numpy(),
            "filtered_logits": filtered[:, start : start + ACTION_BINS].detach().to(torch.float16).cpu().numpy(),
        })
        return filtered, {
            "psc_top_k": self.support_top_k,
            "shr_token_ids_before_constraint": shr_ids.detach().cpu().tolist(),
            "psc_token_ids": psc_ids.detach().cpu().tolist(),
            "positive_rank_of_shr": ranks.detach().cpu().tolist() + [1],
            "changed_by_filter": (psc_ids != shr_ids).detach().cpu().tolist(),
            "positive_support_valid": support_valid,
            "gripper_passthrough": gripper_equal,
        }
