"""Audited Uniform + Semantic Orthogonal Dual-CD policy for OpenVLA."""

from __future__ import annotations

from typing import Sequence

import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.attention_mask_intervention import (
    AttentionMaskTrace,
    block_action_to_visual_attention,
    validate_attention_trace,
)
from research.semantic_token_cd.attention_mask_policy import AttentionMaskEntityCDInference
from research.semantic_token_cd.distractor_policy import ACTION_VOCAB_SIZE, _action_logits
from research.semantic_token_cd.orthogonal_cd import compute_orthogonal_dual_logits
from research.semantic_token_cd.spatial_grid_policy import get_grid_mask_indices


class OrthogonalDualCDInference(AttentionMaskEntityCDInference):
    """Use Uniform as the geometric basis and orthogonalize Semantic against it."""

    lambda_geom: float = 0.5
    lambda_sem: float = 0.35
    orthogonal_eps: float = 1e-8
    preserve_last_action_token: bool = True

    def _negative_scores(
        self,
        inputs,
        selected: Sequence[int],
        action_query_start: int,
        positive_features: torch.Tensor,
    ) -> tuple[torch.Tensor, AttentionMaskTrace, bool, bool]:
        with projector_intervention(self.vla) as feature_trace:
            with block_action_to_visual_attention(
                self.vla,
                selected,
                action_query_start=action_query_start,
                layer_start=self.attention_layer_start,
                layer_end=self.attention_layer_end,
                mask_value=self.attention_mask_value,
            ) as attention_trace:
                scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if feature_trace.before is None:
            raise RuntimeError("Negative branch projector hook was not invoked")
        feature_equal = torch.equal(positive_features, feature_trace.before)
        if not feature_equal:
            raise RuntimeError("Positive and negative visual features are not bit-identical")
        generated = scores.shape[0]
        validate_attention_trace(attention_trace, generated)
        truncated = generated < 7
        return scores, attention_trace, feature_equal, truncated

    @staticmethod
    def _align_scores(scores: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        if scores.shape[0] < 7:
            return torch.cat([scores, positive[scores.shape[0] :]], dim=0)
        return scores[:7]

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            positive_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if positive_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("Positive branch did not produce 7 scores and projector features")

        positive_features = positive_trace.before
        semantic_indices, meta = self._select(positive_features[0].numpy())
        uniform_indices = get_grid_mask_indices("strided_2x2")
        if not semantic_indices:
            raise RuntimeError("Entity selector produced an empty semantic branch")

        input_ids = inputs["input_ids"]
        appended_empty_token = int(not torch.all(input_ids[:, -1] == 29871))
        action_query_start = (
            positive_features.shape[1] + input_ids.shape[1] + appended_empty_token - 1
        )
        uniform_scores, uniform_attention, uniform_equal, uniform_truncated = (
            self._negative_scores(
                inputs, uniform_indices, action_query_start, positive_features
            )
        )
        semantic_scores, semantic_attention, semantic_equal, semantic_truncated = (
            self._negative_scores(
                inputs, semantic_indices, action_query_start, positive_features
            )
        )
        uniform_scores = self._align_scores(uniform_scores, positive_scores)
        semantic_scores = self._align_scores(semantic_scores, positive_scores)

        final_scores, full_diagnostics = compute_orthogonal_dual_logits(
            positive_scores,
            uniform_scores,
            semantic_scores,
            lambda_geom=self.lambda_geom,
            lambda_sem=self.lambda_sem,
            eps=self.orthogonal_eps,
        )
        action_start = int(self.vla.vocab_size) - ACTION_VOCAB_SIZE
        _, action_diagnostics = compute_orthogonal_dual_logits(
            positive_scores[:, action_start : action_start + ACTION_VOCAB_SIZE],
            uniform_scores[:, action_start : action_start + ACTION_VOCAB_SIZE],
            semantic_scores[:, action_start : action_start + ACTION_VOCAB_SIZE],
            lambda_geom=self.lambda_geom,
            lambda_sem=self.lambda_sem,
            eps=self.orthogonal_eps,
        )
        if self.preserve_last_action_token:
            final_scores[-1] = positive_scores[-1]
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite Orthogonal Dual-CD logits")

        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        positive = _action_logits(self, positive_scores)
        uniform = _action_logits(self, uniform_scores)
        semantic = _action_logits(self, semantic_scores)
        final = _action_logits(self, final_scores)

        meta.update({
            "uniform_token_ids": uniform_indices,
            "uniform_num_tokens": len(uniform_indices),
            "positive_token_ids": positive_scores.argmax(-1).detach().cpu().tolist(),
            "uniform_negative_token_ids": uniform_scores.argmax(-1).detach().cpu().tolist(),
            "semantic_negative_token_ids": semantic_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "lambda_geom": float(self.lambda_geom),
            "lambda_sem": float(self.lambda_sem),
            "orthogonal_eps": float(self.orthogonal_eps),
            "preserve_last_action_token": bool(self.preserve_last_action_token),
            "uniform_feature_equal": bool(uniform_equal),
            "semantic_feature_equal": bool(semantic_equal),
            "feature_equal": bool(uniform_equal and semantic_equal),
            "uniform_negative_truncated": bool(uniform_truncated),
            "semantic_negative_truncated": bool(semantic_truncated),
            "uniform_attention_mask": uniform_attention.as_dict(),
            "semantic_attention_mask": semantic_attention.as_dict(),
            "orthogonal_full_vocab": full_diagnostics,
            "orthogonal_action_vocab": action_diagnostics,
            "residual_norm": float(action_diagnostics["norm_r_geom"]),
            "degenerate": False,
        })
        self._episode_logits.append({
            "positive": positive,
            "uniform_negative": uniform,
            "semantic_negative": semantic,
            "final": final,
        })
        self._episode_trace.append(meta)
        return raw_action, actions, meta
