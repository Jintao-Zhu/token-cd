"""Audited semantic and matched-random Attention-Mask CD policies."""

from __future__ import annotations

import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.attention_mask_intervention import (
    block_action_to_visual_attention,
    validate_attention_trace,
)
from research.semantic_token_cd.distractor_policy import (
    AuditedEntityCDInference,
    _action_logits,
)


class AttentionMaskEntityCDInference(AuditedEntityCDInference):
    """Entity selector with a feature-preserving late-layer attention negative branch."""

    attention_layer_start: int = 16
    attention_layer_end: int | None = None
    attention_mask_value: float | None = None

    def _attention_intervention_context(self, selected, action_query_start):
        """Context manager applying the negative-branch attention intervention."""
        return block_action_to_visual_attention(
            self.vla,
            selected,
            action_query_start=action_query_start,
            layer_start=self.attention_layer_start,
            layer_end=self.attention_layer_end,
            mask_value=self.attention_mask_value,
        )

    def _attention_trace_audit(self, attention_trace, n_negative):
        """Validate the intervention trace and return its serializable dict."""
        validate_attention_trace(attention_trace, n_negative)
        return attention_trace.as_dict()

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("Positive branch did not produce 7 scores and projector features")

        visual_features = positive_trace.before
        selected, meta = self._select(visual_features[0].numpy())
        if not selected:
            raise RuntimeError("Entity selector produced an empty negative branch")

        # The final prompt query predicts A0; subsequent cached action queries
        # predict A1..A6. Prismatic inserts all visual tokens after BOS.
        input_ids = inputs["input_ids"]
        appended_empty_token = int(not torch.all(input_ids[:, -1] == 29871))
        action_query_start = (
            visual_features.shape[1] + input_ids.shape[1] + appended_empty_token - 1
        )
        with projector_intervention(self.vla) as negative_trace:
            with self._attention_intervention_context(
                selected, action_query_start
            ) as attention_trace:
                negative_scores = self._forward_scores(
                    inputs, self.unnorm_key, do_sample=False
                )
        if negative_trace.before is None:
            raise RuntimeError("Negative branch projector hook was not invoked")
        feature_equal = torch.equal(visual_features, negative_trace.before)
        if not feature_equal:
            raise RuntimeError("Positive and negative visual features are not bit-identical")

        n_negative = negative_scores.shape[0]
        attention_mask_meta = self._attention_trace_audit(attention_trace, n_negative)
        meta["negative_truncated"] = n_negative < 7
        if n_negative < 7:
            negative_scores = torch.cat(
                [negative_scores, clean_scores[n_negative:]], dim=0
            )
        elif n_negative > 7:
            negative_scores = negative_scores[:7]

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1]
            - self.lambd * negative_scores[:-1]
        )
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite Attention-Mask CD logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        residual = (
            torch.log_softmax(torch.from_numpy(positive).float(), dim=-1)
            - torch.log_softmax(torch.from_numpy(negative).float(), dim=-1)
        )
        meta.update({
            "positive_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "residual_norm": float(torch.linalg.vector_norm(residual).item()),
            "lambda": float(self.lambd),
            "feature_equal": feature_equal,
            "feature_shape": list(visual_features.shape),
            "attention_mask": attention_mask_meta,
            "degenerate": False,
        })
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        return raw_action, actions, meta
