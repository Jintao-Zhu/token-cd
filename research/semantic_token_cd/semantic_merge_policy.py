"""Semantic-Merge vs Attention-K8 CD (Phase 1B mechanism study, guided prefix).

Two arms that share the EXACT same KMeans K=8 semantic selector on the clean
projector features (source/target entity groups), differing only in *how the
selected groups are degraded*:

    semantic_attn_k8_l8_15   — block Action Query -> selected visual keys
                               (layers [8,16), mask -1e4). The information is
                               present but its access path is severed.
    semantic_merge_k8_eta100 — merge each selected KMeans group to its own
                               prototype (v_i -> mu_G for i in G, eta=1.0).
                               All attention paths stay intact; only the
                               within-group detail is collapsed. Source and
                               target groups are merged SEPARATELY (never into a
                               shared mean); non-selected tokens are untouched.

Both use the guided autoregressive prefix (the negative branch is teacher-forced
on the clean branch's greedy tokens, NOT independently greedy), matching the
Phase 1A convention, so the residual r_q = z_q+ - z_q- isolates the visual
degradation alone. This is a *mechanism/development* comparison, not an
independent pre-registration: eta=1.0 was chosen from the Phase 1A global GSM
sweep (GSM100 was the only macro-positive arm), and seeds 200-299 are reused.

CD (identical to the guided CD in global_merge_policy):
    z_q* = z_q+ + 0.5 * (z_q+ - z_q-)   for action dims 0..5
    z_6* = z_6+                          (gripper keeps clean)
"""
from __future__ import annotations

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.attention_mask_policy import (
    AttentionMaskEntityCDInference,
)
from research.semantic_token_cd.distractor_policy import (
    AuditedEntityCDInference,
    _action_logits,
)
from research.semantic_token_cd.global_merge_policy import (
    _action_query_start,
    guided_forward_scores,
    projector_merge_intervention,
    validate_guided_attention_trace,
)


def merge_selected_groups(
    V: torch.Tensor,
    labels: np.ndarray,
    selected_groups: list[int],
    eta: float,
) -> torch.Tensor:
    """Merge each selected KMeans group to its own prototype: v_i -> (1-eta) v_i + eta mu_G.

    ``V`` is the projector output [1, 256, d] (float32, cpu). ``labels`` is the
    (256,) int64 KMeans assignment. ``selected_groups`` are the source/target
    group IDs (already deduplicated by the selector). Only tokens whose label is
    in ``selected_groups`` are modified; every other token is bit-identical.
    Source and target are distinct labels, so each is merged to its OWN mean.
    """
    if V.ndim != 3 or V.shape[1] != 256:
        raise ValueError(f"Expected V [1,256,d], got {tuple(V.shape)}")
    if labels.shape != (256,):
        raise ValueError(f"Expected labels (256,), got {tuple(labels.shape)}")
    out = V.clone()
    for g in selected_groups:
        members = np.flatnonzero(labels == g)
        if members.size == 0:
            continue
        idx = torch.as_tensor(members, dtype=torch.long)
        mu = out[0, idx].mean(dim=0, keepdim=True)
        out[0, idx] = (1.0 - eta) * out[0, idx] + eta * mu
    return out


class GuidedSemanticAttentionCDInference(AttentionMaskEntityCDInference):
    """KMeans K=8 semantic Attention-CD with a guided autoregressive prefix.

    Negative branch = the SAME selected source/target KMeans groups as
    SemanticMergeCDInference, but degraded by blocking the Action Query ->
    selected visual key connections on layers [8,16) instead of merging. The
    guided step mirrors GuidedAttentionCDInference minus the Oracle-GT fallback.
    """

    selection_mode: str = "semantic"
    # L8-15, mask -1e4 (same as the L8-15 KMeans arm and the Oracle-GT baseline).
    attention_layer_start: int = 8
    attention_layer_end: int = 16
    attention_mask_value: float = -1e4

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("Positive branch did not produce 7 scores and projector features")

        V = positive_trace.before  # [1,256,d] float32 cpu
        clean_token_ids = clean_scores.argmax(dim=-1)

        selected, meta = self._select(V[0].numpy())
        if not selected:
            raise RuntimeError("KMeans semantic selector produced an empty negative branch")

        action_query_start = _action_query_start(inputs, V.shape[1])
        with projector_intervention(self.vla) as negative_trace:
            with self._attention_intervention_context(selected, action_query_start) as attention_trace:
                negative_scores = guided_forward_scores(self.vla, inputs, clean_token_ids, V.shape[1])
        if negative_trace.before is None:
            raise RuntimeError("Negative branch projector hook was not invoked")
        feature_equal = torch.equal(V, negative_trace.before)
        if not feature_equal:
            raise RuntimeError("Positive and negative visual features are not bit-identical")

        validate_guided_attention_trace(attention_trace, negative_scores.shape[0])
        meta["attention_mask"] = attention_trace.as_dict()

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1] - self.lambd * negative_scores[:-1]
        )
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite guided Semantic-Attention-CD logits")
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
            "feature_shape": list(V.shape),
            "guided_prefix": True,
            "degenerate": False,
        })
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        return raw_action, actions, meta


class SemanticMergeCDInference(AuditedEntityCDInference):
    """KMeans K=8 semantic prototype-merge CD with a guided autoregressive prefix.

    Negative branch = the SAME selected source/target KMeans groups as the
    Attention arm, but each group is merged to its own prototype (eta=1.0). No
    attention blocking; every attention path is preserved. The merge is applied
    at the projector output, before the LLM transformer.
    """

    selection_mode: str = "semantic"
    eta: float = 1.0

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("Positive branch did not produce 7 scores and projector features")

        V = positive_trace.before  # [1,256,d] float32 cpu
        clean_token_ids = clean_scores.argmax(dim=-1)

        h = V[0].numpy()  # [256,d] float32
        labels, selected_groups, per_entity_score = self._semantic_clusters(h)
        selected = sorted(
            int(index)
            for group_id in selected_groups
            for index in np.flatnonzero(labels == group_id)
        )
        if not selected:
            raise RuntimeError("KMeans semantic selector produced an empty negative branch")

        # Integrity: within-group variance (before merge) must be strictly positive,
        # so eta=1.0 collapse is a real detail-removal, not a no-op on a singleton.
        group_sizes = [int(np.sum(labels == g)) for g in selected_groups]
        var_before = []
        for g in selected_groups:
            members = h[labels == g]
            if members.shape[0] > 1:
                var_before.append(float(np.mean(np.sum((members - members.mean(0)) ** 2, axis=1))))
            else:
                var_before.append(0.0)

        V_tilde = merge_selected_groups(V, labels, selected_groups, self.eta)
        with projector_merge_intervention(self.vla, V_tilde) as negative_trace:
            negative_scores = guided_forward_scores(self.vla, inputs, clean_token_ids, V.shape[1])
        if negative_trace["before"] is None or negative_trace["after"] is None:
            raise RuntimeError("Merge hook was not invoked")
        feature_equal = torch.equal(V, negative_trace["before"])
        if not feature_equal:
            raise RuntimeError("Negative branch projector 'before' != clean V")

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1] - self.lambd * negative_scores[:-1]
        )
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite Semantic-Merge-CD logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        residual = (
            torch.log_softmax(torch.from_numpy(positive).float(), dim=-1)
            - torch.log_softmax(torch.from_numpy(negative).float(), dim=-1)
        )
        meta = {
            "eta": float(self.eta),
            "selection_mode": self.selection_mode,
            "instruction": self._selector_instr,
            "selected_entities": list(self._entities),
            "selected_group_ids": [int(g) for g in selected_groups],
            "selected_token_ids": selected,
            "num_tokens": len(selected),
            "num_groups": len(selected_groups),
            "selected_group_sizes": group_sizes,
            "within_group_variance_before": var_before,
            "per_entity_score": [float(s) for s in per_entity_score],
            "language_score": float(max(per_entity_score)) if per_entity_score else 0.0,
            "kmeans_K": int(self.kmeans_K),
            "positive_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "residual_norm": float(torch.linalg.vector_norm(residual).item()),
            "lambda": float(self.lambd),
            "feature_equal": feature_equal,
            "feature_shape": list(V.shape),
            "n_tokens_negative": int(V.shape[1]),
            "guided_prefix": True,
            "degenerate": False,
        }
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        return raw_action, actions, meta
