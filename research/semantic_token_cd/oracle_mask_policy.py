"""Oracle Semantic-Mask Attention-CD policy.

Replaces the KMeans semantic selector (G_KMeans) with a GT instance-mask
selector (G_Oracle). Everything else — Attention-CD on layers [8,16),
lambda=0.5, mask value -1e4 — is bit-identical to the L8-15 KMeans arm, so any
success/harm delta is attributable to region localization quality alone.

Two oracle modes:
  "full"   : G = {i : o_i > tau},  o_i = per-patch overlap with the GT object mask
  "budget" : G = Top-B patches by o_i, where B = |G_KMeans| for this exact step
             (matches the KMeans arm's token count, isolating region quality
             from mask size).

Empty oracle (full mode, object not localized above tau in a step): fall back to
the clean (vanilla) action for that step — the oracle only fires CD when it can
confidently localize the object. Negative logits are still recorded (= clean) so
the logits array shape stays consistent.
"""
from __future__ import annotations

import numpy as np

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import (
    AuditedEntityCDInference,
    _action_logits,
)
from research.semantic_token_cd.spatial_grid_policy import SpatialGridAttentionCDInference


class OracleAttentionCDInference(SpatialGridAttentionCDInference):
    """Attention-CD whose negative-branch selection comes from a GT object mask."""

    oracle_mode: str  # "full" | "budget"
    _oracle_overlap: np.ndarray | None  # (256,) float32, injected by the rollout each step
    _oracle_tau: float = 0.5

    def _select(self, h: np.ndarray) -> tuple[list[int], dict]:
        overlap = self._oracle_overlap
        if overlap is None:
            raise RuntimeError("Oracle overlap not set before step")
        if overlap.shape != (256,):
            raise RuntimeError(f"Expected oracle overlap (256,), got {overlap.shape}")

        if self.oracle_mode == "budget":
            kmeans_idx, kmeans_meta = AuditedEntityCDInference._select(self, h)
            budget = len(kmeans_idx)
            order = np.argsort(overlap)[::-1]
            selected = sorted(int(i) for i in order[:budget])
            meta = {
                **kmeans_meta,
                "oracle_mode": "budget",
                "oracle_tau": self._oracle_tau,
                "oracle_budget": budget,
                "selected_token_ids": selected,
                "kmeans_reference_token_ids": kmeans_idx,
                "num_tokens": len(selected),
            }
        elif self.oracle_mode == "full":
            idx = np.flatnonzero(overlap > self._oracle_tau)
            selected = sorted(int(i) for i in idx)
            meta = {
                "oracle_mode": "full",
                "oracle_tau": self._oracle_tau,
                "selected_token_ids": selected,
                "num_tokens": len(selected),
                "instruction": self._selector_instr,
                "selected_entities": list(self._entities),
            }
        else:
            raise ValueError(f"Unknown oracle mode: {self.oracle_mode}")

        if len(selected) != len(set(selected)):
            raise RuntimeError("Oracle selector returned duplicate indices")
        return selected, meta

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        overlap = self._oracle_overlap
        if self.oracle_mode == "full" and overlap is not None and not np.any(overlap > self._oracle_tau):
            # Oracle cannot localize the object above tau: emit the clean action.
            inputs = self.process_inputs(image, task_description=task_description)
            with projector_intervention(self.vla) as trace:
                clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
            if clean_scores.shape[0] != 7:
                raise RuntimeError(f"Expected 7 clean action logits, got {clean_scores.shape[0]}")
            token_ids = clean_scores.argmax(dim=-1)
            raw = self._decode_actions(token_ids, self.unnorm_key)[None]
            raw_action, actions = self.postprocess_actions(raw)
            positive = _action_logits(self, clean_scores)
            self._episode_logits.append({"positive": positive, "negative": positive})
            meta = {
                "oracle_empty": True,
                "oracle_mode": self.oracle_mode,
                "oracle_tau": self._oracle_tau,
                "lambda": float(self.lambd),
                "degenerate": True,
                "n_selected": 0,
                "clean_token_ids": token_ids.detach().cpu().tolist(),
                "final_token_ids": token_ids.detach().cpu().tolist(),
            }
            self._episode_trace.append(meta)
            return raw_action, actions, meta
        return super().step(image, contrast_image, task_description, *args, **kwargs)
