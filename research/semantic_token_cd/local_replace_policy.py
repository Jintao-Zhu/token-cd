"""Local Feature-Inpainting CD (LF-CD) policy.

Replaces the Attention-CD negative branch (Action -> selected-visual-key
attention blocking) with a *feature-space* intervention: the target visual
tokens' projector features are overwritten by the mean of the surrounding
background "ring" tokens, producing a negative branch that is closer to a
true object-absent counterfactual than attention blocking.

  Positive  z+ = F(V, instr)                       (V = 256 projector features)
  Negative  z- = F(V~, instr),   V~_i = mu_bg(R(G)) for i in G, V~_i = V_i else
  CD        z* = z+ + lambda * (z+ - z-)            (gripper token 6 keeps clean)

The target region G comes from the GT/oracle instance mask (same G_Oracle as the
Attention-CD arm), so the ONLY difference vs Oracle Attention-CD is *how the
negative branch is produced*.  lambda, seed, mask, and coverage are identical.

Per-object separation (move_near source vs target): each entity's ring is
computed independently and excludes every other entity's target region, so a
"background" mean is never contaminated by the other task object.
"""
from __future__ import annotations

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import (
    AuditedEntityCDInference,
    _action_logits,
)

GRID = 16
DEFAULT_MIN_RING = 8
DEFAULT_MAX_RADIUS = 4


def dilate_indices(indices: set[int], radius: int) -> set[int]:
    """8-neighbourhood (Chebyshev) dilation of patch indices on the 16x16 grid."""
    out: set[int] = set()
    for i in indices:
        r, c = divmod(int(i), GRID)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < GRID and 0 <= cc < GRID:
                    out.add(rr * GRID + cc)
    return out


def compute_ring(G: set[int], all_target: set[int],
                 min_ring: int = DEFAULT_MIN_RING,
                 max_radius: int = DEFAULT_MAX_RADIUS) -> tuple[list[int], int]:
    """Background ring R(G) = Dilate(G, r) \\ G \\ all_target.

    Expands r until |R| >= min_ring (or hits max_radius).  ``all_target`` is the
    union of *every* entity's target patches, so the ring never lands on another
    task object.
    """
    for radius in range(1, max_radius + 1):
        ring = dilate_indices(G, radius) - G - all_target
        if len(ring) >= min_ring:
            return sorted(ring), radius
    ring = dilate_indices(G, max_radius) - G - all_target
    return sorted(ring), max_radius


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class LocalReplaceCDInference(AuditedEntityCDInference):
    """Attention-CD variant whose negative branch replaces target features in
    feature space (projector output) instead of blocking attention."""

    _oracle_overlap: np.ndarray | None = None          # (256,) merged overlap (reference)
    _oracle_entity_overlaps: list[np.ndarray] | None = None  # per-entity (256,)
    _oracle_tau: float = 0.5
    _min_ring: int = DEFAULT_MIN_RING
    _ring_max_radius: int = DEFAULT_MAX_RADIUS

    def _build_replacement(self, V: torch.Tensor) -> tuple[list[int], torch.Tensor, dict]:
        """V [1, 256, D] float32 -> (selected, replacement[1,256,D], meta)."""
        overlaps = self._oracle_entity_overlaps
        if overlaps is None or len(overlaps) == 0:
            raise RuntimeError("Oracle entity overlaps not set before step")

        Gs = [set(int(i) for i in np.flatnonzero(o > self._oracle_tau)) for o in overlaps]
        all_target = set().union(*Gs) if Gs else set()
        selected = sorted(all_target)
        replacement = V.clone()
        V_np = V[0].detach().float().cpu().numpy()

        per_entity = []
        n_ring_total = 0
        for e, G in enumerate(Gs):
            rec: dict = {"entity_index": e, "n_target": len(G), "n_ring": 0,
                         "ring_radius": 0, "empty": True}
            entity_name = self._entities[e] if e < len(self._entities) else None
            if entity_name is not None:
                rec["entity_name"] = entity_name
            if not G:
                per_entity.append(rec)
                continue
            ring, radius = compute_ring(G, all_target, self._min_ring, self._ring_max_radius)
            if not ring:
                per_entity.append(rec)
                continue
            mu = V[0, ring].mean(dim=0)                       # [D] background mean
            mu_np = mu.detach().float().cpu().numpy()
            for i in G:
                replacement[0, i] = mu
            n_ring_total += len(ring)

            # metric D: norm ratio (record only, norm-matching NOT applied in V1)
            ring_norms = np.linalg.norm(V_np[ring], axis=1)
            rec["mu_norm_ratio"] = float(np.linalg.norm(mu_np) / (ring_norms.mean() + 1e-8))
            # metric C: semantic similarity before/after (best-effort, needs entity emb)
            if e < len(self._entity_emb):
                emb = self._entity_emb[e]
                s_before = float(np.mean([_cos(V_np[i], emb) for i in G]))
                s_after = _cos(mu_np, emb)
                rec["semantic_sim_before"] = s_before
                rec["semantic_sim_after"] = s_after
            rec.update({"n_ring": len(ring), "ring_radius": radius, "empty": False})
            per_entity.append(rec)

        meta = {
            "oracle_mode": "local_replace",
            "oracle_tau": self._oracle_tau,
            "min_ring": self._min_ring,
            "n_entities": len(Gs),
            "n_target_tokens": len(selected),
            "n_ring_total": n_ring_total,
            "per_entity": per_entity,
        }
        return selected, replacement, meta

    def _emit_clean(self, clean_scores, meta):
        token_ids = clean_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        positive = _action_logits(self, clean_scores)
        self._episode_logits.append({"positive": positive, "negative": positive})
        meta.update({
            "oracle_empty": True,
            "degenerate": True,
            "lambda": float(self.lambd),
            "clean_token_ids": token_ids.detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
        })
        self._episode_trace.append(meta)
        return raw_action, actions, meta

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("Positive branch did not produce 7 scores and projector features")

        V = positive_trace.before                                  # [1, 256, D] float32 cpu
        selected, replacement, meta = self._build_replacement(V)

        if not selected:
            return self._emit_clean(clean_scores, meta)

        with projector_intervention(self.vla, selected, replacement) as negative_trace:
            negative_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if negative_trace.before is None:
            raise RuntimeError("Negative branch projector hook was not invoked")
        feature_equal = torch.equal(V, negative_trace.before)
        if not feature_equal:
            raise RuntimeError("Positive and negative visual features are not bit-identical")
        if set(negative_trace.changed_indices) != set(selected):
            raise RuntimeError(
                f"Replacement indices {negative_trace.changed_indices} != selected {selected}"
            )
        # If no entity produced a non-empty ring, the negative branch is bit-identical
        # to the positive one -> CD is a no-op. Fall back to clean (degenerate).
        if meta["n_ring_total"] == 0:
            return self._emit_clean(clean_scores, meta)

        n_negative = negative_scores.shape[0]
        meta["negative_truncated"] = n_negative < 7
        if n_negative < 7:
            negative_scores = torch.cat([negative_scores, clean_scores[n_negative:]], dim=0)
        elif n_negative > 7:
            negative_scores = negative_scores[:7]

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1] - self.lambd * negative_scores[:-1]
        )
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite LocalReplace-CD logits")
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
            "degenerate": False,
        })
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        return raw_action, actions, meta
