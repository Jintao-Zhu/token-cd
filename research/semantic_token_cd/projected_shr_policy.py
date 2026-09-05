"""Projected-SHR: project the unchanged SHR residual in action-logit space."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import (
    ACTION_VOCAB_SIZE,
    _action_logits,
)
from research.semantic_token_cd.global_merge_policy import (
    guided_forward_scores,
    projector_merge_intervention,
)
from research.semantic_token_cd.st_shr_policy import (
    EPS,
    STSHRCDInference,
    harmonic_reconstruct,
)


def projected_shr_action_logits(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    lambd: float = 0.5,
    eta: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply Projected-SHR independently to each 256-way action dimension."""
    if positive.shape != negative.shape:
        raise ValueError("positive and negative action logits must have identical shapes")
    if positive.ndim != 2 or positive.shape[-1] != ACTION_VOCAB_SIZE:
        raise ValueError(f"expected [A,{ACTION_VOCAB_SIZE}] action logits")
    if lambd < 0 or not 0.0 <= eta <= 1.0 or eps <= 0:
        raise ValueError("require lambda >= 0, eta in [0,1], and eps > 0")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise FloatingPointError("non-finite Projected-SHR input logits")

    p = positive.float()
    n = negative.float()
    pc = p - p.mean(dim=-1, keepdim=True)
    nc = n - n.mean(dim=-1, keepdim=True)
    residual = pc - nc
    pc_norm_sq = pc.square().sum(dim=-1, keepdim=True)
    alpha = (residual * pc).sum(dim=-1, keepdim=True) / (pc_norm_sq + eps)
    parallel = alpha * pc
    orthogonal = residual - parallel
    guidance = orthogonal + eta * parallel
    projected = p + lambd * guidance
    original_shr = p + lambd * residual

    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    pc_norm = torch.linalg.vector_norm(pc, dim=-1)
    parallel_norm = torch.linalg.vector_norm(parallel, dim=-1)
    orthogonal_norm = torch.linalg.vector_norm(orthogonal, dim=-1)
    residual_dot_pc = (residual * pc).sum(dim=-1)
    orthogonal_dot_pc = (orthogonal * pc).sum(dim=-1)
    scale = 1.0 + lambd * alpha.squeeze(-1)
    lambda_eff = torch.where(
        scale.abs() > eps,
        torch.full_like(scale, float(lambd)) / scale,
        torch.full_like(scale, torch.nan),
    )
    diagnostics = {
        "alpha": alpha.squeeze(-1),
        "residual": residual,
        "parallel": parallel,
        "orthogonal": orthogonal,
        "original_shr": original_shr,
        "residual_norm": residual_norm,
        "parallel_ratio": parallel_norm / (residual_norm + eps),
        "orthogonal_norm": orthogonal_norm,
        "residual_positive_cosine": residual_dot_pc / (residual_norm * pc_norm + eps),
        "orthogonal_positive_relative_dot": orthogonal_dot_pc.abs()
        / (orthogonal_norm * pc_norm + eps),
        "lambda_eff": lambda_eff,
    }
    if not torch.isfinite(projected).all():
        raise FloatingPointError("non-finite Projected-SHR output logits")
    return projected.to(dtype=positive.dtype), diagnostics


class ProjectedSHRCDInference(STSHRCDInference):
    """Original beta=0 SHR negative branch plus action-vocabulary projection."""

    projection_eta: float = 0.0
    projection_eps: float = 1e-8

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            positive_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if positive_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("positive branch did not produce projector features and 7 scores")

        visual = positive_trace.before
        features = visual[0].numpy().astype(np.float32)
        labels, _, per_entity_score = self._semantic_clusters(features)
        entity_groups = self._entity_groups(features, labels)

        negative_features = features.copy()
        claimed: set[int] = set()
        entity_meta = []
        for entity, group in zip(self._entities, entity_groups):
            indices = np.asarray(
                [int(i) for i in np.flatnonzero(labels == group) if int(i) not in claimed],
                dtype=np.int64,
            )
            if indices.size == 0:
                entity_meta.append({
                    "entity": entity,
                    "group": group,
                    "num_tokens": 0,
                    "fallback_reason": "dedup_empty",
                })
                continue
            claimed.update(map(int, indices))
            negative_features[indices] = harmonic_reconstruct(features, indices, beta=0.0)
            entity_meta.append({
                "entity": entity,
                "group": group,
                "num_tokens": int(indices.size),
                "fallback_reason": None,
            })
        if not claimed:
            raise RuntimeError("Projected-SHR selector produced an empty region")

        clean_token_ids = positive_scores.argmax(dim=-1)
        visual_negative = torch.from_numpy(negative_features).unsqueeze(0)
        with projector_merge_intervention(self.vla, visual_negative) as negative_trace:
            negative_scores = guided_forward_scores(
                self.vla, inputs, clean_token_ids, visual.shape[1]
            )
        if negative_trace["before"] is None or not torch.equal(visual, negative_trace["before"]):
            raise RuntimeError("negative projector input differs from clean features")

        action_start = int(self.vla.vocab_size) - ACTION_VOCAB_SIZE
        positive_action = positive_scores[:, action_start : action_start + ACTION_VOCAB_SIZE]
        negative_action = negative_scores[:, action_start : action_start + ACTION_VOCAB_SIZE]
        projected_action, diag = projected_shr_action_logits(
            positive_action[:-1],
            negative_action[:-1],
            lambd=self.lambd,
            eta=self.projection_eta,
            eps=self.projection_eps,
        )
        final_scores = positive_scores.clone()
        final_scores[:-1, action_start : action_start + ACTION_VOCAB_SIZE] = projected_action
        token_ids = clean_token_ids.clone()
        token_ids[:-1] = projected_action.argmax(dim=-1) + action_start
        token_ids[-1] = clean_token_ids[-1]
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        original_indices = diag["original_shr"].argmax(dim=-1)
        projected_indices = projected_action.argmax(dim=-1)
        same = original_indices.eq(projected_indices)
        finite_diag = all(torch.isfinite(value).all() for key, value in diag.items() if key != "lambda_eff")
        meta: dict[str, Any] = {
            "selection_mode": "semantic",
            "negative_branch": "unchanged_beta0_spatial_harmonic_reconstruction",
            "selected_entities": list(self._entities),
            "selected_group_ids": entity_groups,
            "per_entity_score": per_entity_score,
            "selected_token_ids": sorted(claimed),
            "num_tokens": len(claimed),
            "entity_regions": entity_meta,
            "kmeans_K": int(self.kmeans_K),
            "beta": 0.0,
            "lambda": float(self.lambd),
            "projection_eta": float(self.projection_eta),
            "projection_eps": float(self.projection_eps),
            "guided_prefix": True,
            "guided_prefix_type": "clean_greedy_matching_canonical_shr",
            "feature_equal": True,
            "reconstruction_finite": bool(np.isfinite(negative_features).all()),
            "projection_finite": bool(finite_diag),
            "positive_token_ids": clean_token_ids.detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "residual_positive_cosine": diag["residual_positive_cosine"].detach().cpu().tolist(),
            "parallel_ratio": diag["parallel_ratio"].detach().cpu().tolist(),
            "projection_alpha": diag["alpha"].detach().cpu().tolist(),
            "orthogonal_norm": diag["orthogonal_norm"].detach().cpu().tolist(),
            "residual_norm": diag["residual_norm"].detach().cpu().tolist(),
            "lambda_eff": diag["lambda_eff"].detach().cpu().tolist(),
            "orthogonal_positive_relative_dot_max": float(
                diag["orthogonal_positive_relative_dot"].max().item()
            ),
            "original_shr_action_token_indices": original_indices.detach().cpu().tolist(),
            "projected_shr_action_token_indices": projected_indices.detach().cpu().tolist(),
            "projected_matches_original_shr": same.detach().cpu().tolist(),
            "projected_changes_original_shr_count": int((~same).sum().item()),
            "gripper_positive_unchanged": bool(token_ids[-1].eq(clean_token_ids[-1]).item()),
        }
        positive_np = _action_logits(self, positive_scores)
        negative_np = _action_logits(self, negative_scores)
        original_np = torch.cat([diag["original_shr"], positive_action[-1:]], dim=0)
        projected_np = torch.cat([projected_action.float(), positive_action[-1:].float()], dim=0)
        self._episode_logits.append({
            "positive": positive_np,
            "negative": negative_np,
            "original_shr": original_np.detach().to(torch.float16).cpu().numpy(),
            "projected": projected_np.detach().to(torch.float16).cpu().numpy(),
        })
        self._episode_trace.append(meta)
        self._selector_step += 1
        return raw_action, actions, meta
