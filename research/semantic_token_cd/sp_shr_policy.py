"""Structure-preserving spatial harmonic reconstruction policies."""
from __future__ import annotations

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import _action_logits
from research.semantic_token_cd.global_merge_policy import (
    guided_forward_scores,
    projector_merge_intervention,
)
from research.semantic_token_cd.st_shr_policy import EPS, GRID, STSHRCDInference, harmonic_reconstruct


BOUNDARY = "boundary"
PARTIAL50 = "partial50"
PARTIAL_SALT = 0x50534852


def boundary_and_interior(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-token 4-neighbor boundary and the eroded interior."""
    tokens = np.asarray(sorted(set(int(i) for i in region)), dtype=np.int64)
    token_set = set(map(int, tokens))
    boundary = []
    interior = []
    for token in tokens:
        r, c = divmod(int(token), GRID)
        if r == 0 or r == GRID - 1 or c == 0 or c == GRID - 1:
            boundary.append(int(token))
            continue
        neighbors = (token - GRID, token + GRID, token - 1, token + 1)
        (interior if all(int(n) in token_set for n in neighbors) else boundary).append(int(token))
    return np.asarray(boundary, dtype=np.int64), np.asarray(interior, dtype=np.int64)


def deterministic_partial_region(
    region: np.ndarray,
    fraction: float,
    task_id: int,
    episode_seed: int,
    selector_step: int,
    entity_index: int,
) -> tuple[np.ndarray, int]:
    """Choose a reproducible uniform subset, rounding half upward."""
    tokens = np.asarray(sorted(set(int(i) for i in region)), dtype=np.int64)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(np.ceil(tokens.size * fraction)))
    random_seed = int(np.random.SeedSequence([
        int(task_id), int(episode_seed), int(selector_step), int(entity_index), PARTIAL_SALT,
    ]).generate_state(1, dtype=np.uint32)[0])
    rng = np.random.default_rng(random_seed)
    selected = np.sort(rng.choice(tokens, size=count, replace=False)).astype(np.int64)
    return selected, random_seed


class StructurePreservingSHRCDInference(STSHRCDInference):
    """Plain SHR with either clean semantic boundaries or a 50% target subset."""

    sp_variant: str = BOUNDARY
    partial_fraction: float = 0.5
    _task_id: int = 0

    def _variant_region(
        self, region: np.ndarray, entity_index: int
    ) -> tuple[np.ndarray, dict]:
        if self.sp_variant == BOUNDARY:
            boundary, target = boundary_and_interior(region)
            return target, {
                "boundary_token_ids": boundary.tolist(),
                "num_boundary_tokens": int(boundary.size),
                "partial_random_seed": None,
            }
        if self.sp_variant == PARTIAL50:
            target, random_seed = deterministic_partial_region(
                region,
                self.partial_fraction,
                self._task_id,
                self._episode_seed,
                self._selector_step,
                entity_index,
            )
            preserved = np.setdiff1d(region, target, assume_unique=True)
            return target, {
                "boundary_token_ids": [],
                "num_boundary_tokens": 0,
                "partial_random_seed": random_seed,
                "partial_preserved_token_ids": preserved.tolist(),
            }
        raise ValueError(f"unknown SP-SHR variant: {self.sp_variant}")

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("positive branch did not produce projector features and 7 scores")
        visual = positive_trace.before
        features = visual[0].numpy().astype(np.float32)
        labels, _, per_entity_score = self._semantic_clusters(features)
        entity_groups = self._entity_groups(features, labels)

        negative_features = features.copy()
        semantic_claimed: set[int] = set()
        reconstruction_claimed: set[int] = set()
        entity_meta = []
        for entity_index, (entity, group) in enumerate(zip(self._entities, entity_groups)):
            region = np.asarray([
                int(i) for i in np.flatnonzero(labels == group) if int(i) not in semantic_claimed
            ], dtype=np.int64)
            semantic_claimed.update(map(int, region))
            if region.size == 0:
                entity_meta.append({
                    "entity": entity,
                    "group": group,
                    "num_semantic_tokens": 0,
                    "num_reconstructed_tokens": 0,
                    "fallback_reason": "dedup_empty",
                })
                continue
            target, variant_meta = self._variant_region(region, entity_index)
            if target.size:
                solved = harmonic_reconstruct(features, target)
                negative_features[target] = solved
                reconstruction_claimed.update(map(int, target))
            entity_meta.append({
                "entity": entity,
                "group": group,
                "semantic_token_ids": region.tolist(),
                "reconstructed_token_ids": target.tolist(),
                "num_semantic_tokens": int(region.size),
                "num_reconstructed_tokens": int(target.size),
                "reconstruction_fraction": float(target.size / region.size),
                "fallback_reason": "empty_eroded_interior" if target.size == 0 else None,
                **variant_meta,
            })
        if not semantic_claimed:
            raise RuntimeError("SP-SHR selector produced an empty semantic region")

        visual_negative = torch.from_numpy(negative_features).unsqueeze(0)
        clean_token_ids = clean_scores.argmax(dim=-1)
        with projector_merge_intervention(self.vla, visual_negative) as negative_trace:
            negative_scores = guided_forward_scores(
                self.vla, inputs, clean_token_ids, visual.shape[1]
            )
        if negative_trace["before"] is None or not torch.equal(visual, negative_trace["before"]):
            raise RuntimeError("negative projector input differs from clean features")

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1] - self.lambd * negative_scores[:-1]
        )
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("non-finite SP-SHR logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        residual = (
            torch.log_softmax(torch.from_numpy(positive).float(), dim=-1)
            - torch.log_softmax(torch.from_numpy(negative).float(), dim=-1)
        ).numpy()
        untouched = np.asarray(
            sorted(set(range(features.shape[0])) - reconstruction_claimed), dtype=np.int64
        )
        non_target_equal = bool(np.array_equal(features[untouched], negative_features[untouched]))
        meta = {
            "selection_mode": "semantic",
            "sp_variant": self.sp_variant,
            "partial_fraction": self.partial_fraction if self.sp_variant == PARTIAL50 else None,
            "selected_entities": list(self._entities),
            "selected_group_ids": entity_groups,
            "per_entity_score": per_entity_score,
            "reference_semantic_token_ids": sorted(semantic_claimed),
            "num_semantic_tokens": len(semantic_claimed),
            "selected_token_ids": sorted(reconstruction_claimed),
            "num_tokens": len(reconstruction_claimed),
            "preserved_semantic_token_ids": sorted(semantic_claimed - reconstruction_claimed),
            "num_preserved_semantic_tokens": len(semantic_claimed - reconstruction_claimed),
            "entity_regions": entity_meta,
            "kmeans_K": int(self.kmeans_K),
            "beta": 0.0,
            "lambda": float(self.lambd),
            "residual_norm": float(np.linalg.norm(residual)),
            "guided_prefix": True,
            "feature_equal": True,
            "non_target_bit_identical": non_target_equal,
            "reconstruction_finite": bool(np.isfinite(negative_features).all()),
            "positive_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
        }
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        self._selector_step += 1
        return raw_action, actions, meta
