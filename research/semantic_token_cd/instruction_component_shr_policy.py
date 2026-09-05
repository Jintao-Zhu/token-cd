"""Instruction-aware Component Selection SHR (IC-SHR) v1.

The OpenVLA backbone, semantic KMeans-K8 selector, harmonic reconstruction,
guided negative branch, and lambda=0.5 CD rule are inherited from canonical
SHR.  The only method change is an atomic connected-component filter between
semantic selection and harmonic reconstruction.
"""
from __future__ import annotations

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import _action_logits
from research.semantic_token_cd.global_merge_policy import (
    guided_forward_scores,
    projector_merge_intervention,
)
from research.semantic_token_cd.st_shr_policy import GRID, STSHRCDInference, harmonic_reconstruct


SEMANTIC_WEIGHT = 0.5
CENTER_WEIGHT = 0.3
SIZE_WEIGHT = 0.2
CENTER_SIGMA = 4.0
IMAGE_CENTER = (8.0, 8.0)
LAMBDA = 0.5


def connected_components_4(token_ids) -> list[np.ndarray]:
    """Return complete 4-neighbor components on the fixed 16x16 token grid."""
    remaining = set(int(i) for i in token_ids)
    if any(i < 0 or i >= GRID * GRID for i in remaining):
        raise ValueError("visual token IDs must lie in [0, 255]")
    components: list[np.ndarray] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack = [root]
        component = []
        while stack:
            token = stack.pop()
            component.append(token)
            row, col = divmod(token, GRID)
            neighbors = []
            if row > 0:
                neighbors.append(token - GRID)
            if row + 1 < GRID:
                neighbors.append(token + GRID)
            if col > 0:
                neighbors.append(token - 1)
            if col + 1 < GRID:
                neighbors.append(token + 1)
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def select_instruction_components(
    features: np.ndarray,
    group_tokens,
    instruction_embedding: np.ndarray,
    n_keep: int,
    sigma: float = CENTER_SIGMA,
) -> tuple[np.ndarray, dict]:
    """Score whole components and return their atomic top-k union."""
    original = np.asarray(sorted(set(int(i) for i in group_tokens)), dtype=np.int64)
    if original.size == 0:
        raise ValueError("IC-SHR received an empty semantic group")
    if n_keep < 1:
        raise ValueError("n_keep must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    components = connected_components_4(original)
    inst = np.asarray(instruction_embedding, dtype=np.float64)
    inst_norm = float(np.linalg.norm(inst))
    if inst_norm == 0 or not np.isfinite(inst_norm):
        raise ValueError("instruction embedding must be finite and non-zero")

    diagnostics = []
    for index, component in enumerate(components):
        vectors = features[component].astype(np.float64, copy=False)
        denom = np.linalg.norm(vectors, axis=1) * inst_norm + 1e-8
        semantic = float(np.mean((vectors @ inst) / denom))
        rows = component // GRID
        cols = component % GRID
        centroid = (float(rows.mean()), float(cols.mean()))
        distance = float(np.hypot(centroid[0] - IMAGE_CENTER[0], centroid[1] - IMAGE_CENTER[1]))
        center = float(np.exp(-distance / sigma))
        size = float(component.size / original.size)
        score = SEMANTIC_WEIGHT * semantic + CENTER_WEIGHT * center + SIZE_WEIGHT * size
        diagnostics.append({
            "component_index": index,
            "token_ids": component.tolist(),
            "size": int(component.size),
            "centroid": list(centroid),
            "center_distance": distance,
            "semantic_score": semantic,
            "center_score": center,
            "size_score": size,
            "score": score,
        })

    # One component is a literal no-op. Ties are deterministic without splitting.
    keep_count = min(n_keep, len(components))
    order = sorted(
        range(len(components)),
        key=lambda i: (-diagnostics[i]["score"], -diagnostics[i]["size"], diagnostics[i]["token_ids"][0]),
    )
    selected_indices = order[:keep_count]
    selected = np.asarray(
        sorted(int(token) for i in selected_indices for token in components[i]),
        dtype=np.int64,
    )
    selected_set = set(map(int, selected))
    for item in diagnostics:
        item["selected"] = item["component_index"] in selected_indices
    meta = {
        "original_group_tokens": original.tolist(),
        "selected_components": [components[i].tolist() for i in selected_indices],
        "all_components": diagnostics,
        "selected_component_indices": selected_indices,
        "num_components": len(components),
        "selected_num_components": len(selected_indices),
        "filtered_num_components": len(components) - len(selected_indices),
        "mask_tokens_before": int(original.size),
        "mask_tokens_after": int(selected.size),
        "removed_token_ids": sorted(set(map(int, original)) - selected_set),
        "removed_token_fraction": float(1.0 - selected.size / original.size),
        "component_atomic": all(set(map(int, c)).issubset(selected_set) or set(map(int, c)).isdisjoint(selected_set)
                                for c in components),
    }
    return selected, meta


class InstructionComponentSHRInference(STSHRCDInference):
    """Canonical beta=0 SHR with instruction-aware atomic component selection."""

    def _forward_scores(self, inputs, unnorm_key, **kwargs):
        # Fixed-width robot actions must not terminate when action token ID 2
        # aliases the language model's EOS ID. [] preserves the EOS logit.
        kwargs["eos_token_id"] = []
        scores = super()._forward_scores(inputs, unnorm_key, **kwargs)
        if scores.shape[0] != 7:
            raise RuntimeError(f"IC-SHR expected 7 action scores, got {scores.shape[0]}")
        return scores

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if positive_trace.before is None:
            raise RuntimeError("IC-SHR positive branch did not expose projector features")
        visual = positive_trace.before
        features = visual[0].numpy().astype(np.float32)
        labels, _, per_entity_score = self._semantic_clusters(features)
        entity_groups = self._entity_groups(features, labels)
        original = np.asarray(sorted({
            int(i) for group in entity_groups for i in np.flatnonzero(labels == group)
        }), dtype=np.int64)
        n_entities = len(self._entities)
        n_keep = 1 if n_entities <= 1 else 2
        instruction_embedding = self._embed_phrase(self._selector_instr)
        selected, component_meta = select_instruction_components(
            features, original, instruction_embedding, n_keep=n_keep
        )

        negative_features = features.copy()
        negative_features[selected] = harmonic_reconstruct(features, selected)
        untouched = np.setdiff1d(np.arange(features.shape[0]), selected, assume_unique=True)
        non_target_equal = bool(np.array_equal(features[untouched], negative_features[untouched]))

        clean_token_ids = clean_scores.argmax(dim=-1)
        with projector_merge_intervention(
            self.vla, torch.from_numpy(negative_features).unsqueeze(0)
        ) as negative_trace:
            negative_scores = guided_forward_scores(
                self.vla, inputs, clean_token_ids, visual.shape[1]
            )
        if negative_trace["before"] is None or not torch.equal(visual, negative_trace["before"]):
            raise RuntimeError("IC-SHR negative projector input differs from clean features")

        final_scores = clean_scores.clone()
        final_scores[:-1] = 1.5 * clean_scores[:-1] - 0.5 * negative_scores[:-1]
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("non-finite IC-SHR logits")
        final_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(final_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        residual = (
            torch.log_softmax(torch.from_numpy(positive).float(), dim=-1)
            - torch.log_softmax(torch.from_numpy(negative).float(), dim=-1)
        ).numpy()
        meta = {
            "selection_mode": "instruction_component",
            "selected_entities": list(self._entities),
            "target_entity_count": n_entities,
            "target_component_count": n_keep,
            "selected_group_ids": entity_groups,
            "per_entity_score": per_entity_score,
            "selected_token_ids": selected.tolist(),
            "num_tokens": int(selected.size),
            "kmeans_K": int(self.kmeans_K),
            "kmeans_seed": int(self.kmeans_seed),
            "beta": 0.0,
            "lambda": LAMBDA,
            "component_weights": {
                "semantic": SEMANTIC_WEIGHT, "center": CENTER_WEIGHT, "size": SIZE_WEIGHT,
            },
            "center_sigma": CENTER_SIGMA,
            "image_center": list(IMAGE_CENTER),
            "guided_prefix": True,
            "feature_equal": True,
            "non_target_bit_identical": non_target_equal,
            "reconstruction_finite": bool(np.isfinite(negative_features).all()),
            "residual_norm": float(np.linalg.norm(residual)),
            "positive_token_ids": clean_token_ids.detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": final_ids.detach().cpu().tolist(),
            "first_action": clean_token_ids.detach().cpu().tolist(),
            "guided_action": final_ids.detach().cpu().tolist(),
            **component_meta,
        }
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        self._selector_step += 1
        return raw_action, actions, meta
