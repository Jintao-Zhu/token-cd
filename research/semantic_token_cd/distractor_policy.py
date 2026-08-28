"""Audited policies for SIMPLER_DISTRACTOR_SEMANTIC_ENTITY_CD_PHASE0_V2."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from contrast_policies.openvla_contrast import OpenVLAContrastInference
from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.rollout_policy import SemanticEntityCDInference


ACTION_VOCAB_SIZE = 256


def _action_logits(policy, scores: torch.Tensor) -> np.ndarray:
    start = int(policy.vla.vocab_size) - ACTION_VOCAB_SIZE
    action = scores[:, start : start + ACTION_VOCAB_SIZE]
    if action.shape != (7, ACTION_VOCAB_SIZE):
        raise RuntimeError(f"Expected action logits [7,256], got {tuple(action.shape)}")
    return action.detach().to(dtype=torch.float16).cpu().numpy()


class AuditedVanillaInference(OpenVLAContrastInference):
    """Vanilla OpenVLA with action-vocabulary logits recorded per control step."""

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7:
            raise RuntimeError(f"Expected 7 clean action logits, got {clean_scores.shape[0]}")
        token_ids = clean_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        self._episode_logits.append({"positive": _action_logits(self, clean_scores)})
        meta = {"lambda": 0.0, "residual_norm": 0.0}
        self._episode_trace.append(meta)
        return raw_action, actions, meta


class AuditedEntityCDInference(SemanticEntityCDInference):
    """Semantic or size-preserving random-pseudogroup Entity-CD.

    The random arm first computes the semantic groups on its own current state,
    then randomly permutes token-to-group membership while preserving every
    cluster size. Selecting the same group IDs therefore matches the semantic
    reference exactly in group count, per-group size, total token count, lambda,
    and replacement operator. Only token membership loses semantic structure.
    """

    selection_mode: str
    _episode_seed: int
    _selector_step: int

    def reset(self, task_description: str, seed=None) -> None:
        super().reset(task_description, seed)
        if seed is not None:
            self._episode_seed = int(seed)
            self._selector_step = 0

    def _semantic_clusters(self, h: np.ndarray):
        hd = h.astype(np.float64)
        km = KMeans(
            n_clusters=self.kmeans_K,
            random_state=self.kmeans_seed,
            n_init=10,
        ).fit(hd)
        labels = km.labels_
        group_vectors = np.zeros((self.kmeans_K, hd.shape[1]), dtype=np.float64)
        for group_id in range(self.kmeans_K):
            members = np.flatnonzero(labels == group_id)
            if len(members):
                group_vectors[group_id] = hd[members].mean(axis=0)
        normalized = group_vectors / (
            np.linalg.norm(group_vectors, axis=1, keepdims=True) + 1e-8
        )
        selected_groups = []
        per_entity_score = []
        for embedding in self._entity_emb:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            scores = normalized @ embedding
            group_id = int(np.argmax(scores))
            per_entity_score.append(float(scores[group_id]))
            if group_id not in selected_groups:
                selected_groups.append(group_id)
        return labels, selected_groups, per_entity_score

    def _select(self, h: np.ndarray) -> tuple[list[int], dict]:
        labels, semantic_groups, per_entity_score = self._semantic_clusters(h)
        semantic_tokens = sorted(
            int(index)
            for group_id in semantic_groups
            for index in np.flatnonzero(labels == group_id)
        )
        group_sizes = [int(np.sum(labels == group_id)) for group_id in semantic_groups]
        selected = semantic_tokens
        random_seed = None
        random_overlap = None

        if self.selection_mode == "random_matched":
            random_seed = int(
                np.random.SeedSequence(
                    [self._episode_seed, self._selector_step, 0x5EEDCD]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            rng = np.random.default_rng(random_seed)
            randomized_labels = labels[rng.permutation(labels.size)]
            selected = sorted(
                int(index)
                for group_id in semantic_groups
                for index in np.flatnonzero(randomized_labels == group_id)
            )
            random_overlap = len(set(selected) & set(semantic_tokens)) / max(1, len(selected))
        elif self.selection_mode != "semantic":
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")

        if len(selected) != len(semantic_tokens):
            raise RuntimeError("Random control did not exactly match semantic token count")
        meta = {
            "selection_mode": self.selection_mode,
            "instruction": self._selector_instr,
            "selected_entities": list(self._entities),
            "selected_group_ids": semantic_groups,
            "group_score": per_entity_score,
            "language_score": float(max(per_entity_score)) if per_entity_score else 0.0,
            "selected_token_ids": selected,
            "num_tokens": len(selected),
            "num_groups": len(semantic_groups),
            "selected_group_sizes": group_sizes,
            "reference_semantic_token_ids": semantic_tokens,
            "random_seed": random_seed,
            "random_semantic_overlap": random_overlap,
            "kmeans_K": self.kmeans_K,
        }
        self._selector_step += 1
        return selected, meta

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7:
            raise RuntimeError(f"Expected 7 clean action logits, got {clean_scores.shape[0]}")
        h = trace.before[0].detach().float().cpu().numpy()
        selected, meta = self._select(h)
        if not selected:
            raise RuntimeError("Entity selector produced an empty negative branch")

        with projector_intervention(self.vla, selected, self.replacement_mean):
            negative_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        n_negative = negative_scores.shape[0]
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
            raise FloatingPointError("Non-finite Entity-CD logits")
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
            "degenerate": False,
        })
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        return raw_action, actions, meta
