"""SEMANTIC_ENTITY_CD rollout policy — token-level semantic-entity contrastive decoding.

Implements the Phase-2B language-selected semantic entity negative branch (K=8
kmeans grouping on projector features h_i + per-entity top-1 cos(v_i, e) group
match) inside the closed-loop SIMPLER rollout. Mirrors FrozenTokenPCDInference
(research/token_pcd_bridge) but replaces the SAM-object-mask selector with the
autonomous semantic selector (no SAM / no GT object mask / no CLIP / no VLM).

CD (contrastive decoding), matching the PCD convention:
    final_scores[:-1] = (1 + λ)·clean_scores[:-1] - λ·masked_scores[:-1]
    λ ∈ {0.25, 0.5}  (last action dim "terminate/gripper" is left untouched).

Per step:
  1. clean forward + capture projector output h_i [256,4096] via projector_intervention
     (the hook fires once inside vla.generate; h_i is vision-only and thus identical
     to the Phase-2B teacher-forced h_i for the same resized image).
  2. kmeans(K=8, random_state=0, n_init=10) on h_i -> group labels.
  3. group mean v_i; per entity e (mean-pooled embed_tokens) pick top-1 group by
     cos(v_i, e); entity_set = deduped union of those groups.
  4. masked forward on the union of patches in the selected groups via
     projector_intervention(selected_indices, replacement_mean).
  5. contrast, argmax, decode, postprocess.

Run environment (from research/token_pcd_bridge):
  PYTHONPATH = <pcd_root>/source/PCD  +  repo root  +  task1/shim_site
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from contrast_policies.openvla_contrast import OpenVLAContrastInference
from research.ar_token_counterfactual.intervention import projector_intervention

PREPOSITIONS = {"into", "in", "on", "onto", "near", "to", "next", "from", "under", "over",
                "behind", "beside", "above", "below", "inside", "at", "with"}
DETERMINERS = {"the", "a", "an"}


def extract_relation(instruction: str) -> dict[str, str]:
    """Decompose instruction -> {action, source, relation, target, ...} (Phase-2A verified)."""
    words = instruction.lower().split()
    verb = words[0]
    rest = words[1:]
    prep_pos = [i for i, w in enumerate(rest) if w in PREPOSITIONS]
    if not prep_pos:
        obj = " ".join(w for w in rest if w not in DETERMINERS)
        return {"action": verb, "source": obj, "relation": "", "target": obj,
                "rel_phrase": obj, "noun_phrase": obj}
    first, last = prep_pos[0], prep_pos[-1]
    src = " ".join(w for w in rest[:first] if w not in DETERMINERS)
    rel = rest[last]
    tgt = " ".join(w for w in rest[last + 1:] if w not in DETERMINERS)
    rel_phrase = " ".join(w for w in rest if w not in DETERMINERS)
    noun_phrase = " ".join(w for w in rest if w not in DETERMINERS and w not in PREPOSITIONS)
    return {"action": verb, "source": src, "relation": rel, "target": tgt,
            "rel_phrase": rel_phrase, "noun_phrase": noun_phrase}


def extract_entities(instruction: str) -> list[str]:
    """Rule-based entity list = deduped [source, target] (1 or 2 entities, no LLM)."""
    r = extract_relation(instruction)
    ents: list[str] = []
    for e in (r["source"], r["target"]):
        if e and e not in ents:
            ents.append(e)
    return ents


class SemanticEntityCDInference(OpenVLAContrastInference):
    """OpenVLA-7B with a token-level semantic-entity negative branch (Phase-2B selector).

    Expected to be built from a pre-loaded OpenVLAInference via copy.copy + __class__
    reassignment (shares the frozen 7B model/processor across arms) — see
    research/token_pcd_bridge/runner.py for the identical pattern.
    """

    def __init__(self, replacement_mean: torch.Tensor, lambd: float,
                 kmeans_K: int = 8, kmeans_seed: int = 0, **kwargs):
        super().__init__(alpha=lambd, **kwargs)
        self.replacement_mean = replacement_mean
        self.lambd = lambd
        self.kmeans_K = kmeans_K
        self.kmeans_seed = kmeans_seed
        self._selector_instr: str | None = None
        self._entities: list[str] = []
        self._entity_emb: list[np.ndarray] = []
        self._emb_cache: dict[str, np.ndarray] = {}
        self._episode_trace: list[dict] = []

    def reset(self, task_description: str, seed=None) -> None:
        super().reset(task_description, seed)
        # Recompute entities only when the instruction changes (also fires on
        # subtask advancement via process_inputs). Do NOT clear _episode_trace here
        # — the runner owns the per-episode trace buffer.
        if task_description != self._selector_instr:
            self._entities = extract_entities(task_description)
            self._entity_emb = [self._embed_phrase(e) for e in self._entities]
            self._selector_instr = task_description

    def _embed_phrase(self, text: str) -> np.ndarray:
        if text in self._emb_cache:
            return self._emb_cache[text]
        tok = self.processor.tokenizer
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tok(text, add_special_tokens=True)["input_ids"]
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.vla.device)
        emb = self.vla.language_model.model.embed_tokens(ids_t)
        e = emb[0].mean(dim=0).detach().float().cpu().numpy()
        self._emb_cache[text] = e
        return e

    def _select(self, h: np.ndarray) -> tuple[list[int], dict]:
        """h [256,4096] float32 -> (sorted selected patch indices, selector meta)."""
        hd = h.astype(np.float64)
        km = KMeans(n_clusters=self.kmeans_K, random_state=self.kmeans_seed, n_init=10).fit(hd)
        labels = km.labels_
        K = self.kmeans_K
        v = np.zeros((K, hd.shape[1]), dtype=np.float64)
        for k in range(K):
            idx = np.flatnonzero(labels == k)
            v[k] = hd[idx].mean(axis=0) if len(idx) else 0.0
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

        sel_groups: list[int] = []
        per_entity_score: list[float] = []
        for e in self._entity_emb:
            en = e / (np.linalg.norm(e) + 1e-8)
            c = vn @ en
            g = int(np.argmax(c))
            per_entity_score.append(float(c[g]))
            if g not in sel_groups:
                sel_groups.append(g)

        selected = sorted({int(x) for g in sel_groups for x in np.flatnonzero(labels == g)})
        meta = {
            "selected_groups": sel_groups,
            "selected_token_ids": selected,
            "entity_names": list(self._entities),
            "per_entity_score": per_entity_score,
            "language_score": float(max(per_entity_score)) if per_entity_score else 0.0,
            "kmeans_K": K,
        }
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
            # Degenerate: no entity group matched any token. Fall back to clean.
            token_ids = clean_scores.argmax(dim=-1)
            meta["degenerate"] = True
            meta["negative_truncated"] = False
            negative_token_ids = None
        else:
            with projector_intervention(self.vla, selected, self.replacement_mean) as trace2:
                negative_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
            n_neg = negative_scores.shape[0]
            meta["negative_truncated"] = n_neg < 7
            if n_neg < 7:
                # Degenerate negative branch: the masked model emitted an early EOS
                # token and stopped before decoding all 7 action dims. Fill the
                # missing dims with the clean scores so no contrastive signal is
                # applied there (conservative fallback — CD is inactive on those dims).
                negative_scores = torch.cat([negative_scores, clean_scores[n_neg:]], dim=0)
            elif n_neg > 7:
                negative_scores = negative_scores[:7]
            final_scores = clean_scores.clone()
            final_scores[:-1] = (1 + self.lambd) * clean_scores[:-1] - self.lambd * negative_scores[:-1]
            if not torch.isfinite(final_scores).all():
                raise FloatingPointError("Non-finite entity-CD logits")
            token_ids = final_scores.argmax(dim=-1)
            meta["degenerate"] = False
            negative_token_ids = negative_scores.argmax(-1).detach().cpu().tolist()

        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        meta["clean_token_ids"] = clean_scores.argmax(-1).detach().cpu().tolist()
        meta["final_token_ids"] = token_ids.detach().cpu().tolist()
        meta["negative_token_ids"] = negative_token_ids
        meta["n_selected"] = len(selected)
        meta["lambd"] = self.lambd
        self._episode_trace.append(meta)
        return raw_action, actions, meta
