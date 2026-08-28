"""LIBERO semantic-entity CD primitives (HF-path OpenVLAForActionPrediction).

Mirrors the SIMPLER Phase-2B semantic-entity selector
(research/semantic_token_cd/rollout_policy.py) but runs on the LIBERO HF model
path (research/ar_token_counterfactual/libero_runtime.load_policy). The selector
is identical (K=8 kmeans on projector h_i + per-entity top-1 cos match); only the
model/processor/image-pipeline differ.

Image pipeline (LIBERO finetuned augmented checkpoint):
  agentview 256x256 -> rotate180 -> resize224 (lanczos3 + JPEG roundtrip)
                    -> center-crop 0.9 -> 224x224 PIL -> vision backbone -> 16x16
The GT oracle mask must traverse the SAME geometric transform to align with the
256 visual tokens.

Reuse:
  - research.ar_token_counterfactual.intervention.projector_intervention /
    teacher_forced_forward / clean_action_token_ids
  - research.ar_token_counterfactual.libero_runtime.prepare_agentview / build_prompt
  - research.cw_lpcd.core.action_token_slice / log_softmax_residual
  - research.cw_lpcd.metrics.per_position_cosine
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from research.cw_lpcd.core import (
    N_ACTION_TOKENS,
    N_VISUAL,
    PATCH_GRID,
    action_token_slice,
    log_softmax_residual,
)
from research.ar_token_counterfactual.intervention import (
    clean_action_token_ids,
    projector_intervention,
    teacher_forced_forward,
)
from research.ar_token_counterfactual.libero_runtime import (
    build_prompt,
    center_crop_for_aug,
    prepare_agentview,
    resize_libero_image,
)

PREPOSITIONS = {"into", "in", "on", "onto", "near", "to", "next", "from", "under", "over",
                "behind", "beside", "above", "below", "inside", "at", "with"}
DETERMINERS = {"the", "a", "an"}
LIBERO_VERBS = {"pick", "place", "put", "move", "stack", "grab", "open", "close",
                "push", "turn", "lift", "hold"}
LIBERO_PARTICLES = {"up", "down"}
LIBERO_PRONOUNS = {"it", "them", "this", "that", "itself"}


def extract_entities_libero(instruction: str) -> list[str]:
    """LIBERO-Object template-aware entity extraction (rule-based, no LLM/CLIP).

    "pick up the alphabet soup and place it in the basket" -> ["alphabet soup", "basket"].
    Split on "and", then per clause drop determiners/prepositions/verbs/particles/pronouns.
    """
    clauses = instruction.lower().split(" and ")
    ents: list[str] = []
    for clause in clauses:
        words = [w for w in clause.split()
                 if w not in DETERMINERS and w not in PREPOSITIONS
                 and w not in LIBERO_VERBS and w not in LIBERO_PARTICLES
                 and w not in LIBERO_PRONOUNS]
        e = " ".join(words)
        if e and e not in ents:
            ents.append(e)
    return ents


def embed_phrase(model, tokenizer, text: str) -> np.ndarray:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    ids_t = torch.tensor([ids], dtype=torch.long, device=model.device)
    emb = model.language_model.model.embed_tokens(ids_t)
    return emb[0].mean(dim=0).detach().float().cpu().numpy()


def _processor_inputs(processor, model, image_pil: Image.Image, instruction: str):
    return processor(build_prompt(instruction), image_pil).to(model.device, dtype=torch.bfloat16)


def forward_logits(model, processor, image_pil, instruction, clean_ids, selected=(), mean=None):
    """Teacher-forced [7, vocab] action logits, optionally masking selected visual tokens."""
    inputs = _processor_inputs(processor, model, image_pil, instruction)
    out = teacher_forced_forward(model, inputs, clean_ids,
                                 selected_indices=list(selected), replacement_mean=mean)
    return out.logits[0]  # [7, vocab]


def extract_h(model, processor, image_pil, instruction, clean_ids) -> np.ndarray:
    """One clean teacher-forced forward -> projector h_i [256, 4096]."""
    inputs = _processor_inputs(processor, model, image_pil, instruction)
    out = teacher_forced_forward(model, inputs, clean_ids)
    if out.trace.before is None or out.trace.before.shape[1] != N_VISUAL:
        raise RuntimeError(f"Expected [1,256,D] projector output, got {out.trace.before.shape}")
    return out.trace.before[0].detach().float().cpu().numpy()


def _mask_to_224(mask: np.ndarray) -> np.ndarray:
    """Apply the LIBERO image geometric transform (rotate180 -> resize224 -> crop0.9)
    to a single-channel 0/1 mask, returning a 224x224 float mask."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    m = mask[::-1, ::-1].astype(np.float32)  # rotate 180 (same as prepare_agentview)
    m = cv2.resize(m, (224, 224), interpolation=cv2.INTER_AREA)
    # center_crop_for_aug: crop central sqrt(0.9) fraction, resize to 224x224.
    side = int(round(224 * (0.9 ** 0.5)))
    top = (224 - side) // 2
    m = cv2.resize(m[top:top + side, top:top + side], (224, 224), interpolation=cv2.INTER_AREA)
    return m


def mask_patch_overlaps(mask_224: np.ndarray) -> np.ndarray:
    """224x224 float mask -> [256] patch overlaps (16x16 grid mean)."""
    patch = 224 // PATCH_GRID
    return mask_224.reshape(PATCH_GRID, patch, PATCH_GRID, patch).mean(axis=(1, 3)).reshape(-1)


def token_ids_from_mask(mask: np.ndarray, threshold: float = 0.05) -> tuple[list[int], np.ndarray]:
    """Raw GT segmentation mask (256x256) -> visual token indices + overlaps."""
    m224 = _mask_to_224(mask)
    overlaps = mask_patch_overlaps(m224)
    ids = np.flatnonzero(overlaps >= threshold).astype(int).tolist()
    return ids, overlaps


def kmeans_groups(h: np.ndarray, K: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """h [256,4096] -> (labels [256], group means v [K,4096] L2-normalized)."""
    from sklearn.cluster import KMeans
    hd = h.astype(np.float64)
    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(hd)
    labels = km.labels_
    v = np.zeros((K, hd.shape[1]), dtype=np.float64)
    for k in range(K):
        idx = np.flatnonzero(labels == k)
        v[k] = hd[idx].mean(axis=0) if len(idx) else 0.0
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return labels, vn


def entity_select(h: np.ndarray, entity_embs: list[np.ndarray], K: int = 8, seed: int = 0):
    """language-selected entity group set -> (selected token ids, meta)."""
    labels, vn = kmeans_groups(h, K, seed)
    sel_groups: list[int] = []
    per_entity_score: list[float] = []
    for e in entity_embs:
        en = e / (np.linalg.norm(e) + 1e-8)
        c = vn @ en
        g = int(np.argmax(c))
        per_entity_score.append(float(c[g]))
        if g not in sel_groups:
            sel_groups.append(g)
    selected = sorted({int(x) for g in sel_groups for x in np.flatnonzero(labels == g)})
    meta = {"selected_groups": sel_groups, "selected_token_ids": selected,
            "per_entity_score": per_entity_score,
            "language_score": float(max(per_entity_score)) if per_entity_score else 0.0,
            "n_entities": len(entity_embs)}
    return selected, meta


def compute_clean_ids(model, processor, image_pil, instruction) -> torch.Tensor:
    inputs = _processor_inputs(processor, model, image_pil, instruction)
    return clean_action_token_ids(model, inputs, N_ACTION_TOKENS).detach()
