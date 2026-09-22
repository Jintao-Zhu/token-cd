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

import re

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
    """Rule-based LIBERO entity extraction (no LLM/CLIP/GT object list).

    The LIBERO-Spatial instructions contain compound relational phrases, e.g.
    ``pick up the black bowl between the plate and the ramekin and place it on
    the plate``.  Splitting only on the word ``and`` would merge the relation
    object into the source phrase (``black bowl between plate``).  Instead,
    tokenize the instruction and start a new noun phrase at every determiner,
    verb, preposition, conjunction, particle, or pronoun.  Consecutive content
    words remain together, which preserves names such as ``cookie box``,
    ``table center``, ``top drawer``, and ``alphabet soup``.
    """
    stop = (
        PREPOSITIONS
        | DETERMINERS
        | LIBERO_VERBS
        | LIBERO_PARTICLES
        | LIBERO_PRONOUNS
        | {"and", "of", "between", "next", "to"}
    )
    ents: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            phrase = " ".join(current)
            if phrase and phrase not in ents:
                ents.append(phrase)
            current.clear()

    for word in re.findall(r"[a-z0-9]+", instruction.lower()):
        if word in stop:
            flush()
        else:
            current.append(word)
    flush()
    return ents


def extract_source_target_entities_libero(instruction: str) -> list[str]:
    """SIMPLER-style source/target extraction for LIBERO instructions.

    The original SIMPLER semantic selector keeps the object being manipulated
    and the placement target, not the spatial-relation object.  For example:

      pick up the black bowl next to the ramekin and place it on the plate
      -> ["black bowl", "plate"]

    This is intentionally separate from ``extract_entities_libero`` so the
    existing multi-entity LIBERO arm remains available as a controlled
    comparison.
    """
    text = instruction.lower().strip()
    # Split off the placement clause.
    parts = re.split(r"\band\s+(?=(?:place|put|move)\b)", text, maxsplit=1)
    source_clause = parts[0]
    target_clause = parts[1] if len(parts) > 1 else ""

    # Source: remove the action prefix, then cut at the first spatial relation.
    source_clause = re.sub(
        r"^(?:pick\s+up|pick|grab|lift|hold)\s+(?:the\s+)?", "", source_clause
    ).strip()
    relation = re.search(
        r"\b(?:on|in|into|onto|next\s+to|between|from|under|over|near|to)\b",
        source_clause,
    )
    source = source_clause[: relation.start()].strip() if relation else source_clause
    source = re.sub(r"^(?:the|a|an)\s+", "", source)
    source = " ".join(re.findall(r"[a-z0-9]+", source))

    # Target: keep the noun phrase after the placement preposition.
    target = ""
    placement = re.search(
        r"\b(?:place|put|move)\b.*?\b(?:in|into|on|onto|near|to|next\s+to)\b\s+(.*)$",
        target_clause,
    )
    if placement:
        target = placement.group(1)
    else:
        # Fallback: use the final prepositional phrase.
        placement = re.search(
            r"\b(?:in|into|on|onto|near|to|next\s+to)\b\s+(.*)$",
            target_clause,
        )
        target = placement.group(1) if placement else ""
    target = re.sub(r"\b(?:it|them|this|that)\b", "", target)
    target = re.sub(r"^(?:the|a|an)\s+", "", target.strip())
    target = " ".join(re.findall(r"[a-z0-9]+", target))

    entities: list[str] = []
    for entity in (source, target):
        if entity and entity not in entities:
            entities.append(entity)
    return entities


def extract_source_target_entities_libero90(instruction: str) -> list[str]:
    """Explicit source/target parser for the official LIBERO-90 instruction mix.

    LIBERO-90 contains direct placement commands such as ``put the white mug on
    the plate`` and compound displacement commands such as ``put the butter at
    the back in the top drawer ... and close it``.  The Spatial parser above is
    intentionally left unchanged for reproducing the Spatial experiments.
    This parser keeps the same source/destination roles and projects both roles
    into the existing entity-cosine budget rule.
    """
    text = instruction.lower().strip().rstrip(".")
    locative = r"(?:in|into|on|onto|near|to|next\s+to)"

    # Relocate-then-place: the source is the object being picked up and the
    # target is the destination after the second action verb.
    pick_and_place = re.match(
        rf"pick\s+up\s+(?P<source>.*?)\s+and\s+(?:place|put|move)\s+"
        rf"(?:it|them|this|that)?\s*{locative}\s+(?P<target>.*)$",
        text,
    )
    if pick_and_place:
        source = pick_and_place.group("source")
        target = pick_and_place.group("target")
    else:
        # Direct placement: split at the first strong locative relation.  The
        # word "at" is deliberately not a split point, so disambiguating source
        # phrases such as "butter at the back" remain attached to the source.
        direct_place = re.match(
            rf"(?:put|place|move)\s+(?P<source>.*?)\s+{locative}\s+(?P<target>.*)$",
            text,
        )
        if not direct_place:
            raise RuntimeError(f"unsupported LIBERO-90 source/target instruction: {instruction!r}")
        source = direct_place.group("source")
        target = direct_place.group("target")

    # For relocated objects, remove the source-side spatial disambiguator while
    # preserving the manipulated object phrase.  This mirrors the established
    # Spatial source-role semantics.
    source = re.split(rf"\b{locative}\b", source, maxsplit=1)[0]
    target = re.sub(r"\s+and\s+(?:close|open)\b.*$", "", target)
    target = re.sub(r"\b(?:it|them|this|that)\b", " ", target)

    def clean_phrase(value: str) -> str:
        value = value.strip()
        value = re.sub(r"^(?:the|a|an)\s+", "", value)
        return " ".join(re.findall(r"[a-z0-9]+", value))

    source = clean_phrase(source)
    target = clean_phrase(target)
    entities: list[str] = []
    for entity in (source, target):
        if entity and entity not in entities:
            entities.append(entity)
    return entities


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
