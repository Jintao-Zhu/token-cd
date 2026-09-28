#!/usr/bin/env python3
"""LIBERO-Spatial rollout for the L11 + Matched-count policy.

Only the matched arm is implemented here.  For each state:
  1. clean OpenVLA action generation;
  2. KMeans(K=8) on projector features + LIBERO entity-phrase matching -> m;
  3. L11 prompt-attention Top-m token selection;
  4. harmonic reconstruction of selected projector rows;
  5. SHR-style guided/contrastive decode.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers.generation.logits_process import LogitsProcessorList, SuppressTokensLogitsProcessor

from research.ar_token_counterfactual.intervention import (
    decode_action_ids,
    ensure_empty_action_token,
    projector_intervention,
)
from research.ar_token_counterfactual.libero_runtime import (
    build_prompt,
    encode_video,
    load_policy,
    prepare_agentview,
    prepare_env_action,
    set_determinism,
)
from research.semantic_token_cd.libero_policy import (
    embed_phrase,
    entity_select,
    extract_entities_libero,
    extract_source_target_entities_libero90,
    extract_source_target_entities_libero,
    forward_logits,
    token_ids_from_mask,
)

# Match the SIMPLER LIBERO/OpenVLA evaluation path: avoid oversubscribing CPU
# threads when four closed-loop workers share each host.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

GRID = 16


def stable_top_m(scores: np.ndarray, m: int) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (N_VISUAL,) or not np.isfinite(values).all():
        raise ValueError(f"invalid prompt attention scores: {values.shape}")
    if not 1 <= int(m) <= N_VISUAL:
        raise ValueError(f"invalid coverage m={m}")
    order = np.lexsort((np.arange(N_VISUAL), -values))
    return sorted(int(x) for x in order[: int(m)])


def harmonic_reconstruct(features: np.ndarray, region: np.ndarray) -> np.ndarray:
    region = np.asarray(sorted(set(int(i) for i in region)), dtype=np.int64)
    if not region.size:
        raise ValueError("empty harmonic region")
    if region.size == N_VISUAL:
        # A full-grid harmonic system has no Dirichlet boundary and is
        # singular (its constant vector is in the nullspace).  This degenerate
        # case should be rare, but guarding it keeps long autonomous runs
        # alive: use the global visual mean as the fallback negative branch.
        clean = features.astype(np.float64, copy=False)
        return np.repeat(clean.mean(axis=0, keepdims=True), region.size, axis=0).astype(np.float32)
    position = {int(token): row for row, token in enumerate(region)}
    matrix = np.zeros((region.size, region.size), dtype=np.float64)
    rhs = np.zeros((region.size, features.shape[1]), dtype=np.float64)
    clean = features.astype(np.float64, copy=False)
    for row, token in enumerate(region):
        r, c = divmod(int(token), GRID)
        neighbors = []
        if r > 0: neighbors.append(token - GRID)
        if r + 1 < GRID: neighbors.append(token + GRID)
        if c > 0: neighbors.append(token - 1)
        if c + 1 < GRID: neighbors.append(token + 1)
        matrix[row, row] = len(neighbors)
        for neighbor in neighbors:
            column = position.get(int(neighbor))
            if column is None:
                rhs[row] += clean[neighbor]
            else:
                matrix[row, column] -= 1.0
    try:
        solved = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("singular harmonic region") from exc
    if not np.isfinite(solved).all():
        raise FloatingPointError("non-finite harmonic reconstruction")
    return solved.astype(np.float32)

LAMBDA = 0.5
K = 8
KMEANS_SEED = 0
MAX_STEPS = 220
N_VISUAL = 256

# Diagnostic-only simulator instance names.  Never used by the attention arms.
GT_TARGET_NAME = "akita_black_bowl_1"
GT_REFERENCE_NAMES = {
    0: ("plate_1", "glazed_rim_porcelain_ramekin_1"),
    1: ("glazed_rim_porcelain_ramekin_1",),
    3: ("cookies_1",),
    4: ("wooden_cabinet_1",),
    5: ("glazed_rim_porcelain_ramekin_1",),
    6: ("cookies_1",),
    7: ("flat_stove_1",),
    8: ("plate_1",),
    9: ("wooden_cabinet_1",),
}


def priority_top_m(scores: np.ndarray, priority_tokens: set[int], m: int) -> list[int]:
    """Top-m with a pre-registered GT priority set; fill by attention if needed."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (N_VISUAL,) or not np.isfinite(values).all():
        raise ValueError(f"invalid ranking scores: {values.shape}")
    priority = sorted(int(x) for x in priority_tokens if 0 <= int(x) < N_VISUAL)
    m = int(m)
    if not priority:
        return stable_top_m(values, m)
    order = np.lexsort((np.arange(N_VISUAL), -values))
    if len(priority) >= m:
        rank = {int(token): rank for rank, token in enumerate(priority)}
        chosen = sorted(priority, key=lambda token: (-values[token], token))[:m]
    else:
        chosen = list(priority)
        outside = [int(x) for x in order if int(x) not in set(priority)]
        chosen.extend(outside[:m - len(chosen)])
    return sorted(chosen)


def find_instruction_span(ids: list[int], tokenizer, instruction: str) -> tuple[int, int]:
    """Locate the unique instruction-token span inside the official prompt."""
    candidates: list[tuple[int, int]] = []
    for add_special in (False, True):
        sub = tokenizer(instruction, add_special_tokens=add_special)["input_ids"]
        if add_special and sub and int(sub[0]) == int(tokenizer.bos_token_id):
            sub = sub[1:]
        sub = [int(value) for value in sub]
        if not sub or len(sub) > len(ids):
            continue
        for start in range(len(ids) - len(sub) + 1):
            if ids[start:start + len(sub)] == sub:
                candidates.append((start, start + len(sub)))
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            "instruction token span is not unique in the official prompt: "
            f"instruction={instruction!r}, candidates={unique}"
        )
    return unique[0]


def parse_layer_spec(value: str) -> tuple[int, ...]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            layers.extend(range(lo, hi + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    if not layers:
        raise ValueError("empty attention layer specification")
    return tuple(layers)


def find_phrase_span(
    ids: list[int], tokenizer, phrase: str, search_start: int = 0, search_end: int | None = None
) -> tuple[int, int]:
    phrase = phrase.lower().strip()
    if search_end is None:
        search_end = len(ids)
    candidates: list[tuple[int, int]] = []
    for add_special in (False, True):
        sub = tokenizer(phrase, add_special_tokens=add_special)["input_ids"]
        if add_special and sub and int(sub[0]) == int(tokenizer.bos_token_id):
            sub = sub[1:]
        sub = [int(value) for value in sub]
        if not sub or len(sub) > search_end - search_start:
            continue
        for start in range(search_start, search_end - len(sub) + 1):
            if ids[start:start + len(sub)] == sub:
                candidates.append((start, start + len(sub)))
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"phrase span is not unique: {phrase!r}: {unique}")
    return unique[0]


def source_relation_text(instruction: str) -> str:
    target, relation, reference = role_texts(instruction)
    return f"{target} {relation} {reference}"


def destination_text(instruction: str) -> str:
    text = instruction.lower()
    match = re.search(
        r"\b(?:place|put|move)\b.*?\b(?:on|in|onto|into|near|to|next\s+to)\b\s+(.*)$",
        text,
    )
    if not match:
        raise RuntimeError(f"cannot parse destination from {instruction!r}")
    phrase = match.group(1).strip()
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase)
    phrase = re.sub(r"\b(?:it|them|this|that)\b", "", phrase)
    return " ".join(re.findall(r"[a-z0-9]+", phrase))


def parse_head_spec(value: str) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        layer, head = part.split(':', 1)
        values.append((int(layer), int(head)))
    return tuple(sorted(set(values)))


def role_texts(instruction: str) -> tuple[str, str, str]:
    target = extract_source_target_entities_libero(instruction)[0]
    first_clause = re.split(r"\band\s+(?:place|put|move)\b", instruction.lower(), maxsplit=1)[0]
    stripped = re.sub(r"^pick\s+up\s+(?:the\s+)?", "", first_clause).strip()
    target_words = target.split()
    words = stripped.split()
    while words and words[0] in {"the", "a", "an"}:
        words.pop(0)
    if words[:len(target_words)] == target_words:
        words = words[len(target_words):]
    remainder = " ".join(words).strip()
    for relation in ("next to", "between", "from", "on", "in"):
        if remainder.startswith(relation + " "):
            return target, relation, remainder[len(relation):].strip()
    raise RuntimeError(f"cannot parse relation/reference from {instruction!r}: {remainder!r}")


def span_indices(
    ids: list[int], tokenizer, texts: tuple[str, ...], search_start: int = 0, search_end: int | None = None
) -> list[int]:
    result: set[int] = set()
    for text in texts:
        start, end = find_phrase_span(ids, tokenizer, text, search_start, search_end)
        result.update(range(start, end))
    return sorted(result)


def _normalize_attention(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("invalid attention map mass")
    return values / total


def _attention_scores(attention, query_indices: list[int]) -> np.ndarray:
    values = attention[0, :, [N_VISUAL + i for i in query_indices], 1 : 1 + N_VISUAL]
    return values.detach().float().cpu().mean(dim=(0, 1)).numpy()


def prompt_attention_and_features(
    model,
    processor,
    image: Image.Image,
    instruction: str,
    inputs=None,
    expected_h=None,
    query_mode: str = "instruction_only",
    attention_layers: tuple[int, ...] = (11,),
    attention_heads: tuple[tuple[int, int], ...] = (),
    destination_weight: float = 0.0,
    lambda_scale: float = 1.0,
):
    if inputs is None:
        inputs = processor(build_prompt(instruction), image).to(model.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    if out.attentions is None or out.projector_features is None:
        raise RuntimeError("LIBERO model did not return attentions/projector features")
    if not attention_layers or min(attention_layers) < 0 or max(attention_layers) >= len(out.attentions):
        raise ValueError(f"invalid attention layers: {attention_layers}")

    tokenizer = processor.tokenizer
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    special = set(int(x) for x in tokenizer.all_special_ids)
    prompt_text_indices = [i for i, token in enumerate(ids) if token not in special]
    if not prompt_text_indices or any(i == 0 for i in prompt_text_indices):
        raise RuntimeError(f"invalid LIBERO prompt query layout: {prompt_text_indices}")

    instruction_span = find_instruction_span(ids, tokenizer, instruction)
    role_spans = None
    query_indices = None
    source_relation_span = None
    destination_span = None
    if query_mode == "prompt":
        query_indices = prompt_text_indices
    elif query_mode == "instruction_only":
        query_indices = list(range(instruction_span[0], instruction_span[1]))
    elif query_mode == "full_instruction_endpoint":
        query_indices = [instruction_span[1] - 1]
    elif query_mode == "target_relation_endpoint":
        source_relation_span = find_phrase_span(ids, tokenizer, source_relation_text(instruction))
        query_indices = [source_relation_span[1] - 1]
    elif query_mode == "source_relation":
        source_relation_span = find_phrase_span(ids, tokenizer, source_relation_text(instruction))
        destination_clause = re.split(r"\band\s+(?:place|put|move)\b", instruction.lower(), maxsplit=1)[1].strip()
        destination_clause_span = find_phrase_span(ids, tokenizer, destination_clause)
        destination_span = find_phrase_span(
            ids, tokenizer, destination_text(instruction), *destination_clause_span
        )
    elif query_mode == "role":
        target_text, relation_text, reference_text = role_texts(instruction)
        source_text = re.split(r"\band\s+(?:place|put|move)\b", instruction.lower(), maxsplit=1)[0]
        source_span = find_phrase_span(ids, tokenizer, source_text)
        role_spans = {
            "target": span_indices(ids, tokenizer, (target_text,), *source_span),
            "relation": span_indices(ids, tokenizer, (relation_text,), *source_span),
            "reference": span_indices(ids, tokenizer, (reference_text,), *source_span),
        }
    else:
        raise ValueError(f"unknown attention query_mode: {query_mode}")

    def combine_source_destination(source_map: np.ndarray, destination_map: np.ndarray) -> np.ndarray:
        combined = _normalize_attention(source_map)
        if destination_weight > 0:
            combined = _normalize_attention(combined + float(destination_weight) * _normalize_attention(destination_map))
        return combined

    def head_score(attention, head: int) -> np.ndarray:
        if head < 0 or head >= attention.shape[1]:
            raise ValueError(f"invalid attention head {head} for layer shape {tuple(attention.shape)}")
        if query_mode == "role":
            role_maps = [
                _normalize_attention(_attention_scores(attention[:, head:head + 1], role_spans[name]))
                for name in ("target", "relation", "reference")
            ]
            return _normalize_attention(np.mean(role_maps, axis=0))
        if query_mode == "source_relation":
            source_map = _attention_scores(attention[:, head:head + 1], source_relation_span)
            destination_map = _attention_scores(attention[:, head:head + 1], destination_span)
            return combine_source_destination(source_map, destination_map)
        assert query_indices is not None
        return _normalize_attention(_attention_scores(attention[:, head:head + 1], query_indices))

    if attention_heads:
        scores = _normalize_attention(np.mean(
            [head_score(out.attentions[layer], head) for layer, head in attention_heads],
            axis=0,
        )).astype(np.float32)
    else:
        per_layer_scores = []
        for layer in attention_layers:
            attention = out.attentions[layer]
            if query_mode == "role":
                role_maps = [
                    _normalize_attention(_attention_scores(attention, role_spans[name]))
                    for name in ("target", "relation", "reference")
                ]
                score = _normalize_attention(np.mean(role_maps, axis=0))
            elif query_mode == "source_relation":
                source_map = _attention_scores(attention, source_relation_span)
                destination_map = _attention_scores(attention, destination_span)
                score = combine_source_destination(source_map, destination_map)
            else:
                assert query_indices is not None
                score = _normalize_attention(_attention_scores(attention, query_indices))
            per_layer_scores.append(score)
        scores = _normalize_attention(np.mean(per_layer_scores, axis=0)).astype(np.float32)
    if scores.shape != (N_VISUAL,) or not np.isfinite(scores).all():
        raise RuntimeError(f"invalid attention scores: {scores.shape}")

    h = out.projector_features[0].detach().float().cpu().numpy()
    if h.shape[0] != N_VISUAL:
        raise RuntimeError(f"invalid projector feature shape: {h.shape}")
    if expected_h is not None:
        expected = expected_h.detach().float().cpu().numpy()
        max_abs_diff = float(np.max(np.abs(h - expected)))
        if not np.allclose(h, expected, rtol=1e-3, atol=1e-3) or max_abs_diff > 1e-2:
            raise RuntimeError(
                "prompt-attention projector features differ from clean generation: "
                f"max_abs_diff={max_abs_diff}"
            )
    meta = {
        "attention_query_mode": query_mode,
        "attention_layers": list(attention_layers),
        "attention_heads": [list(x) for x in attention_heads],
        "prompt_text_indices": prompt_text_indices,
        "query_indices": query_indices,
        "instruction_span": list(instruction_span),
        "source_relation_span": list(source_relation_span) if source_relation_span is not None else None,
        "destination_span": list(destination_span) if destination_span is not None else None,
        "destination_weight": float(destination_weight),
        "role_spans": role_spans,
    }
    return scores, h, meta


def generate_clean_action(model, processor, image: Image.Image, instruction: str, inputs=None):
    """AR-generate the seven clean action tokens and their per-position logits.

    This mirrors ``PromptAttentionSHRInference._forward_scores``: EOS is
    suppressed so every action dimension is emitted even when the language EOS
    token would otherwise terminate generation early.  The projector hook
    captures the clean visual features used by the positive branch.
    """
    if inputs is None:
        inputs = processor(build_prompt(instruction), image).to(model.device, dtype=torch.bfloat16)
    input_ids, attention_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    eos_id = int(model.generation_config.eos_token_id)
    with projector_intervention(model) as trace:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=7,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList([SuppressTokensLogitsProcessor([eos_id])]),
        )
    if trace.before is None:
        raise RuntimeError("clean generation did not expose projector features")
    if not getattr(generated, "scores", None) or len(generated.scores) != 7:
        got = None if not getattr(generated, "scores", None) else len(generated.scores)
        raise RuntimeError(f"clean generation emitted {got} action positions instead of 7")
    clean_logits = torch.stack([row[0].detach().float() for row in generated.scores], dim=0)
    model_vocab_size = int(model.vocab_size)
    logits_vocab_size = int(clean_logits.shape[1])
    action_start = model_vocab_size - 256
    if clean_logits.shape[0] != 7 or logits_vocab_size < model_vocab_size:
        raise RuntimeError(f"invalid clean action logits shape={tuple(clean_logits.shape)}")
    if not torch.isfinite(clean_logits[:, action_start:action_start + 256]).all():
        raise FloatingPointError("non-finite clean action-bin logits")
    clean_ids = clean_logits.argmax(dim=-1).unsqueeze(0)
    return clean_ids, clean_logits.detach().float().cpu(), trace.before


def predict_matched(
    model,
    processor,
    image: Image.Image,
    instruction: str,
    inputs=None,
    entity_mode: str = "source_target",
    query_mode: str = "instruction_only",
    attention_layers: tuple[int, ...] = (11,),
    attention_heads: tuple[tuple[int, int], ...] = (),
    destination_weight: float = 0.0,
    lambda_scale: float = 1.0,
    unnorm_key: str = "libero_spatial",
    position_mode: str = "attention",
    selector_transform: str = "identity",
    fixed_m: int | None = None,
    priority_tokens: set[int] | None = None,
    target_priority_tokens: set[int] | None = None,
    reference_priority_tokens: set[int] | None = None,
    distractor_priority_tokens: set[int] | None = None,
    spatial_null_prior: np.ndarray | None = None,
    spatial_null_seed: int | None = None,
    rank_bin: int | None = None,
    random_seed: int | None = None,
):
    """Return decoded action [7] and per-step matched metadata."""
    if inputs is None:
        inputs = processor(build_prompt(instruction), image).to(model.device, dtype=torch.bfloat16)
    clean_ids, clean_logits, clean_h = generate_clean_action(model, processor, image, instruction, inputs)

    # An object-only mask can legitimately be empty when that object is fully
    # occluded or occupies less than the segmentation-to-patch overlap
    # threshold. In that state the requested intervention has no visible
    # tokens, so execute the clean action instead of aborting the episode.
    if fixed_m == 0:
        clean_token_ids = clean_ids[0].detach().cpu().tolist()
        action = decode_action_ids(model, clean_ids.detach().cpu(), unnorm_key)
        target_set = set(int(x) for x in (target_priority_tokens or ()))
        distractor_set = set(int(x) for x in (distractor_priority_tokens or ()))
        meta = {
            "entity_mode": entity_mode,
            "unnorm_key": unnorm_key,
            "attention_query_mode": query_mode,
            "attention_layers": list(attention_layers),
            "attention_heads": [list(x) for x in attention_heads],
            "destination_weight": float(destination_weight),
            "entities": [],
            "kmeans_groups": None,
            "kmeans_cluster_token_ids": None,
            "m_matched": None,
            "m_used": 0,
            "fixed_m": 0,
            "position_mode": position_mode,
            "selector_transform": selector_transform,
            "selector_source": "empty_object_mask_clean_fallback",
            "priority_token_count": len(priority_tokens) if priority_tokens is not None else 0,
            "target_priority_token_count": len(target_set),
            "distractor_priority_token_count": len(distractor_set),
            "reference_priority_token_count": len(reference_priority_tokens or ()),
            "selected_priority_count": 0,
            "selected_target_count": 0,
            "selected_distractor_count": 0,
            "selected_reference_count": 0,
            "selected_outside_priority_count": 0,
            "selected_attention_overlap_count": 0,
            "selected_attention_jaccard": 0.0,
            "attention_top_m_token_ids": [],
            "selected_token_ids": [],
            "positive_token_ids": clean_token_ids,
            "negative_token_ids": clean_token_ids,
            "final_token_ids": clean_token_ids,
            "guided_prefix": False,
            "guided_changed_dims": 0,
            "eos_token_id": int(model.generation_config.eos_token_id),
            "attention_sha256": None,
            "feature_perturbation_norm": 0.0,
            "feature_perturbation_relative": 0.0,
            "non_target_bit_identical": True,
            "reconstruction_finite": True,
            "lambda": float(LAMBDA) * float(lambda_scale),
            "lambda_scale": float(lambda_scale),
            "empty_object_mask_clean_fallback": True,
        }
        return action, meta

    attention, h, attention_meta = prompt_attention_and_features(
        model,
        processor,
        image,
        instruction,
        inputs,
        expected_h=clean_h,
        query_mode=query_mode,
        attention_layers=attention_layers,
        attention_heads=attention_heads,
        destination_weight=destination_weight,
    )
    if rank_bin is not None or random_seed is not None:
        if fixed_m != 32 or query_mode != "instruction_only" or tuple(attention_layers) != (11,):
            raise ValueError("rank-bin/Random32 closed-loop arms require fixed K=32, instruction-only L11")
        if rank_bin is not None and (not 0 <= int(rank_bin) < 8 or random_seed is not None):
            raise ValueError("rank_bin must be in [0,7] and cannot be combined with random_seed")
        entities = []
        kmeans_meta = {}
        matched_m = None
    else:
        if entity_mode == "source_target":
            entities = extract_source_target_entities_libero(instruction)
        elif entity_mode == "source_target_libero90":
            entities = extract_source_target_entities_libero90(instruction)
        elif entity_mode == "all":
            entities = extract_entities_libero(instruction)
        else:
            raise ValueError(f"unknown entity_mode: {entity_mode}")
        if not entities:
            raise RuntimeError(f"no entities extracted from instruction: {instruction}")
        entity_embs = [embed_phrase(model, processor.tokenizer, e) for e in entities]
        kmeans_selected, kmeans_meta = entity_select(h, entity_embs, K=K, seed=KMEANS_SEED)
        matched_m = len(kmeans_selected)
    m = matched_m if fixed_m is None else int(fixed_m)
    if not 1 <= m <= N_VISUAL:
        raise ValueError(f"invalid intervention budget m={m}")
    if spatial_null_prior is not None:
        prior = np.asarray(spatial_null_prior, dtype=np.float64).reshape(-1)
        if prior.shape != (N_VISUAL,) or not np.isfinite(prior).all() or (prior < 0).any() or prior.sum() <= 0:
            raise ValueError("invalid spatial_null_prior")
        if spatial_null_seed is None:
            raise ValueError("spatial_null_seed is required with spatial_null_prior")
        if selector_transform != "identity":
            raise ValueError("spatial null cannot be combined with selector_transform")
        prior = prior / prior.sum()
        selected = np.random.default_rng(int(spatial_null_seed)).choice(
            N_VISUAL, size=m, replace=False, p=prior
        ).astype(np.int64).tolist()
        selector_source = "task_position_prior_weighted_sampling"
    elif rank_bin is not None:
        if selector_transform != "identity" or priority_tokens is not None or spatial_null_prior is not None:
            raise ValueError("rank-bin selector cannot be combined with selector transforms or spatial priorities")
        order = np.lexsort((np.arange(N_VISUAL), -np.asarray(attention, dtype=np.float64)))
        selected = sorted(int(x) for x in order[int(rank_bin) * 32:(int(rank_bin) + 1) * 32])
        selector_scores = attention
        selector_source = f"current_l11_rank_bin_{int(rank_bin) + 1}"
    elif random_seed is not None:
        if selector_transform != "identity" or priority_tokens is not None or spatial_null_prior is not None:
            raise ValueError("Random32 cannot be combined with selector transforms or spatial priorities")
        selected = sorted(int(x) for x in np.random.default_rng(int(random_seed)).choice(
            N_VISUAL, size=32, replace=False
        ))
        selector_scores = attention
        selector_source = "deterministic_random32"
    elif selector_transform == "identity":
        selector_scores = attention
        selected = None
        selector_source = "current_l11_attention"
    elif selector_transform == "rot180":
        # A fixed spatial placebo for selector-only experiments.  Rotating the
        # score grid by 180 degrees is a graph automorphism of the 16x16
        # four-neighbor lattice, so it preserves score values and mask topology
        # while changing their correspondence to the observed scene.
        selector_scores = np.rot90(attention.reshape(GRID, GRID), 2).reshape(-1).copy()
        selected = None
        selector_source = "rotated_l11_attention"
    else:
        raise ValueError(f"unknown selector transform: {selector_transform}")
    if selected is None:
        if priority_tokens is None:
            selected = stable_top_m(selector_scores, m)
        else:
            selected = priority_top_m(selector_scores, priority_tokens, m)

    replacement = h.copy()
    replacement[np.asarray(selected, dtype=np.int64)] = harmonic_reconstruct(h, np.asarray(selected, dtype=np.int64))
    neg_logits = forward_logits(
        model, processor, image, instruction, clean_ids,
        selected=selected, mean=torch.from_numpy(replacement).float(),
    )
    if neg_logits.shape != clean_logits.shape:
        raise RuntimeError(f"negative/clean logit shape mismatch: {neg_logits.shape} vs {clean_logits.shape}")

    # Preserve the seventh (gripper) action dimension exactly as in SHR, and
    # keep EOS finite-but-impossible so argmax can only select action-bin tokens.
    lambd = float(LAMBDA) * float(lambda_scale)
    if not np.isfinite(lambd) or lambd < 0:
        raise ValueError(f"invalid lambda_scale={lambda_scale}")
    attention_selected = set(stable_top_m(attention, m))
    selected_set = set(int(x) for x in selected)
    target_set = set(int(x) for x in (target_priority_tokens or ()))
    reference_set = set(int(x) for x in (reference_priority_tokens or ()))
    distractor_set = set(int(x) for x in (distractor_priority_tokens or ()))
    final = clean_logits.clone()
    final[:-1] = (1.0 + lambd) * clean_logits[:-1] - lambd * neg_logits[:-1]
    eos_id = int(model.generation_config.eos_token_id)
    final[:, eos_id] = torch.finfo(final.dtype).min
    if not torch.isfinite(final).all():
        raise FloatingPointError("non-finite final SHR action logits")
    token_ids = final.argmax(dim=-1)
    action = decode_action_ids(model, token_ids.unsqueeze(0).detach().cpu(), unnorm_key)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError(f"invalid decoded action: shape={action.shape}, action={action}")
    meta = {
        "entity_mode": entity_mode,
        "unnorm_key": unnorm_key,
        "attention_query_mode": query_mode,
        "attention_layers": attention_meta["attention_layers"],
        "attention_heads": attention_meta["attention_heads"],
        "destination_weight": attention_meta["destination_weight"],
        "attention_query_indices": attention_meta["query_indices"],
        "attention_role_spans": attention_meta["role_spans"],
        "instruction_span": attention_meta["instruction_span"],
        "entities": entities,
        "kmeans_groups": kmeans_meta.get("selected_groups"),
        "kmeans_cluster_token_ids": kmeans_meta.get("selected_token_ids"),
        "m_matched": int(matched_m) if matched_m is not None else None,
        "m_used": int(m),
        "fixed_m": int(fixed_m) if fixed_m is not None else None,
        "position_mode": position_mode,
        "selector_transform": selector_transform,
        "selector_source": selector_source,
        "priority_token_count": len(priority_tokens) if priority_tokens is not None else 0,
        "target_priority_token_count": len(target_set),
        "distractor_priority_token_count": len(distractor_set),
        "reference_priority_token_count": len(reference_set),
        "selected_priority_count": len(selected_set & (target_set | reference_set | distractor_set)),
        "selected_target_count": len(selected_set & target_set),
        "selected_distractor_count": len(selected_set & distractor_set),
        "selected_reference_count": len(selected_set & reference_set),
        "selected_outside_priority_count": len(selected_set - target_set - reference_set - distractor_set),
        "selected_attention_overlap_count": len(selected_set & attention_selected),
        "selected_attention_jaccard": len(selected_set & attention_selected) / max(1, len(selected_set | attention_selected)),
        "attention_top_m_token_ids": sorted(attention_selected),
        "selected_token_ids": selected,
        "positive_token_ids": clean_ids[0].detach().cpu().tolist(),
        "negative_token_ids": neg_logits.argmax(dim=-1).detach().cpu().tolist(),
        "final_token_ids": token_ids.detach().cpu().tolist(),
        "guided_prefix": True,
        "guided_changed_dims": int((token_ids[:6] != clean_ids[0, :6].detach().cpu()).sum().item()),
        "eos_token_id": eos_id,
        "attention_sha256": __import__("hashlib").sha256(
            np.ascontiguousarray(attention.astype(np.float32)).tobytes()
        ).hexdigest(),
        "feature_perturbation_norm": float(np.linalg.norm(replacement - h)),
        "feature_perturbation_relative": float(np.linalg.norm(replacement - h) / (np.linalg.norm(h) + 1e-12)),
        "non_target_bit_identical": True,
        "reconstruction_finite": bool(np.isfinite(replacement).all()),
        "lambda": lambd,
        "lambda_scale": float(lambda_scale),
    }
    return action, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", required=True, help="comma-separated episode ids")
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument(
        "--entity-mode",
        choices=("all", "source_target", "source_target_libero90"),
        default="source_target",
    )
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--unnorm-key", default="libero_spatial")
    p.add_argument("--dataset-statistics", type=Path, default=None)
    p.add_argument(
        "--query-mode",
        choices=("prompt", "instruction_only", "role", "source_relation", "target_relation_endpoint", "full_instruction_endpoint"),
        default="instruction_only",
    )
    p.add_argument("--attention-layers", default="11")
    p.add_argument("--attention-heads", default="", help="comma-separated layer:head pairs")
    p.add_argument("--destination-weight", type=float, default=0.0)
    p.add_argument("--lambda-scale", type=float, default=1.0)
    p.add_argument(
        "--position-mode",
        choices=(
            "attention", "gt_target", "gt_target_reference",
            "object_target_only", "object_distractor_only",
            "non_bowl_target_count", "non_bowl_distractor_count",
        ),
        default="attention",
        help="diagnostic-only GT position priority; attention is the method",
    )
    p.add_argument(
        "--selector-transform",
        choices=("identity", "rot180"),
        default="identity",
        help="identity is canonical; rot180 is a fixed spatial placebo preserving m and mask topology",
    )
    p.add_argument("--fixed-m", type=int, default=None, help="diagnostic arm: hold selected-token count fixed")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--settle-steps", type=int, default=10)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--video-dir", type=Path, default=None)
    a = p.parse_args()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

    episodes = []
    for part in a.episodes.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            episodes.extend(range(lo, hi + 1))
        else:
            episodes.append(int(part))
    episodes = sorted(set(episodes))
    if a.smoke:
        episodes = episodes[:1]
    default_max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = int(
        a.max_steps
        if a.max_steps is not None
        else (3 if a.smoke else default_max_steps.get(a.suite, MAX_STEPS))
    )
    suite = benchmark.get_benchmark_dict()[a.suite]()
    task_index = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == a.task)
    task = suite.get_task(task_index)
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env_cls = SegmentationRenderEnv if a.position_mode != "attention" else OffScreenRenderEnv
    env = env_cls(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    init_states = suite.get_task_init_states(task_index)

    code_dir = Path(
        os.environ.get(
            "OPENVLA_HF_CODE_DIR",
            Path(__file__).resolve().parents[2] / "third_party/openvla/prismatic/extern/hf",
        )
    )
    set_determinism(7)
    model, processor = load_policy(
        a.checkpoint,
        code_dir,
        device=f"cuda:{a.gpu}",
        dataset_statistics_path=a.dataset_statistics,
        unnorm_key=a.unnorm_key,
    )
    root = a.artifact.resolve() / task.name
    root.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        out_path = root / f"episode_{ep:03d}.json"
        if out_path.exists():
            print(json.dumps({"skip": True, "task": task.name, "episode": ep}), flush=True)
            continue
        init_index = ep % len(init_states)
        init = init_states[init_index]
        env.seed(int(a.env_seed))
        env.reset()
        obs = env.set_init_state(init)
        for _ in range(int(a.settle_steps)):
            obs, _reward, _done, _info = env.step([0, 0, 0, 0, 0, 0, -1])
        initial_state = np.asarray(env.get_sim_state()).copy()
        trajectory = []
        trace = []
        video_frames = []
        done = False
        started = time.monotonic()
        for step in range(max_steps):
            _, image = prepare_agentview(obs)
            if a.video_dir is not None:
                video_frames.append(np.asarray(image.copy()))
            priority_tokens = None
            target_priority_tokens = None
            reference_priority_tokens = None
            distractor_priority_tokens = None
            fixed_m_this_step = a.fixed_m
            if a.position_mode != "attention":
                seg = np.asarray(obs["agentview_segmentation_instance"])
                target_mask = seg[..., 0] == env.instance_to_id[GT_TARGET_NAME]
                target_ids, _ = token_ids_from_mask(target_mask.astype(np.uint8))
                target_priority_tokens = set(int(x) for x in target_ids)
                object_only_modes = {
                    "object_target_only", "object_distractor_only",
                    "non_bowl_target_count", "non_bowl_distractor_count",
                }
                if a.position_mode in object_only_modes:
                    distractor_mask = seg[..., 0] == env.instance_to_id["akita_black_bowl_2" if GT_TARGET_NAME == "akita_black_bowl_1" else "akita_black_bowl_1"]
                    distractor_ids, _ = token_ids_from_mask(distractor_mask.astype(np.uint8))
                    distractor_priority_tokens = set(int(x) for x in distractor_ids)
                    if a.position_mode == "object_target_only":
                        priority_tokens = set(target_priority_tokens)
                        fixed_m_this_step = len(target_priority_tokens)
                    elif a.position_mode == "object_distractor_only":
                        priority_tokens = set(distractor_priority_tokens)
                        fixed_m_this_step = len(distractor_priority_tokens)
                    else:
                        priority_tokens = set(range(N_VISUAL)) - target_priority_tokens - distractor_priority_tokens
                        if a.position_mode == "non_bowl_target_count":
                            fixed_m_this_step = len(target_priority_tokens)
                        else:
                            fixed_m_this_step = len(distractor_priority_tokens)
                else:
                    distractor_priority_tokens = set()
                reference_mask = np.zeros_like(target_mask)
                if a.position_mode == "gt_target_reference":
                    for name in GT_REFERENCE_NAMES.get(task_index, ()):
                        if name in env.instance_to_id:
                            reference_mask |= seg[..., 0] == env.instance_to_id[name]
                reference_ids, _ = token_ids_from_mask(reference_mask.astype(np.uint8))
                reference_priority_tokens = set(int(x) for x in reference_ids)
                if a.position_mode == "gt_target":
                    priority_tokens = set(target_priority_tokens)
                elif a.position_mode == "gt_target_reference":
                    priority_tokens = set(target_priority_tokens) | set(reference_priority_tokens)
            raw_action, meta = predict_matched(
                model,
                processor,
                image,
                task.language,
                entity_mode=a.entity_mode,
                query_mode=a.query_mode,
                attention_layers=parse_layer_spec(a.attention_layers),
                attention_heads=parse_head_spec(a.attention_heads),
                destination_weight=a.destination_weight,
                lambda_scale=a.lambda_scale,
                unnorm_key=a.unnorm_key,
                position_mode=a.position_mode,
                selector_transform=a.selector_transform,
                fixed_m=fixed_m_this_step,
                priority_tokens=priority_tokens,
                target_priority_tokens=target_priority_tokens,
                reference_priority_tokens=reference_priority_tokens,
                distractor_priority_tokens=distractor_priority_tokens,
            )
            action = prepare_env_action(raw_action)
            obs, _reward, done, _info = env.step(action.tolist())
            trajectory.append(action.copy())
            trace.append(meta)
            if done:
                break
        success = bool(env.check_success())
        matched_counts = [x.get("m_matched") for x in trace if x.get("m_matched") is not None]
        effective_mask_counts = [int(x.get("m_used", 0)) for x in trace]
        if a.video_dir is not None and video_frames:
            encode_video(
                video_frames,
                a.video_dir.resolve() / task.name / f"episode_{ep:03d}.mp4",
                fps=30,
            )
        summary = {
            "protocol_id": f"{a.suite.upper()}_L11_MATCHED_OFFICIAL_V1",
            "benchmark": a.suite,
            "task_id": int(task_index),
            "unnorm_key": a.unnorm_key,
            "dataset_statistics_path": str(a.dataset_statistics) if a.dataset_statistics else None,
            "entity_mode": a.entity_mode,
            "query_mode": a.query_mode,
            "attention_layers": list(parse_layer_spec(a.attention_layers)),
            "attention_heads": [list(x) for x in parse_head_spec(a.attention_heads)],
            "destination_weight": float(a.destination_weight),
            "lambda_scale": float(a.lambda_scale),
            "position_mode": a.position_mode,
            "selector_transform": a.selector_transform,
            "fixed_m": a.fixed_m,
            "task": task.name,
            "instruction": task.language,
            "episode": int(ep),
            "init_state_index": int(init_index),
            "env_seed": int(a.env_seed),
            "settle_steps": int(a.settle_steps),
            "max_policy_steps": int(max_steps),
            "success": success,
            "steps": len(trajectory),
            "done": bool(done),
            "normal_end": bool(success or done or len(trajectory) >= max_steps),
            "runtime_seconds": time.monotonic() - started,
            "initial_state_sha256": __import__("hashlib").sha256(initial_state.tobytes()).hexdigest(),
            "selected_token_count_mean": float(np.mean(matched_counts)) if matched_counts else 0.0,
            "selected_token_count_std": float(np.std(matched_counts)) if matched_counts else 0.0,
            "effective_mask_token_count_mean": float(np.mean(effective_mask_counts)) if effective_mask_counts else 0.0,
            "empty_object_mask_fallback_steps": sum(bool(x.get("empty_object_mask_clean_fallback")) for x in trace),
            "trajectory": [x.tolist() for x in trajectory],
            "trace": trace,
        }
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"task": task.name, "episode": ep, "success": success,
                          "steps": len(trajectory), "mean_m": summary["selected_token_count_mean"]}), flush=True)
        if a.smoke:
            break
    env.close()


if __name__ == "__main__":
    main()
