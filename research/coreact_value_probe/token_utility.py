"""Signed value selection and masked sampling for causal calibration."""

from __future__ import annotations

import numpy as np
import torch

from research.coreact_closed_loop.guidance import (
    GuidanceConfig,
    _full_velocity,
    _prefix_cache,
    replace_visual_tokens,
    select_visual_tokens,
    tensor_sha256,
)
from research.coreact_exploration.instrumentation import attention_ranking_scores, build_prefix_span_map
from research.coreact_value_probe.representation import contextualize_embedded_prefix, pool_contextual_prefix


def probability(probe, representation: torch.Tensor) -> np.ndarray:
    return probe.predict_proba(representation.detach().float().cpu().numpy())[:, 1]


@torch.inference_mode()
def signed_token_sets(model, prepared: dict, noise: torch.Tensor, means: torch.Tensor, probe, *, proposal_count: int = 32, group_count: int = 8) -> dict:
    prefix, pad, att = model.embed_prefix(prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], state=prepared["state"])
    spans = build_prefix_span_map(model, prepared["images"], prepared["image_masks"], prepared["lang_tokens"], prepared["lang_masks"], pad, camera_ids=("camera1", "camera2"))[0]
    _, traces = _full_velocity(model, prefix, pad, att, noise, torch.ones(1, device=noise.device), record_attention=True)
    scores = attention_ranking_scores(traces, prefix.shape[1])["late_half_action_to_context_attention"]
    candidates = select_visual_tokens(spans, scores, method="top", count=proposal_count, random_seed=0)
    clean_hidden = contextualize_embedded_prefix(model, prefix, pad, att)
    clean_rep, _ = pool_contextual_prefix(clean_hidden, [spans])
    clean_value = float(probability(probe, clean_rep)[0])
    masked_prefixes = torch.cat([replace_visual_tokens(prefix, spans, [index], means, ("camera1", "camera2")) for index in candidates])
    masked_hidden = contextualize_embedded_prefix(model, masked_prefixes, pad.expand(proposal_count, -1), att.expand(proposal_count, -1))
    masked_reps, _ = pool_contextual_prefix(masked_hidden, [spans] * proposal_count)
    masked_values = probability(probe, masked_reps)
    utilities = np.asarray([clean_value - value for value in masked_values])
    order_positive = np.argsort(-utilities, kind="stable")[:group_count]
    order_negative = np.argsort(utilities, kind="stable")[:group_count]
    used = set(order_positive.tolist()) | set(order_negative.tolist())
    neutral_order = [index for index in np.argsort(np.abs(utilities), kind="stable") if index not in used][:group_count]
    if len(neutral_order) != group_count:
        raise RuntimeError("not enough disjoint neutral candidates")
    sets = {
        "mask_strong_positive": [candidates[i] for i in order_positive],
        "mask_strong_negative": [candidates[i] for i in order_negative],
        "mask_near_zero": [candidates[i] for i in neutral_order],
    }
    return {"prefix": prefix, "pad": pad, "att": att, "spans": spans, "sets": sets, "clean_value": clean_value, "candidate_indices": candidates, "candidate_utilities": utilities.tolist(), "prefix_sha256": tensor_sha256(prefix)}


@torch.inference_mode()
def sample_selected_mask(model, selection: dict, indices: list[int], noise: torch.Tensor, means: torch.Tensor) -> tuple[torch.Tensor, dict]:
    masked = replace_visual_tokens(selection["prefix"], selection["spans"], indices, means, ("camera1", "camera2"))
    changed = (selection["prefix"] != masked).any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
    if sorted(changed) != sorted(indices):
        raise RuntimeError("masked prefix changed unexpected indices")
    cache = _prefix_cache(model, masked, selection["pad"], selection["att"])
    x = noise.clone(); dt = -1.0 / model.config.num_steps
    for step in range(model.config.num_steps):
        time = torch.full((1,), 1.0 + step * dt, device=noise.device)
        velocity = model.denoise_step(selection["pad"], cache, x, time)
        if not torch.isfinite(velocity).all():
            raise RuntimeError("nonfinite masked velocity")
        x = x + dt * velocity
    return x, {"selected_indices": indices, "changed_indices": changed, "masked_prefix_sha256": tensor_sha256(masked), "clean_value": selection["clean_value"], "candidate_indices": selection["candidate_indices"], "candidate_utilities": selection["candidate_utilities"]}
