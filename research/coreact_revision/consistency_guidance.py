"""Online nuisance-selective CoreAct guidance using cross-tau action consistency."""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Sequence

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
from research.coreact_exploration.instrumentation import (
    attention_ranking_scores,
    build_prefix_span_map,
    grouped_intervention,
)
from research.coreact_exploration.run_development_gates import replacements_for
from research.coreact_revision.run_online_consistency_qualification import FEATURES, features


@torch.no_grad()
def sample_consistency_guided_actions(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    means: dict,
    classifier,
    numeric_features: Sequence[str],
    *,
    camera_ids: Sequence[str] = ("camera1", "camera2"),
    candidate_count: int = 32,
    config: GuidanceConfig = GuidanceConfig(),
    mode: Literal["away", "toward", "mask_only"] = "away",
    nuisance_threshold: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("consistency guidance requires a frozen eval model")
    if state.shape[0] != 1 or noise.shape != (1, model.config.chunk_size, model.config.max_action_dim):
        raise ValueError("consistency guidance requires one checkpoint-native action chunk")
    prefix, prefix_pad, prefix_att = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
    span_map = build_prefix_span_map(model, images, image_masks, lang_tokens, lang_masks, prefix_pad, camera_ids=camera_ids)[0]
    first_time = torch.ones((1,), dtype=torch.float32, device=state.device)
    _, attention_trace = _full_velocity(model, prefix, prefix_pad, prefix_att, noise, first_time, record_attention=True)
    attention = attention_ranking_scores(attention_trace, prefix.shape[1])["late_half_action_to_context_attention"]
    candidates = select_visual_tokens(span_map, attention, method="top", count=candidate_count, random_seed=0)

    replacements = replacements_for(span_map, means, mode="position")
    candidate_prefixes = grouped_intervention(prefix, span_map, [[index] for index in candidates], replacements)
    candidate_pad = prefix_pad.expand(candidate_count, -1)
    positive_cache = _prefix_cache(model, prefix, prefix_pad, prefix_att)
    candidate_cache = _prefix_cache(model, candidate_prefixes, candidate_pad, prefix_att.expand(candidate_count, -1))
    clean_estimates: list[torch.Tensor] = []
    masked_estimates: list[list[torch.Tensor]] = [[] for _ in candidates]
    x_t = noise.clone()
    all_finite = True
    for step in range(config.num_steps):
        tau = 1.0 - step / config.num_steps
        timestep = torch.tensor([tau], dtype=torch.float32, device=state.device)
        clean_velocity = model.denoise_step(prefix_pad, positive_cache, x_t, timestep)
        all_finite &= bool(torch.isfinite(clean_velocity).all())
        if tau in (0.2, 0.5, 0.8):
            masked_velocity = model.denoise_step(
                candidate_pad,
                candidate_cache,
                x_t.expand(candidate_count, -1, -1),
                timestep.expand(candidate_count),
            )
            all_finite &= bool(torch.isfinite(masked_velocity).all())
            clean_estimates.append((x_t - tau * clean_velocity)[0, :, : config.action_dim].reshape(-1).cpu())
            masked_action = (
                x_t.expand(candidate_count, -1, -1) - tau * masked_velocity
            )[:, :, : config.action_dim]
            for row in range(candidate_count):
                masked_estimates[row].append(masked_action[row].reshape(-1).cpu())
        x_t = x_t - clean_velocity / config.num_steps

    classifier_rows = []
    raw_features = []
    clean_stack = torch.stack(clean_estimates)
    for index, estimates in zip(candidates, masked_estimates, strict=True):
        values = features(clean_stack, torch.stack(estimates))
        raw_features.append(values)
        classifier_rows.append(
            [values[name.removeprefix("tau3_seed1729_")] for name in numeric_features]
            + [span_map[index].camera_id]
        )
    probabilities = classifier.predict_proba(np.asarray(classifier_rows, dtype=object))[:, 1]
    if mode not in ("away", "toward", "mask_only"):
        raise ValueError(f"unknown consistency mode {mode}")
    if not 0 <= nuisance_threshold <= 1:
        raise ValueError("nuisance threshold must lie in [0, 1]")
    order = np.asarray(
        [index for index in np.argsort(-probabilities, kind="stable") if probabilities[index] >= nuisance_threshold][
            : config.group_count
        ],
        dtype=int,
    )
    selected = [candidates[int(i)] for i in order]
    negative_prefix = replace_visual_tokens(prefix, span_map, selected, means["visual_position_mean"], camera_ids)
    changed = (prefix != negative_prefix).any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
    if sorted(changed) != sorted(selected):
        raise RuntimeError(f"consistency negative prefix changed {changed}, expected {selected}")

    negative_cache = _prefix_cache(model, negative_prefix, prefix_pad, prefix_att)
    dt = -1.0 / config.num_steps
    x_t = noise.clone()
    step_traces = []
    fallback_count = 0
    for step in range(config.num_steps):
        flow_time = 1.0 + step * dt
        timestep = torch.full((1,), flow_time, dtype=torch.float32, device=state.device)
        clean_velocity = model.denoise_step(prefix_pad, positive_cache, x_t, timestep)
        masked_velocity = model.denoise_step(prefix_pad, negative_cache, x_t, timestep)
        raw_guidance = clean_velocity - masked_velocity
        raw_guidance[..., config.action_dim :] = 0
        clean_norm = torch.linalg.vector_norm(clean_velocity[..., : config.action_dim])
        raw_norm = torch.linalg.vector_norm(raw_guidance[..., : config.action_dim])
        clip_scale = torch.clamp(config.trust_region_kappa * clean_norm / (raw_norm + 1e-12), max=1.0)
        if mode == "mask_only":
            guided_velocity = masked_velocity
        else:
            direction_sign = 1.0 if mode == "away" else -1.0
            guided_velocity = clean_velocity + direction_sign * config.guidance_scale * raw_guidance * clip_scale
        finite = bool(torch.isfinite(guided_velocity).all())
        all_finite &= finite
        if not finite:
            guided_velocity = clean_velocity
            fallback_count += 1
        x_t = x_t + dt * guided_velocity
        step_traces.append({"step": step, "flow_time": flow_time, "clean_velocity_norm": float(clean_norm), "raw_guidance_norm": float(raw_norm), "clip_scale": float(clip_scale), "finite": finite})

    return x_t, {
        "selection": "consistency_nuisance_thresholded_top8_from_attention_top32",
        "mode": mode,
        "nuisance_threshold": nuisance_threshold,
        "candidate_indices": candidates,
        "candidate_attention": [float(attention[index]) for index in candidates],
        "candidate_nuisance_probabilities": [float(value) for value in probabilities],
        "selected_indices": selected,
        "selected_probabilities": [float(probabilities[int(i)]) for i in order],
        "selected_count": len(selected),
        "abstained": len(selected) == 0,
        "changed_indices": changed,
        "selected_tokens": [asdict(span_map[index]) for index in selected],
        "prefix_sha256": tensor_sha256(prefix),
        "negative_prefix_sha256": tensor_sha256(negative_prefix),
        "noise_sha256": tensor_sha256(noise),
        "attention_layer_count": len(attention_trace),
        "step_traces": step_traces,
        "fallback_count": fallback_count,
        "all_output_finite": all_finite and bool(torch.isfinite(x_t).all()),
        "protected_tokens_untouched": all(span_map[index].modality == "visual" and span_map[index].intervention_allowed for index in changed),
        "feature_names": list(FEATURES),
    }
