"""Frozen SmolVLA teacher-forced probes for CoreAct development experiments."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


@dataclass(frozen=True)
class PrefixToken:
    index: int
    modality: str
    camera_id: str | None = None
    visual_token_index: int | None = None
    language_token_id: int | None = None
    decoded_language_piece: str | None = None
    is_special: bool = False
    is_padding: bool = False
    is_state: bool = False
    intervention_allowed: bool = False


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_prefix_span_map(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    camera_ids: Sequence[str] | None = None,
) -> list[list[PrefixToken]]:
    """Describe every prefix position using the verified embed_prefix order."""
    camera_ids = camera_ids or [f"camera{i + 1}" for i in range(len(images))]
    if len(camera_ids) != len(images) or len(images) != len(image_masks):
        raise ValueError("camera, image and image-mask counts must match")

    with torch.inference_mode():
        visual_counts = [model.vlm_with_expert.embed_image(image).shape[1] for image in images]
    tokenizer = model.vlm_with_expert.processor.tokenizer
    special_ids = set(tokenizer.all_special_ids)
    maps: list[list[PrefixToken]] = []
    for batch_index in range(lang_tokens.shape[0]):
        entries: list[PrefixToken] = []
        index = 0
        for camera_id, count, image_mask in zip(camera_ids, visual_counts, image_masks, strict=True):
            if model.add_image_special_tokens:
                for _ in range(2):
                    entries.append(
                        PrefixToken(index=index, modality="image_special", camera_id=camera_id, is_special=True)
                    )
                    index += 1
            camera_valid = bool(image_mask[batch_index])
            for visual_index in range(count):
                entries.append(
                    PrefixToken(
                        index=index,
                        modality="visual",
                        camera_id=camera_id,
                        visual_token_index=visual_index,
                        is_padding=not camera_valid,
                        intervention_allowed=camera_valid,
                    )
                )
                index += 1
            if model.add_image_special_tokens:
                entries.append(
                    PrefixToken(index=index, modality="image_special", camera_id=camera_id, is_special=True)
                )
                index += 1

        for token_id, valid in zip(
            lang_tokens[batch_index].tolist(), lang_masks[batch_index].tolist(), strict=True
        ):
            special = token_id in special_ids
            entries.append(
                PrefixToken(
                    index=index,
                    modality="language",
                    language_token_id=token_id,
                    decoded_language_piece=tokenizer.decode([token_id], skip_special_tokens=False),
                    is_special=special,
                    is_padding=not bool(valid),
                    intervention_allowed=bool(valid) and not special,
                )
            )
            index += 1

        entries.append(
            PrefixToken(index=index, modality="state", is_state=True, intervention_allowed=False)
        )
        index += 1
        while index < prefix_pad_masks.shape[1]:
            entries.append(PrefixToken(index=index, modality="prefix_padding", is_padding=True))
            index += 1

        if len(entries) != prefix_pad_masks.shape[1]:
            raise AssertionError(
                f"span map length {len(entries)} does not match prefix {prefix_pad_masks.shape[1]}"
            )
        for entry in entries:
            if entry.intervention_allowed and (entry.is_special or entry.is_padding or entry.is_state):
                raise AssertionError(f"protected prefix token marked eligible: {entry}")
            if bool(prefix_pad_masks[batch_index, entry.index]) == entry.is_padding:
                raise AssertionError(f"prefix pad-mask mismatch at index {entry.index}")
        maps.append(entries)
    return maps


def validate_intervention_group(span_map: Sequence[PrefixToken], indices: Sequence[int]) -> None:
    for index in indices:
        if index < 0 or index >= len(span_map):
            raise IndexError(f"prefix index {index} is outside the span map")
        if not span_map[index].intervention_allowed:
            raise ValueError(f"prefix index {index} is protected ({span_map[index].modality})")


def grouped_intervention(
    prefix_embeddings: torch.Tensor,
    span_map: Sequence[PrefixToken],
    groups: Sequence[Sequence[int]],
    replacements: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Build one batch row per group without changing prefix shape or ordering."""
    if prefix_embeddings.shape[0] != 1:
        raise ValueError("grouped_intervention currently requires one source state")
    output = prefix_embeddings.expand(len(groups), -1, -1).clone()
    for row, group in enumerate(groups):
        validate_intervention_group(span_map, group)
        for index in group:
            if index not in replacements:
                raise KeyError(f"missing replacement for prefix index {index}")
            replacement = replacements[index].to(device=output.device, dtype=output.dtype)
            if replacement.shape != output[row, index].shape:
                raise ValueError(f"replacement shape mismatch for prefix index {index}")
            output[row, index] = replacement
    return output


def _repeat_batch(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.shape[0] == size:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"cannot expand batch size {tensor.shape[0]} to {size}")
    return tensor.expand(size, *tensor.shape[1:])


def predict_teacher_forced_velocity(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    tau: torch.Tensor,
    noise: torch.Tensor,
    prefix_intervention: Callable[[torch.Tensor, list[list[PrefixToken]]], torch.Tensor] | None = None,
    record_attention: bool = False,
    camera_ids: Sequence[str] | None = None,
) -> dict:
    """Run the exact SmolVLA training path while exposing velocity and prefix metadata."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("teacher-forced probe requires a frozen model in eval mode")
    if actions.shape != noise.shape or tau.shape != (actions.shape[0],):
        raise ValueError("actions/noise/tau shapes are inconsistent")

    time_expanded = tau[:, None, None]
    x_tau = time_expanded * noise + (1 - time_expanded) * actions
    u_target = noise - actions
    prefix_before, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks, state=state
    )
    span_maps = build_prefix_span_map(
        model,
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        prefix_pad_masks,
        camera_ids=camera_ids,
    )
    prefix_after = (
        prefix_before
        if prefix_intervention is None
        else prefix_intervention(prefix_before.clone(), span_maps)
    )
    if prefix_after.ndim != 3 or prefix_after.shape[1:] != prefix_before.shape[1:]:
        raise ValueError("prefix intervention changed sequence length or embedding width")

    batch_size = prefix_after.shape[0]
    x_tau = _repeat_batch(x_tau, batch_size)
    tau = _repeat_batch(tau[:, None], batch_size)[:, 0]
    u_target = _repeat_batch(u_target, batch_size)
    prefix_pad_masks = _repeat_batch(prefix_pad_masks, batch_size)
    prefix_att_masks = _repeat_batch(prefix_att_masks, batch_size)
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_tau, tau)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    attention_mask = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    traces: list[dict] = []
    trace_owner = model.vlm_with_expert
    previous_callback = getattr(trace_owner, "attention_trace_callback", None)
    trace_owner.attention_trace_callback = traces.append if record_attention else None
    try:
        (_, suffix_out), _ = trace_owner.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_after, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
    finally:
        trace_owner.attention_trace_callback = previous_callback
    suffix_out = suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)
    velocity = model.action_out_proj(suffix_out)

    changed = (prefix_before[:1] != prefix_after[:1]).any(dim=-1).nonzero(as_tuple=False)
    return {
        "v_pred": velocity,
        "u_target": u_target,
        "x_tau": x_tau,
        "prefix_embeddings_before": prefix_before,
        "prefix_embeddings_after": prefix_after,
        "prefix_summary": {
            "before_shape": list(prefix_before.shape),
            "after_shape": list(prefix_after.shape),
            "before_sha256": _tensor_sha256(prefix_before),
            "after_sha256": _tensor_sha256(prefix_after),
            "changed_indices_first_row": changed[:, 1].tolist(),
        },
        "prefix_span_map": [[asdict(token) for token in tokens] for tokens in span_maps],
        "attention_trace": traces,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def attention_ranking_scores(traces: Sequence[dict], prefix_length: int) -> dict[str, torch.Tensor]:
    """Aggregate action-query probabilities without treating attention as attribution."""
    if not traces:
        raise ValueError("no actual expert-layer attention traces were recorded")
    ordered = sorted(traces, key=lambda item: item["layer_index"])
    layer_scores = [item["probabilities"].mean(dim=(0, 1, 2))[:prefix_length] for item in ordered]
    last = layer_scores[-1]
    late_start = len(layer_scores) // 2
    late = torch.stack(layer_scores[late_start:]).mean(dim=0)
    output = {
        "raw_last_expert_layer_attention": last,
        "late_half_action_to_context_attention": late,
    }
    if all("value_weighted_scores" in item for item in ordered):
        value_scores = [item["value_weighted_scores"][:prefix_length] for item in ordered]
        output["raw_last_expert_layer_value_weighted_attention"] = value_scores[-1]
        output["late_half_value_weighted_attention"] = torch.stack(value_scores[late_start:]).mean(dim=0)
    return output
