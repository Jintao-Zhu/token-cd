"""Extract a fixed, contextual SmolVLA state representation without changing policy behavior."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from research.coreact_exploration.instrumentation import PrefixToken, build_prefix_span_map


POOL_ORDER = ("camera1", "camera2", "language", "state")


def contextual_prefix(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[list[PrefixToken]]]:
    """Return final-normalized prefix hidden states and their verified span map."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("value representation requires a frozen model in eval mode")
    prefix, pad_mask, att_mask = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks, state=state
    )
    span_maps = build_prefix_span_map(
        model,
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        pad_mask,
        camera_ids=("camera1", "camera2"),
    )
    hidden = contextualize_embedded_prefix(model, prefix, pad_mask, att_mask)
    return hidden, pad_mask, span_maps


def contextualize_embedded_prefix(
    model,
    prefix: torch.Tensor,
    pad_mask: torch.Tensor,
    att_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the native prefix transformer on clean or intervened prefix embeddings."""
    attention_mask = make_att_2d_masks(pad_mask, att_mask)
    position_ids = torch.cumsum(pad_mask, dim=1) - 1
    outputs, _ = model.vlm_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=False,
        # This is the checkpoint-native prefix-only branch used by sample_actions().
        # No cache is retained because use_cache is false.
        fill_kv_cache=True,
    )
    hidden = outputs[0].to(dtype=torch.float32)
    if hidden.shape[:2] != pad_mask.shape or not torch.isfinite(hidden).all():
        raise RuntimeError("invalid contextual prefix representation")
    return hidden


def _pool_indices(hidden: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        raise RuntimeError("representation pool has no valid tokens")
    return hidden[:, indices].mean(dim=1)


def pool_contextual_prefix(
    hidden: torch.Tensor,
    span_maps: list[list[PrefixToken]],
) -> tuple[torch.Tensor, dict[str, list[int]]]:
    """Mean-pool camera/language/state spans separately, then concatenate."""
    if hidden.shape[0] != len(span_maps):
        raise ValueError("hidden batch and span-map batch differ")
    rows = []
    first_indices: dict[str, list[int]] = {}
    for batch_index, span_map in enumerate(span_maps):
        pools = {
            "camera1": [t.index for t in span_map if t.modality == "visual" and t.camera_id == "camera1" and not t.is_padding],
            "camera2": [t.index for t in span_map if t.modality == "visual" and t.camera_id == "camera2" and not t.is_padding],
            "language": [t.index for t in span_map if t.modality == "language" and not t.is_padding],
            "state": [t.index for t in span_map if t.modality == "state"],
        }
        if batch_index == 0:
            first_indices = pools
        rows.append(torch.cat([_pool_indices(hidden[batch_index : batch_index + 1], pools[name]) for name in POOL_ORDER], dim=-1))
    representation = torch.cat(rows, dim=0)
    if not torch.isfinite(representation).all():
        raise RuntimeError("pooled representation contains NaN/Inf")
    return representation, first_indices


@torch.inference_mode()
def extract_value_representation(model, prepared: dict) -> tuple[torch.Tensor, dict[str, list[int]]]:
    hidden, _, span_maps = contextual_prefix(
        model,
        prepared["images"],
        prepared["image_masks"],
        prepared["lang_tokens"],
        prepared["lang_masks"],
        prepared["state"],
    )
    return pool_contextual_prefix(hidden, span_maps)
