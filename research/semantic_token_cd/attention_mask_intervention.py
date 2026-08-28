"""Feature-preserving action-to-visual attention blocking for OpenVLA."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch


VISUAL_TOKEN_OFFSET = 1  # Prismatic inserts visual patches immediately after BOS.


@dataclass
class AttentionMaskTrace:
    layer_indices: tuple[int, ...]
    blocked_patch_indices: tuple[int, ...]
    blocked_key_indices: tuple[int, ...]
    action_query_start: int
    mask_value: float | None
    hook_calls: int = 0
    query_positions: set[int] = field(default_factory=set)
    calls_per_layer: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_indices": list(self.layer_indices),
            "blocked_patch_indices": list(self.blocked_patch_indices),
            "blocked_key_indices": list(self.blocked_key_indices),
            "action_query_start": self.action_query_start,
            "mask_value": self.mask_value,
            "hook_calls": self.hook_calls,
            "query_positions": sorted(self.query_positions),
            "calls_per_layer": {
                str(layer): count for layer, count in sorted(self.calls_per_layer.items())
            },
        }


def _llama_layers(model: Any) -> torch.nn.ModuleList:
    try:
        return model.language_model.model.layers
    except AttributeError as error:
        raise TypeError("Expected OpenVLA with a Llama language_model.model.layers stack") from error


@contextlib.contextmanager
def block_action_to_visual_attention(
    model: Any,
    selected_patch_indices: Sequence[int],
    action_query_start: int,
    layer_start: int = 16,
    layer_end: int | None = None,
    mask_value: float | None = None,
) -> Iterator[AttentionMaskTrace]:
    """Block selected visual keys only for action-producing query positions.

    The intervention is applied to the additive 4D causal mask received by
    selected Llama self-attention layers. Visual embeddings and every other
    query/key connection remain untouched.
    """
    patches = tuple(sorted(set(int(index) for index in selected_patch_indices)))
    if len(patches) != len(selected_patch_indices):
        raise ValueError("selected_patch_indices must be unique")
    if not patches or patches[0] < 0 or patches[-1] >= 256:
        raise ValueError(f"Expected non-empty visual patch indices in [0, 256), got {patches}")
    if action_query_start < 257:
        raise ValueError(f"Invalid action query start: {action_query_start}")

    layers = _llama_layers(model)
    end = len(layers) if layer_end is None else int(layer_end)
    indices = tuple(range(int(layer_start), end))
    if not indices or indices[0] < 0 or indices[-1] >= len(layers):
        raise ValueError(f"Invalid layer interval [{layer_start}, {end}) for {len(layers)} layers")
    implementation = getattr(model.language_model.config, "_attn_implementation", None)
    if implementation != "eager":
        raise RuntimeError(
            f"Attention-mask CD requires eager attention; model uses {implementation!r}"
        )

    keys = tuple(VISUAL_TOKEN_OFFSET + index for index in patches)
    effective_mask_value = None if mask_value is None else float(mask_value)
    trace = AttentionMaskTrace(
        indices, patches, keys, int(action_query_start), effective_mask_value
    )
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, args, kwargs):
            mask = kwargs.get("attention_mask")
            cache_position = kwargs.get("cache_position")
            if mask is None or mask.ndim != 4:
                raise RuntimeError(
                    f"Layer {layer_index} did not receive a 4D additive attention mask"
                )
            if cache_position is None or cache_position.ndim != 1:
                raise RuntimeError(f"Layer {layer_index} did not receive 1D cache_position")
            if mask.shape[-2] != cache_position.numel():
                raise RuntimeError(
                    f"Mask query length {mask.shape[-2]} != cache positions {cache_position.numel()}"
                )

            local_rows = torch.nonzero(
                cache_position >= action_query_start, as_tuple=False
            ).flatten()
            if local_rows.numel() == 0:
                return args, kwargs
            if keys[-1] >= mask.shape[-1]:
                raise RuntimeError(
                    f"Visual key {keys[-1]} outside attention key length {mask.shape[-1]}"
                )

            key_tensor = torch.tensor(keys, dtype=torch.long, device=mask.device)
            # Selected visual keys are in the causal past of every action query.
            original = mask[0, 0][local_rows[:, None], key_tensor[None, :]]
            if not torch.all(original == 0):
                raise RuntimeError("Attempted to block visual connections that were already masked")
            changed = mask.clone()
            written_value = (
                torch.finfo(mask.dtype).min if mask_value is None else mask_value
            )
            changed[0, 0][local_rows[:, None], key_tensor[None, :]] = written_value
            check = changed[0, 0][local_rows[:, None], key_tensor[None, :]]
            expected = torch.tensor(written_value, dtype=mask.dtype, device=mask.device)
            if not torch.all(check == expected):
                raise RuntimeError("Attention mask write was not consumed")
            kwargs["attention_mask"] = changed

            trace.hook_calls += 1
            trace.calls_per_layer[layer_index] = trace.calls_per_layer.get(layer_index, 0) + 1
            trace.query_positions.update(int(cache_position[row].item()) for row in local_rows)
            return args, kwargs

        return hook

    for index in indices:
        handles.append(
            layers[index].self_attn.register_forward_pre_hook(
                make_hook(index), with_kwargs=True
            )
        )
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class GlobalTokenMaskTrace:
    layer_indices: tuple[int, ...]
    blocked_patch_indices: tuple[int, ...]
    blocked_key_indices: tuple[int, ...]
    mask_value: float | None
    hook_calls: int = 0
    calls_per_layer: dict[int, int] = field(default_factory=dict)
    blocked_cell_counts: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_indices": list(self.layer_indices),
            "blocked_patch_indices": list(self.blocked_patch_indices),
            "blocked_key_indices": list(self.blocked_key_indices),
            "mask_value": self.mask_value,
            "hook_calls": self.hook_calls,
            "calls_per_layer": {
                str(layer): count for layer, count in sorted(self.calls_per_layer.items())
            },
            "blocked_cell_counts": list(self.blocked_cell_counts),
        }


@contextlib.contextmanager
def block_semantic_tokens_globally(
    model: Any,
    selected_patch_indices: Sequence[int],
    layer_start: int = 16,
    layer_end: int | None = None,
    mask_value: float | None = None,
) -> Iterator[GlobalTokenMaskTrace]:
    """Make selected visual tokens unreadable as key/value to every query.

    Unlike ``block_action_to_visual_attention`` (which only masks the Action
    Query -> selected visual key connections), this blocks the selected visual
    tokens' key/value columns for *all* query positions on the selected layers.
    Sequence length and positions are preserved; only the additive 4D causal
    mask is rewritten, so visual embeddings and every non-selected connection
    remain untouched.
    """
    patches = tuple(sorted(set(int(index) for index in selected_patch_indices)))
    if len(patches) != len(selected_patch_indices):
        raise ValueError("selected_patch_indices must be unique")
    if not patches or patches[0] < 0 or patches[-1] >= 256:
        raise ValueError(f"Expected non-empty visual patch indices in [0, 256), got {patches}")

    layers = _llama_layers(model)
    end = len(layers) if layer_end is None else int(layer_end)
    indices = tuple(range(int(layer_start), end))
    if not indices or indices[0] < 0 or indices[-1] >= len(layers):
        raise ValueError(f"Invalid layer interval [{layer_start}, {end}) for {len(layers)} layers")
    implementation = getattr(model.language_model.config, "_attn_implementation", None)
    if implementation != "eager":
        raise RuntimeError(
            f"Global token-mask CD requires eager attention; model uses {implementation!r}"
        )

    keys = tuple(VISUAL_TOKEN_OFFSET + index for index in patches)
    effective_mask_value = None if mask_value is None else float(mask_value)
    trace = GlobalTokenMaskTrace(indices, patches, keys, effective_mask_value)
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, args, kwargs):
            mask = kwargs.get("attention_mask")
            cache_position = kwargs.get("cache_position")
            if mask is None or mask.ndim != 4:
                raise RuntimeError(
                    f"Layer {layer_index} did not receive a 4D additive attention mask"
                )
            if cache_position is None or cache_position.ndim != 1:
                raise RuntimeError(f"Layer {layer_index} did not receive 1D cache_position")
            if mask.shape[-2] != cache_position.numel():
                raise RuntimeError(
                    f"Mask query length {mask.shape[-2]} != cache positions {cache_position.numel()}"
                )
            if keys[-1] >= mask.shape[-1]:
                raise RuntimeError(
                    f"Visual key {keys[-1]} outside attention key length {mask.shape[-1]}"
                )

            written_value = (
                torch.finfo(mask.dtype).min if mask_value is None else mask_value
            )
            expected = torch.tensor(written_value, dtype=mask.dtype, device=mask.device)
            pos = cache_position
            changed = mask.clone()
            blocked_cells = 0
            for key in keys:
                # A selected visual key is readable by any query at position >= key
                # (causal past). Block all such rows for this column.
                row_sel = pos >= key
                current = mask[0, 0][row_sel, key]
                if not torch.all(current == 0):
                    raise RuntimeError(
                        f"Layer {layer_index} key {key}: attempted to block an "
                        "already-masked or out-of-range connection"
                    )
                changed[0, 0][row_sel, key] = written_value
                blocked_cells += int(row_sel.sum().item())
            for key in keys:
                row_sel = pos >= key
                if not torch.all(changed[0, 0][row_sel, key] == expected):
                    raise RuntimeError("Global token mask write was not consumed")
            kwargs["attention_mask"] = changed

            trace.hook_calls += 1
            trace.calls_per_layer[layer_index] = trace.calls_per_layer.get(layer_index, 0) + 1
            trace.blocked_cell_counts.append(blocked_cells)
            return args, kwargs

        return hook

    for index in indices:
        handles.append(
            layers[index].self_attn.register_forward_pre_hook(
                make_hook(index), with_kwargs=True
            )
        )
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def validate_global_token_mask(
    trace: GlobalTokenMaskTrace,
    generated_score_count: int,
) -> None:
    """Fail closed when any intended generation step or layer missed the intervention."""
    expected_calls = generated_score_count * len(trace.layer_indices)
    if trace.hook_calls != expected_calls:
        raise RuntimeError(f"Global token-mask hook calls {trace.hook_calls} != expected {expected_calls}")
    for layer in trace.layer_indices:
        if trace.calls_per_layer.get(layer, 0) != generated_score_count:
            raise RuntimeError(
                f"Layer {layer} hook calls {trace.calls_per_layer.get(layer, 0)} "
                f"!= {generated_score_count}"
            )
    if not trace.blocked_cell_counts or any(count <= 0 for count in trace.blocked_cell_counts):
        raise RuntimeError("Global token mask never blocked any connection")


def validate_attention_trace(
    trace: AttentionMaskTrace,
    generated_score_count: int,
) -> None:
    """Fail closed when any intended action step or layer missed the intervention."""
    expected_positions = set(
        range(trace.action_query_start, trace.action_query_start + generated_score_count)
    )
    if trace.query_positions != expected_positions:
        raise RuntimeError(
            f"Blocked query positions {sorted(trace.query_positions)} != expected "
            f"{sorted(expected_positions)}"
        )
    expected_calls = generated_score_count * len(trace.layer_indices)
    if trace.hook_calls != expected_calls:
        raise RuntimeError(f"Attention hook calls {trace.hook_calls} != expected {expected_calls}")
    for layer in trace.layer_indices:
        if trace.calls_per_layer.get(layer, 0) != generated_score_count:
            raise RuntimeError(
                f"Layer {layer} hook calls {trace.calls_per_layer.get(layer, 0)} "
                f"!= {generated_score_count}"
            )
