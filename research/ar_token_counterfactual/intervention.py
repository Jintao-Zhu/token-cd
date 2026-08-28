"""Post-projector intervention and AR teacher-forced analysis primitives."""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass
class ProjectorTrace:
    before: torch.Tensor | None = None
    after: torch.Tensor | None = None
    changed_indices: tuple[int, ...] = ()


@contextlib.contextmanager
def projector_intervention(
    model: Any,
    selected_indices: Sequence[int] = (),
    replacement_mean: torch.Tensor | None = None,
) -> Iterator[ProjectorTrace]:
    """Capture projector output and optionally replace exact visual-token rows."""
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if len(selected) != len(selected_indices):
        raise ValueError("selected_indices must be unique")
    if selected and replacement_mean is None:
        raise ValueError("replacement_mean is required for a non-empty intervention")
    trace = ProjectorTrace()

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        if output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError(f"Expected projector output [1, tokens, dim], got {tuple(output.shape)}")
        if selected and (selected[0] < 0 or selected[-1] >= output.shape[1]):
            raise IndexError(f"Visual token index outside [0, {output.shape[1]}): {selected}")
        trace.before = output.detach().float().cpu().clone()
        if not selected:
            trace.after = trace.before.clone()
            return None

        mean = replacement_mean
        if mean is None:
            raise AssertionError("unreachable")
        if mean.ndim == 3:
            if mean.shape[0] != 1:
                raise ValueError(f"Expected replacement batch size 1, got {tuple(mean.shape)}")
            mean = mean[0]
        if mean.shape != output.shape[1:]:
            raise ValueError(f"Replacement shape {tuple(mean.shape)} != projector shape {tuple(output.shape[1:])}")
        changed = output.clone()
        index_tensor = torch.tensor(selected, device=output.device, dtype=torch.long)
        mean_rows = mean[torch.tensor(selected, device=mean.device, dtype=torch.long)]
        changed[:, index_tensor] = mean_rows.to(device=output.device, dtype=output.dtype)
        trace.after = changed.detach().float().cpu().clone()
        row_changed = torch.any(trace.before != trace.after, dim=-1)[0]
        trace.changed_indices = tuple(torch.nonzero(row_changed, as_tuple=False).flatten().tolist())
        if trace.changed_indices != selected:
            raise RuntimeError(f"Changed projector indices {trace.changed_indices} != selected indices {selected}")
        return changed

    handle = model.projector.register_forward_hook(hook)
    try:
        yield trace
    finally:
        handle.remove()


def ensure_empty_action_token(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.all(input_ids[:, -1] == 29871):
        return input_ids, attention_mask
    empty = torch.full((input_ids.shape[0], 1), 29871, dtype=input_ids.dtype, device=input_ids.device)
    visible = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([input_ids, empty], dim=1), torch.cat([attention_mask, visible], dim=1)


@torch.inference_mode()
def clean_action_token_ids(model: Any, inputs: dict[str, torch.Tensor], action_dim: int = 7) -> torch.Tensor:
    input_ids, attention_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs["pixel_values"],
        max_new_tokens=action_dim,
        do_sample=False,
    )
    return generated[:, -action_dim:]


def decode_action_ids(model: Any, action_ids: torch.Tensor, unnorm_key: str = "libero_spatial") -> np.ndarray:
    token_ids = action_ids[0].detach().cpu().numpy()
    discretized = model.vocab_size - token_ids
    discretized = np.clip(discretized - 1, 0, model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


@torch.inference_mode()
def masked_action_token_ids(
    model: Any,
    inputs: dict[str, torch.Tensor],
    selected_indices: Sequence[int],
    replacement_mean: torch.Tensor,
    action_dim: int = 7,
) -> tuple[torch.Tensor, ProjectorTrace]:
    input_ids, attention_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    with projector_intervention(model, selected_indices, replacement_mean) as trace:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=action_dim,
            do_sample=False,
        )
    return generated[:, -action_dim:], trace


@dataclass
class TeacherForcedOutput:
    logits: torch.Tensor
    attention_scores: torch.Tensor | None
    visual_token_count: int
    action_query_indices: tuple[int, ...]
    trace: ProjectorTrace


@torch.inference_mode()
def teacher_forced_forward(
    model: Any,
    inputs: dict[str, torch.Tensor],
    clean_action_ids: torch.Tensor,
    selected_indices: Sequence[int] = (),
    replacement_mean: torch.Tensor | None = None,
    record_attention: bool = False,
) -> TeacherForcedOutput:
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    teacher_ids = torch.cat([base_ids, clean_action_ids[:, :-1]], dim=1)
    teacher_mask = torch.cat(
        [base_mask, torch.ones_like(clean_action_ids[:, :-1], dtype=base_mask.dtype, device=base_mask.device)], dim=1
    )
    with projector_intervention(model, selected_indices, replacement_mean) as trace:
        output = model(
            input_ids=teacher_ids,
            attention_mask=teacher_mask,
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            output_attentions=record_attention,
            return_dict=True,
        )
    if trace.before is None:
        raise RuntimeError("Projector hook was not invoked")
    visual_count = trace.before.shape[1]
    action_queries = tuple(visual_count + base_ids.shape[1] - 1 + offset for offset in range(clean_action_ids.shape[1]))
    logits = output.logits[:, list(action_queries)].detach().float().cpu()

    attention_scores = None
    if record_attention:
        attentions = output.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attentions")
        late_half = attentions[len(attentions) // 2 :]
        per_layer = []
        for attention in late_half:
            selected_attention = attention[:, :, list(action_queries), 1 : 1 + visual_count]
            per_layer.append(selected_attention.detach().float().cpu())
        attention_scores = torch.stack(per_layer).mean(dim=(0, 1, 2, 3))

    return TeacherForcedOutput(logits, attention_scores, visual_count, action_queries, trace)


def action_logit_metrics(clean_logits: torch.Tensor, masked_logits: torch.Tensor) -> dict[str, np.ndarray]:
    clean_log_prob = torch.log_softmax(clean_logits.float(), dim=-1)
    masked_log_prob = torch.log_softmax(masked_logits.float(), dim=-1)
    clean_prob = clean_log_prob.exp()
    masked_prob = masked_log_prob.exp()
    mixture = 0.5 * (clean_prob + masked_prob)
    mixture_log = torch.log(mixture.clamp_min(torch.finfo(mixture.dtype).tiny))
    js = 0.5 * (
        torch.sum(clean_prob * (clean_log_prob - mixture_log), dim=-1)
        + torch.sum(masked_prob * (masked_log_prob - mixture_log), dim=-1)
    )
    kl = torch.sum(clean_prob * (clean_log_prob - masked_log_prob), dim=-1)
    clean_top2 = torch.topk(clean_logits, 2, dim=-1).values
    masked_top2 = torch.topk(masked_logits, 2, dim=-1).values
    return {
        "js_div": js.cpu().numpy(),
        "kl_clean_mask": kl.cpu().numpy(),
        "argmax_flip": (clean_logits.argmax(-1) != masked_logits.argmax(-1)).cpu().numpy(),
        "clean_margin": (clean_top2[..., 0] - clean_top2[..., 1]).cpu().numpy(),
        "masked_margin": (masked_top2[..., 0] - masked_top2[..., 1]).cpu().numpy(),
    }


def select_visual_tokens(attention_scores: torch.Tensor, random_seed: int) -> dict[str, list[int]]:
    scores = attention_scores.detach().float().cpu().numpy()
    if scores.ndim != 1 or not np.isfinite(scores).all() or scores.size < 16:
        raise ValueError(f"Invalid attention scores: shape={scores.shape}")
    order = np.argsort(scores, kind="stable")
    bottom = order[:4].tolist()
    top = order[-4:][::-1].tolist()
    middle_start = scores.size // 2 - 2
    middle = order[middle_start : middle_start + 4].tolist()
    excluded = set(top + middle + bottom)
    candidates = np.array([index for index in range(scores.size) if index not in excluded])
    random = np.random.default_rng(random_seed).choice(candidates, size=4, replace=False).tolist()
    selected = top + middle + bottom + random
    if len(set(selected)) != 16:
        raise RuntimeError("Token selection categories overlap")
    return {"attention_top": top, "attention_middle": middle, "attention_bottom": bottom, "deterministic_random": random}
