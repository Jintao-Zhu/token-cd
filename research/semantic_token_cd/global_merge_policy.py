"""Global Spatial Merge CD (GSM-CD, Phase 1A) — guided-prefix contrastive decoding.

Negative branch = *coarse vision* obtained by 2x2 block-mean smoothing of the
projector output V in R^{16x16xd} (256 tokens = 16x16 patch grid). Token count is
UNCHANGED (256 -> 256); each token i in block B_k becomes

    v_i -> (1 - eta) * v_i + eta * mu_k,   mu_k = (1/4) sum_{j in B_k} v_j

for eta in {0.25, 0.5, 1.0}. This is a fine-to-coarse visual degradation (local
4-token smoothing), NOT object-absence inpainting.

Guided CD (matching the Phase-1 spec): both branches share the SAME
autoregressive prefix a_{<q}^* = the CLEAN branch's greedy tokens. The negative
branch is teacher-forced on that prefix (not independently greedy-decoded), so
the residual r_q = z_q^+ - z_q^- isolates the visual merge alone.

    z_q^+ = F(V,   a_{<q}^clean)    (greedy, self-consistent)
    z_q^- = F(V~,  a_{<q}^clean)    (teacher-forced)
    z_q^* = z_q^+ + 0.5 * (z_q^+ - z_q^-)   for action dims 0..5
    z_6^* = z_6^+                           (gripper keeps clean)

``GuidedAttentionCDInference`` is the *baseline* arm: identical guided-prefix CD,
but the negative branch is built by the Oracle-GT-mask attention blocking
(layers [8,16), mask -1e4) instead of feature merging, so the only difference
between the two CD arms is *how the negative branch is built*.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import (
    AuditedEntityCDInference,
    _action_logits,
)
from research.semantic_token_cd.oracle_mask_policy import OracleAttentionCDInference


VISUAL_TOKEN_COUNT = 256  # SigLIP 224/14 -> 16x16 patches


def validate_guided_attention_trace(trace, n_action_tokens: int) -> None:
    """Fail closed for the single-pass (teacher-forced) guided attention forward.

    ``GuidedAttentionCDInference`` inherits ``_attention_intervention_context``
    from ``AttentionMaskEntityCDInference`` (``block_action_to_visual_attention``),
    which yields an ``AttentionMaskTrace``. In the guided forward the full teacher
    sequence is processed in ONE ``use_cache=False`` pass, so the hook blocks ALL
    ``n_action_tokens`` action-query positions at once and fires once per layer
    (unlike the 7-step ``generate`` loop, where it fires once per step).
    """
    expected_positions = set(range(trace.action_query_start, trace.action_query_start + n_action_tokens))
    if trace.query_positions != expected_positions:
        raise RuntimeError(
            f"Blocked query positions {sorted(trace.query_positions)} != expected {sorted(expected_positions)}"
        )
    if trace.hook_calls != len(trace.layer_indices):
        raise RuntimeError(
            f"Guided attention hook calls {trace.hook_calls} != layers {len(trace.layer_indices)}"
        )
    for layer in trace.layer_indices:
        if trace.calls_per_layer.get(layer, 0) != 1:
            raise RuntimeError(f"Layer {layer} hook calls {trace.calls_per_layer.get(layer, 0)} != 1")


def merge_2x2(V: torch.Tensor, eta: float) -> torch.Tensor:
    """2x2 block-mean merge on the 16x16 grid: v_i -> (1-eta) v_i + eta * mu_block.

    ``V`` is the projector output [1, 256, d] (float32, any device). Returns the
    merged features with the same shape/dtype. Token index = row*16 + col
    (row-major), matching the mask_to_overlap / grid-mask conventions.
    """
    if V.ndim != 3 or V.shape[1] != VISUAL_TOKEN_COUNT:
        raise ValueError(f"Expected V [1,256,d], got {tuple(V.shape)}")
    b = V.reshape(1, 8, 2, 8, 2, -1)  # [1, block_row, i, block_col, j, d]
    mu = b.mean(dim=(2, 4), keepdim=True)  # [1, br, 1, bc, 1, d]
    return ((1.0 - eta) * b + eta * mu).reshape(1, VISUAL_TOKEN_COUNT, -1)


@contextlib.contextmanager
def projector_merge_intervention(
    model: Any, merged_features: torch.Tensor
) -> Iterator[dict[str, torch.Tensor | None]]:
    """Replace the entire projector output with merged_features [1,256,d].

    Captures ``before`` (the untouched projector output) and ``after`` (the
    bf16-quantized merged tensor) so the caller can assert that the negative
    forward saw the same clean V before the merge was applied.
    """
    trace: dict[str, torch.Tensor | None] = {"before": None, "after": None}

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        if output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError(f"Expected projector output [1,tokens,dim], got {tuple(output.shape)}")
        merged = merged_features
        if merged.ndim == 2:
            merged = merged.unsqueeze(0)
        if merged.shape != output.shape:
            raise ValueError(f"Merged shape {tuple(merged.shape)} != projector shape {tuple(output.shape)}")
        trace["before"] = output.detach().float().cpu().clone()
        changed = merged.to(device=output.device, dtype=output.dtype)
        trace["after"] = changed.detach().float().cpu().clone()
        return changed

    handle = model.projector.register_forward_hook(hook)
    try:
        yield trace
    finally:
        handle.remove()


@torch.inference_mode()
def guided_forward_scores(
    model: Any,
    inputs: dict[str, torch.Tensor],
    clean_token_ids: torch.Tensor,
    visual_count: int = VISUAL_TOKEN_COUNT,
) -> torch.Tensor:
    """Teacher-forced forward guided by the clean greedy prefix.

    Feeds [prompt, empty-token, clean a_0..a_{n-2}] and reads the logits at the
    n action-query positions, so the returned logits row q is F(., a_{<q}^clean).
    Returns a [n, vocab] float32 CPU tensor. The caller is expected to wrap this
    call in the negative intervention context manager (merge or attention).
    """
    from research.ar_token_counterfactual.intervention import ensure_empty_action_token

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    base_ids, base_mask = ensure_empty_action_token(input_ids, attention_mask)
    prefix = clean_token_ids[:-1].unsqueeze(0)  # [1, n-1]
    teacher_ids = torch.cat([base_ids, prefix], dim=1)
    teacher_mask = torch.cat(
        [base_mask, torch.ones_like(prefix, dtype=base_mask.dtype, device=base_mask.device)], dim=1
    )
    output = model(
        input_ids=teacher_ids,
        attention_mask=teacher_mask,
        pixel_values=inputs["pixel_values"],
        use_cache=False,
        return_dict=True,
    )
    n_actions = int(clean_token_ids.shape[0])
    action_queries = [visual_count + base_ids.shape[1] - 1 + offset for offset in range(n_actions)]
    # Keep on the model device to match ``_forward_scores`` (which returns the
    # ``generate`` scores on cuda). The CD combine requires both branches on the
    # same device; downstream ``_action_logits``/``argmax`` handle cuda fine.
    return output.logits[:, action_queries].detach().float()[0]  # [n, vocab] cuda


def _action_query_start(inputs: dict[str, torch.Tensor], visual_count: int) -> int:
    input_ids = inputs["input_ids"]
    appended_empty = int(not torch.all(input_ids[:, -1] == 29871))
    return visual_count + int(input_ids.shape[1]) + appended_empty - 1


class GuidedMergeCDInference(AuditedEntityCDInference):
    """Global 2x2 spatial merge CD with a guided autoregressive prefix."""

    eta: float

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("Positive branch did not produce 7 scores and projector features")

        V = positive_trace.before  # [1,256,d] float32 cpu
        clean_token_ids = clean_scores.argmax(dim=-1)  # [7]

        V_tilde = merge_2x2(V, self.eta)
        with projector_merge_intervention(self.vla, V_tilde) as negative_trace:
            negative_scores = guided_forward_scores(self.vla, inputs, clean_token_ids, V.shape[1])
        if negative_trace["before"] is None or negative_trace["after"] is None:
            raise RuntimeError("Merge hook was not invoked")
        feature_equal = torch.equal(V, negative_trace["before"])
        if not feature_equal:
            raise RuntimeError("Negative branch projector 'before' != clean V")

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1] - self.lambd * negative_scores[:-1]
        )
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite GSM-CD logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        residual = (
            torch.log_softmax(torch.from_numpy(positive).float(), dim=-1)
            - torch.log_softmax(torch.from_numpy(negative).float(), dim=-1)
        )
        meta = {
            "eta": float(self.eta),
            "positive_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "residual_norm": float(torch.linalg.vector_norm(residual).item()),
            "lambda": float(self.lambd),
            "feature_equal": feature_equal,
            "feature_shape": list(V.shape),
            "n_tokens_negative": VISUAL_TOKEN_COUNT,
            "guided_prefix": True,
            "degenerate": False,
        }
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        return raw_action, actions, meta


class GuidedAttentionCDInference(OracleAttentionCDInference):
    """Oracle semantic attention CD with the same guided autoregressive prefix.

    Identical guided-prefix CD to ``GuidedMergeCDInference``; the negative branch
    is produced by Oracle-GT-mask attention blocking (layers [8,16), mask -1e4)
    instead of feature merging.
    """

    def _clean_action(self, image, task_description, inputs):
        """Empty-oracle fallback: emit the clean action for this step."""
        with projector_intervention(self.vla) as trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7:
            raise RuntimeError(f"Expected 7 clean action logits, got {clean_scores.shape[0]}")
        token_ids = clean_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        positive = _action_logits(self, clean_scores)
        self._episode_logits.append({"positive": positive, "negative": positive})
        meta = {
            "oracle_empty": True,
            "oracle_mode": self.oracle_mode,
            "oracle_tau": self._oracle_tau,
            "lambda": float(self.lambd),
            "degenerate": True,
            "n_selected": 0,
            "guided_prefix": True,
            "clean_token_ids": token_ids.detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
        }
        self._episode_trace.append(meta)
        return raw_action, actions, meta

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        overlap = self._oracle_overlap
        if self.oracle_mode == "full" and overlap is not None and not np.any(overlap > self._oracle_tau):
            return self._clean_action(image, task_description, inputs)

        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("Positive branch did not produce 7 scores and projector features")

        V = positive_trace.before
        clean_token_ids = clean_scores.argmax(dim=-1)
        selected, meta = self._select(V[0].numpy())
        if not selected:
            raise RuntimeError("Oracle selector produced an empty negative branch")

        action_query_start = _action_query_start(inputs, V.shape[1])
        with projector_intervention(self.vla) as negative_trace:
            with self._attention_intervention_context(selected, action_query_start) as attention_trace:
                negative_scores = guided_forward_scores(self.vla, inputs, clean_token_ids, V.shape[1])
        if negative_trace.before is None:
            raise RuntimeError("Negative branch projector hook was not invoked")
        feature_equal = torch.equal(V, negative_trace.before)
        if not feature_equal:
            raise RuntimeError("Positive and negative visual features are not bit-identical")

        validate_guided_attention_trace(attention_trace, negative_scores.shape[0])
        meta["attention_mask"] = attention_trace.as_dict()

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1] - self.lambd * negative_scores[:-1]
        )
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite guided Attention-CD logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        residual = (
            torch.log_softmax(torch.from_numpy(positive).float(), dim=-1)
            - torch.log_softmax(torch.from_numpy(negative).float(), dim=-1)
        )
        meta.update({
            "positive_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "residual_norm": float(torch.linalg.vector_norm(residual).item()),
            "lambda": float(self.lambd),
            "feature_equal": feature_equal,
            "feature_shape": list(V.shape),
            "guided_prefix": True,
            "degenerate": False,
        })
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        return raw_action, actions, meta
