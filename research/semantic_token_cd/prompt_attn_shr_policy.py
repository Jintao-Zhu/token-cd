"""Prompt-Attn-SHR v1 selector on top of the locked beta=0 SHR operator.

Only the identity of the selected visual tokens changes.  Coverage is always
the size of the canonical KMeans/entity selector on the arm's own observation.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from transformers.generation.logits_process import LogitsProcessorList, SuppressTokensLogitsProcessor

from research.ar_token_counterfactual.intervention import (
    ensure_empty_action_token,
    projector_intervention,
)
from research.semantic_token_cd.distractor_policy import ACTION_VOCAB_SIZE, _action_logits
from research.semantic_token_cd.global_merge_policy import (
    guided_forward_scores,
    projector_merge_intervention,
)
from research.semantic_token_cd.st_shr_policy import STSHRCDInference, harmonic_reconstruct


N_VISUAL = 256
LAYER_START = 16
LAYER_END = 32
RANDOM_SALT = 0x50A77E11
EPS = 1e-12


def mask_spatial_stats(selected: list[int]) -> tuple[int, float]:
    """Return 4-neighbor component count and isolated-token ratio on 16x16."""
    remaining = set(int(index) for index in selected)
    components = 0
    isolated = 0
    while remaining:
        components += 1
        start = remaining.pop()
        stack = [start]
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            row, col = divmod(index, 16)
            neighbors = []
            if row > 0:
                neighbors.append(index - 16)
            if row < 15:
                neighbors.append(index + 16)
            if col > 0:
                neighbors.append(index - 1)
            if col < 15:
                neighbors.append(index + 1)
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        isolated += int(size == 1)
    return components, isolated / max(1, len(selected))


def stable_top_m(scores: np.ndarray, m: int) -> list[int]:
    """Descending score with ascending token index as the exact tie-break."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (N_VISUAL,) or not np.isfinite(values).all():
        raise ValueError(f"invalid prompt attention scores: {values.shape}")
    if not 1 <= m <= N_VISUAL:
        raise ValueError(f"invalid coverage m={m}")
    order = np.lexsort((np.arange(N_VISUAL), -values))
    return sorted(int(index) for index in order[:m])


def stable_rank(scores: np.ndarray, indices: list[int], descending: bool = True) -> list[int]:
    """Rank a candidate subset with ascending token id as the tie-break."""
    values = np.asarray(scores, dtype=np.float64)
    candidates = np.asarray(indices, dtype=np.int64)
    if values.shape != (N_VISUAL,) or not np.isfinite(values).all():
        raise ValueError(f"invalid attention scores: {values.shape}")
    if candidates.ndim != 1 or len(set(candidates.tolist())) != len(candidates):
        raise ValueError("candidate indices must be one-dimensional and unique")
    primary = -values[candidates] if descending else values[candidates]
    order = np.lexsort((candidates, primary))
    return [int(value) for value in candidates[order]]


def construct_prompt_action_masks(
    prompt_scores: np.ndarray, action_scores: np.ndarray, m: int
) -> dict[str, dict]:
    """Construct the four locked Prompt/Action complement masks for one state."""
    original = stable_top_m(prompt_scores, m)
    r = int(m // 4)
    core = stable_top_m(prompt_scores, m - r)
    core_set = set(core)
    prompt_order = stable_rank(
        prompt_scores, [index for index in range(N_VISUAL) if index not in core_set], True
    )
    split = (len(prompt_order) + 1) // 2
    candidate_high, candidate_low = prompt_order[:split], prompt_order[split:]
    if r > len(candidate_high) or r > len(candidate_low):
        raise RuntimeError("candidate pool too small for complement supplement")
    supplements = {
        "original": sorted(set(original) - core_set),
        "prompt_high_action_high": stable_rank(action_scores, candidate_high, True)[:r],
        "prompt_high_action_low": stable_rank(action_scores, candidate_high, False)[:r],
        "prompt_low_action_high": stable_rank(action_scores, candidate_low, True)[:r],
    }
    prompt_global = stable_rank(prompt_scores, list(range(N_VISUAL)), True)
    action_global = stable_rank(action_scores, list(range(N_VISUAL)), True)
    prompt_rank = {token: rank + 1 for rank, token in enumerate(prompt_global)}
    action_rank = {token: rank + 1 for rank, token in enumerate(action_global)}
    result = {}
    for arm, supplement in supplements.items():
        selected = original if arm == "original" else sorted(core + supplement)
        expected_pool = (candidate_low if arm == "prompt_low_action_high" else candidate_high)
        pool_ok = arm == "original" or all(token in expected_pool for token in supplement)
        if len(selected) != m or len(set(selected)) != m or not core_set.issubset(selected):
            raise RuntimeError(f"invalid Prompt/Action complement mask for {arm}")
        result[arm] = {
            "selected": selected,
            "supplement": sorted(supplement),
            "supplement_count_r": r,
            "core": sorted(core),
            "candidate_high": sorted(candidate_high),
            "candidate_low": sorted(candidate_low),
            "actual_replacements_vs_original": len(set(selected) - set(original)),
            "supplement_prompt_scores": [float(prompt_scores[x]) for x in supplement],
            "supplement_action_scores": [float(action_scores[x]) for x in supplement],
            "supplement_prompt_global_ranks": [prompt_rank[x] for x in supplement],
            "supplement_action_global_ranks": [action_rank[x] for x in supplement],
            "core_exact": core_set.issubset(selected),
            "candidate_pool_membership_exact": pool_ok,
        }
    return result


def construct_prompt_action_rerank(
    prompt_scores: np.ndarray,
    action_scores: np.ndarray,
    m: int,
    candidate_multiplier: int = 3,
) -> dict:
    """Prompt Top-candidate pool followed by Action-attention Top-m reranking.

    Both ranking stages use the repository-wide deterministic tie-break:
    descending score, then ascending visual-token id.  The returned final mask
    is sorted spatially for the harmonic reconstruction code path.
    """
    if not 2 <= int(candidate_multiplier) <= 8:
        raise ValueError(f"invalid candidate multiplier: {candidate_multiplier}")
    if not 1 <= int(m) <= N_VISUAL:
        raise ValueError(f"invalid coverage m={m}")
    original = stable_top_m(prompt_scores, int(m))
    raw_candidate_count = int(candidate_multiplier) * int(m)
    candidate_count = min(N_VISUAL, raw_candidate_count)
    candidates = stable_top_m(prompt_scores, candidate_count)
    selected = sorted(stable_rank(action_scores, candidates, descending=True)[: int(m)])
    if len(selected) != int(m) or len(set(selected)) != int(m):
        raise RuntimeError("Prompt/Action rerank failed exact matched coverage")
    if not set(selected).issubset(candidates):
        raise RuntimeError("Prompt/Action rerank selected outside Prompt candidate pool")
    overlap = len(set(selected) & set(original))
    union = len(set(selected) | set(original))
    return {
        "selected": selected,
        "original_prompt": original,
        "candidates": candidates,
        "candidate_multiplier": int(candidate_multiplier),
        "raw_candidate_count": raw_candidate_count,
        "candidate_count": candidate_count,
        "candidate_pool_saturated": raw_candidate_count >= N_VISUAL,
        "overlap_count": overlap,
        "overlap_ratio": overlap / int(m),
        "jaccard": overlap / max(1, union),
        "mean_selected_prompt_attention": float(np.mean(np.asarray(prompt_scores)[selected])),
        "mean_selected_action_attention": float(np.mean(np.asarray(action_scores)[selected])),
        "mean_original_action_attention": float(np.mean(np.asarray(action_scores)[original])),
    }


def construct_prompt_action_bounded(
    prompt_scores: np.ndarray,
    action_scores: np.ndarray,
    m: int,
    mode: str,
    candidate_multiplier: int = 2,
    protected_fraction: float = 0.9,
) -> dict:
    """Protect Prompt core and use Action-only or joint rank fusion for a tail.

    ``joint`` uses the harmonic mean of Prompt and Action rank percentiles, so
    neither score can compensate for a very poor value of the other.  At most
    ``m-ceil(protected_fraction*m)`` tokens can differ from Prompt Top-m.
    """
    if mode not in {"action_only", "joint"}:
        raise ValueError(f"invalid bounded Prompt/Action mode: {mode}")
    if int(candidate_multiplier) != 2 or float(protected_fraction) != 0.9:
        raise ValueError("PA-Constrained v1 locks Top-2K and 90% protected core")
    original = stable_top_m(prompt_scores, int(m))
    candidate_count = min(N_VISUAL, int(candidate_multiplier) * int(m))
    candidates = stable_top_m(prompt_scores, candidate_count)
    core_count = min(int(m), int(np.ceil(float(protected_fraction) * int(m))))
    core = stable_top_m(prompt_scores, core_count)
    core_set = set(core)
    tail_count = int(m) - core_count
    eligible = [index for index in candidates if index not in core_set]
    prompt_order = stable_rank(prompt_scores, candidates, descending=True)
    action_order = stable_rank(action_scores, candidates, descending=True)
    prompt_rank = {token: rank for rank, token in enumerate(prompt_order)}
    action_rank = {token: rank for rank, token in enumerate(action_order)}
    denominator = max(1, candidate_count - 1)
    prompt_percentile = np.zeros(N_VISUAL, dtype=np.float64)
    action_percentile = np.zeros(N_VISUAL, dtype=np.float64)
    # The shared deterministic ranker validates the full 256-vector as finite;
    # -1 is strictly below every in-pool percentile/harmonic score in [0, 1].
    joint_score = np.full(N_VISUAL, -1.0, dtype=np.float64)
    for token in candidates:
        p = 1.0 - prompt_rank[token] / denominator
        a = 1.0 - action_rank[token] / denominator
        prompt_percentile[token] = p
        action_percentile[token] = a
        joint_score[token] = 2.0 * p * a / (p + a + EPS)
    if tail_count:
        tail = (
            stable_rank(action_scores, eligible, descending=True)[:tail_count]
            if mode == "action_only"
            else stable_rank(joint_score, eligible, descending=True)[:tail_count]
        )
    else:
        tail = []
    selected = sorted(core + tail)
    if len(selected) != int(m) or len(set(selected)) != int(m):
        raise RuntimeError("bounded Prompt/Action selector failed exact coverage")
    if not core_set.issubset(selected) or not set(selected).issubset(candidates):
        raise RuntimeError("bounded Prompt/Action selector violated core/pool constraint")
    overlap = len(set(selected) & set(original))
    return {
        "selected": selected,
        "original_prompt": original,
        "candidates": candidates,
        "core": sorted(core),
        "tail": sorted(tail),
        "mode": mode,
        "candidate_count": candidate_count,
        "candidate_pool_saturated": int(candidate_multiplier) * int(m) >= N_VISUAL,
        "core_count": core_count,
        "tail_count": tail_count,
        "overlap_count": overlap,
        "overlap_ratio": overlap / int(m),
        "jaccard": overlap / max(1, len(set(selected) | set(original))),
        "mean_selected_prompt_attention": float(np.mean(np.asarray(prompt_scores)[selected])),
        "mean_selected_action_attention": float(np.mean(np.asarray(action_scores)[selected])),
        "mean_original_action_attention": float(np.mean(np.asarray(action_scores)[original])),
        "selected_prompt_percentiles": [float(prompt_percentile[x]) for x in selected],
        "selected_action_percentiles": [float(action_percentile[x]) for x in selected],
        "selected_joint_scores": [float(joint_score[x]) for x in selected],
    }


def construct_prompt_action_full_joint(
    prompt_scores: np.ndarray,
    action_scores: np.ndarray,
    m: int,
    candidate_multiplier: int = 2,
) -> dict:
    """Rank all Prompt Top-2K candidates by joint Prompt/Action support."""
    if int(candidate_multiplier) != 2:
        raise ValueError("PA-Full-Joint v1 locks the candidate multiplier to 2")
    original = stable_top_m(prompt_scores, int(m))
    candidate_count = min(N_VISUAL, int(candidate_multiplier) * int(m))
    candidates = stable_top_m(prompt_scores, candidate_count)
    prompt_order = stable_rank(prompt_scores, candidates, descending=True)
    action_order = stable_rank(action_scores, candidates, descending=True)
    prompt_rank = {token: rank for rank, token in enumerate(prompt_order)}
    action_rank = {token: rank for rank, token in enumerate(action_order)}
    denominator = max(1, candidate_count - 1)
    prompt_percentile = np.zeros(N_VISUAL, dtype=np.float64)
    action_percentile = np.zeros(N_VISUAL, dtype=np.float64)
    joint_score = np.full(N_VISUAL, -1.0, dtype=np.float64)
    for token in candidates:
        p = 1.0 - prompt_rank[token] / denominator
        a = 1.0 - action_rank[token] / denominator
        prompt_percentile[token] = p
        action_percentile[token] = a
        joint_score[token] = 2.0 * p * a / (p + a + EPS)
    selected = sorted(stable_rank(joint_score, candidates, descending=True)[: int(m)])
    if len(selected) != int(m) or not set(selected).issubset(candidates):
        raise RuntimeError("PA-Full-Joint failed exact coverage/pool constraint")
    overlap = len(set(selected) & set(original))
    return {
        "selected": selected,
        "original_prompt": original,
        "candidates": candidates,
        "candidate_count": candidate_count,
        "candidate_pool_saturated": int(candidate_multiplier) * int(m) >= N_VISUAL,
        "overlap_count": overlap,
        "overlap_ratio": overlap / int(m),
        "jaccard": overlap / max(1, len(set(selected) | set(original))),
        "mean_selected_prompt_attention": float(np.mean(np.asarray(prompt_scores)[selected])),
        "mean_selected_action_attention": float(np.mean(np.asarray(action_scores)[selected])),
        "mean_original_action_attention": float(np.mean(np.asarray(action_scores)[original])),
        "selected_prompt_percentiles": [float(prompt_percentile[x]) for x in selected],
        "selected_action_percentiles": [float(action_percentile[x]) for x in selected],
        "selected_joint_scores": [float(joint_score[x]) for x in selected],
    }


@torch.inference_mode()
def extract_action_attention(
    policy,
    inputs,
    clean_token_ids: torch.Tensor,
    clean_scores: torch.Tensor,
    clean_visual: torch.Tensor,
    layers: tuple[int, ...] = tuple(range(LAYER_START, LAYER_END)),
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Extract clean teacher-forced A0..A5-to-vision post-softmax attention.

    Query q predicts action token Aq: the final prompt/empty-action position for
    A0, then the positions containing A0..A4 for A1..A5.  The seventh gripper
    query is deliberately excluded from the selector score.
    """
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    prefix = clean_token_ids[:-1].unsqueeze(0)
    teacher_ids = torch.cat([base_ids, prefix], dim=1)
    teacher_mask = torch.cat([
        base_mask,
        torch.ones_like(prefix, dtype=base_mask.dtype, device=base_mask.device),
    ], dim=1)
    with projector_intervention(policy.vla) as trace:
        output = policy.vla(
            input_ids=teacher_ids,
            attention_mask=teacher_mask,
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    if trace.before is None or not torch.equal(clean_visual, trace.before):
        raise RuntimeError("action-attention projector features differ from clean generation")
    attentions = output.attentions
    if attentions is None or len(attentions) != LAYER_END:
        raise RuntimeError(f"expected {LAYER_END} language attention layers")
    chosen = tuple(int(value) for value in layers)
    if chosen != tuple(range(LAYER_START, LAYER_END)):
        raise RuntimeError(f"action attention must use locked layers 16..31, got {chosen}")
    all_queries = [N_VISUAL + int(base_ids.shape[1]) - 1 + offset for offset in range(7)]
    action_queries = all_queries[:6]
    if max(all_queries) >= attentions[0].shape[-2]:
        raise RuntimeError("teacher-forced action query exceeds multimodal sequence")

    # [32 layers, 6 action dimensions, 256 visual tokens], mean over heads only.
    per_layer_action = torch.stack([
        attention[0, :, action_queries, 1:1 + N_VISUAL].detach().float().cpu().mean(dim=0)
        for attention in attentions
    ]).numpy().astype(np.float32)
    per_layer = per_layer_action.mean(axis=1).astype(np.float32)
    per_action = per_layer_action[list(chosen)].mean(axis=0).astype(np.float32)
    score = per_action.mean(axis=0).astype(np.float32)
    full_layer_score = per_layer.mean(axis=0).astype(np.float32)
    l11_score = per_layer[11].astype(np.float32)

    teacher_scores = output.logits[0, all_queries].detach().float()
    start = int(policy.vla.vocab_size) - ACTION_VOCAB_SIZE
    clean_action_logits = clean_scores[:, start:start + ACTION_VOCAB_SIZE].detach().float()
    teacher_action_logits = teacher_scores[:, start:start + ACTION_VOCAB_SIZE]
    clean_ids = clean_action_logits.argmax(dim=-1)
    teacher_ids_local = teacher_action_logits.argmax(dim=-1)
    greedy_equal = bool(torch.equal(clean_ids, teacher_ids_local))
    max_abs = float(torch.max(torch.abs(clean_action_logits - teacher_action_logits)).item())
    if per_layer_action.shape != (32, 6, N_VISUAL) or not np.isfinite(per_layer_action).all():
        raise FloatingPointError("invalid action attention payload")

    meta = {
        "action_attention_layers": list(chosen),
        "action_attention_layer_indexing": "zero_based_inclusive_16_31",
        "action_attention_head_count": int(attentions[0].shape[1]),
        "action_attention_query_indices": action_queries,
        "all_action_query_indices_including_gripper": all_queries,
        "action_attention_dimensions": [0, 1, 2, 3, 4, 5],
        "action_attention_visual_key_indices": [1, N_VISUAL],
        "action_attention_post_softmax": True,
        "action_attention_visual_renormalized": False,
        "action_attention_mass_on_visual": float(score.sum()),
        # This compares cached autoregressive generation with a no-cache
        # teacher-forced forward.  A near-tie can flip despite the extra
        # forward having no side effect; the selector uses only its attention,
        # while executed actions always remain from clean autoregressive logits.
        "teacher_forced_vs_autoregressive_greedy_equal": greedy_equal,
        "teacher_forced_vs_autoregressive_changed_dims": int(
            (clean_ids != teacher_ids_local).sum().item()
        ),
        "teacher_forced_action_local_ids": teacher_ids_local.detach().cpu().tolist(),
        "autoregressive_action_local_ids": clean_ids.detach().cpu().tolist(),
        "teacher_forced_clean_action_logits_max_abs_diff": max_abs,
        "action_attention_sha256": hashlib.sha256(score.tobytes()).hexdigest(),
    }
    arrays = {
        "action_attention": score,
        "action_attention_per_layer": per_layer,
        "action_attention_per_dimension": per_action,
        "action_attention_l11": l11_score,
        "action_attention_full_layers": full_layer_score,
    }
    return score, meta, arrays


def prompt_query_layout(
    input_ids: torch.Tensor,
    special_ids: set[int],
    visual_count: int = N_VISUAL,
) -> tuple[list[int], list[int], list[int]]:
    """Return text indices, multimodal query positions, and token ids.

    Prismatic keeps BOS at multimodal position 0, inserts visual tokens at
    positions 1..256, then shifts every remaining text token by 256.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids [1,T], got {tuple(input_ids.shape)}")
    ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    text_indices = [index for index, token in enumerate(ids) if token not in special_ids]
    if not text_indices or any(index == 0 for index in text_indices):
        raise RuntimeError("instruction query is empty or unexpectedly includes BOS")
    multimodal = [visual_count + index for index in text_indices]
    return text_indices, multimodal, [ids[index] for index in text_indices]


@torch.inference_mode()
def extract_prompt_attention_per_layer(
    policy, inputs, instruction: str, clean_visual: torch.Tensor
) -> tuple[np.ndarray, dict]:
    """Extract one full-prompt visual score vector for every language layer."""
    input_ids = inputs["input_ids"]
    tokenizer = policy.processor.tokenizer
    expected = tokenizer(instruction, return_tensors="pt")["input_ids"]
    if not torch.equal(input_ids.detach().cpu(), expected):
        raise RuntimeError("processor input_ids differ from direct instruction tokenization")
    text_indices, query_positions, query_token_ids = prompt_query_layout(
        input_ids, set(int(value) for value in tokenizer.all_special_ids)
    )
    with projector_intervention(policy.vla) as trace:
        output = policy.vla(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    if trace.before is None or not torch.equal(clean_visual, trace.before):
        raise RuntimeError("attention prefill projector features differ from clean generation")
    attentions = output.attentions
    if attentions is None or len(attentions) != LAYER_END:
        raise RuntimeError(f"expected {LAYER_END} language attention layers")
    if max(query_positions) >= attentions[0].shape[-2]:
        raise RuntimeError("instruction query index exceeds multimodal sequence")
    # [layer, visual], with heads and instruction queries equally weighted.
    # Visual keys are deliberately not renormalized.
    payload = torch.stack([
        attention[0, :, query_positions, 1 : 1 + N_VISUAL].detach().float().cpu().mean(dim=(0, 1))
        for attention in attentions
    ]).numpy().astype(np.float32)
    if payload.shape != (LAYER_END, N_VISUAL) or not np.isfinite(payload).all():
        raise FloatingPointError("invalid per-layer prompt attention")
    meta = {
        "attention_available_layers": list(range(LAYER_END)),
        "attention_head_count": int(attentions[0].shape[1]),
        "prompt_text_indices": text_indices,
        "prompt_multimodal_query_indices": query_positions,
        "prompt_query_token_ids": query_token_ids,
        "prompt_query_tokens": tokenizer.convert_ids_to_tokens(query_token_ids),
        "visual_key_indices": [1, N_VISUAL],
        "attention_post_softmax": True,
        "attention_visual_renormalized": False,
        "per_layer_attention_sha256": hashlib.sha256(payload.tobytes()).hexdigest(),
        "per_layer_attention_mass_on_visual": payload.sum(axis=1).astype(float).tolist(),
    }
    return payload, meta


def extract_prompt_attention(
    policy,
    inputs,
    instruction: str,
    clean_visual: torch.Tensor,
    layers: tuple[int, ...] | list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Aggregate locked layers after per-layer head/query averaging."""
    chosen = tuple(range(LAYER_START, LAYER_END)) if layers is None else tuple(int(x) for x in layers)
    if not chosen or len(set(chosen)) != len(chosen) or min(chosen) < 0 or max(chosen) >= LAYER_END:
        raise ValueError(f"invalid attention layers: {chosen}")
    per_layer, meta = extract_prompt_attention_per_layer(policy, inputs, instruction, clean_visual)
    payload = per_layer[list(chosen)].mean(axis=0).astype(np.float32)
    meta.update({
        "attention_layers": list(chosen),
        "attention_layer_count": len(chosen),
        "attention_sha256": hashlib.sha256(payload.tobytes()).hexdigest(),
        "attention_mass_on_visual": float(payload.sum()),
    })
    return payload, meta


def _instruction_inputs(policy, image: np.ndarray, instruction: str):
    """Build processor inputs for an arbitrary selector instruction.

    Mirrors OpenVLAContrastInference.process_inputs but must NOT call reset():
    reset would re-derive task entities and change the policy's task state.
    The image is processed identically, so the projector features captured for
    these inputs equal the clean visual features of the actual task branch.
    """
    from PIL import Image
    image = policy._resize_image(image)
    pil_image = Image.fromarray(image)
    return policy.processor(instruction, pil_image).to("cuda:0", dtype=torch.bfloat16)


class PromptAttentionSHRInference(STSHRCDInference):
    """Locked SHR with standard, prompt-attention, or matched-random selection."""

    selector_mode: str = "prompt_attention"
    task_index: int = 0
    attention_layers: tuple[int, ...] = tuple(range(LAYER_START, LAYER_END))
    # None preserves the original own-state Standard-SHR matched coverage.
    # An integer selects exactly that many L11-ranked tokens and deliberately
    # bypasses KMeans, so fixed-K arms do not depend on the baseline selector.
    selection_count: int | None = None
    selection_top_p: float | None = None
    selection_min_count: int = 16
    selection_max_count: int = 64
    # Optional matched-budget causal controls. A frozen schedule bypasses
    # KMeans and supplies one count per control step. A scale keeps the
    # own-state SHR/KMeans count but changes only its absolute dose.
    selection_budget_schedule: tuple[int, ...] | None = None
    selection_budget_scale: float = 1.0
    selection_budget_label: str | None = None
    selection_budget_source: str = "matched"
    selection_budget_entities: tuple[str, ...] | None = None
    selection_budget_random_salt: int = 0xB0D6E7
    save_prompt_attention: bool = True
    # Simulator-annotated region mode (research mechanism tool, not an SHR
    # performance selector). ``region_token_ids`` must be set externally before
    # each step; the mask and its harmonic reconstruction are then instruction
    # independent and identical across role-switch comparisons.
    region_token_ids: list[int] | None = None
    region_label: str | None = None
    region_aux: dict | None = None
    # Optional separate instruction used only to extract prompt attention.
    # None keeps the historical behaviour (attention from the actual task
    # instruction).  When set, the positive/negative decode always still uses
    # the actual task_description passed to step().
    selector_instruction: str | None = None
    # Optional full-instruction contrast used only to rank the reconstruction
    # mask. ``selector_difference_formula`` specifies how visual-normalized P
    # (real instruction) and Q (contrast instruction) are combined.  Action
    # decoding always continues to use the real task instruction.
    selector_contrast_instruction: str | None = None
    selector_difference_eta: float | None = None
    selector_difference_kind: str | None = None
    selector_difference_epsilon: float = 1e-8
    # Optional confidence gate for positive semantic differences.  The gate is
    # relative to the real-prompt attention, c_i=(P_i-Q_i)/(P_i+eps), so it is
    # invariant to the matched mask budget and avoids an absolute-rank/task
    # confound.  None preserves every historical difference selector exactly.
    selector_difference_confidence_threshold: float | None = None
    # ``log_boost`` preserves the historical semantic-difference experiment.
    # The target-specific experiment uses visual-normalized linear scores:
    #   target_diff  : P - Q
    #   target_boost : P + eta * (P - Q)
    #   reverse_diff : Q - P
    #   positive_boost: P + eta * max(P - Q, 0)
    #   reverse_boost : P + eta * max(Q - P, 0)
    # where P is the real instruction and Q is a grammatical generic frame.
    selector_difference_formula: str = "log_boost"
    # Optional Prompt/Action complement mask experiment.  All modes preserve
    # Top-(m-floor(m/4)) Prompt-L11 tokens and differ only in the supplement.
    complement_arm: str | None = None
    action_attention_layers: tuple[int, ...] = tuple(range(16, 32))
    # Optional sequential selector: Prompt-L11 Top-(multiplier*m) candidate
    # pool, then clean-positive Action attention Top-m within that pool.
    prompt_action_rerank_multiplier: int | None = None
    # Conservative v1 selector: protect Prompt Top-ceil(.9K), use a joint or
    # Action-only rule only for the remaining tail inside Prompt Top-2K.
    prompt_action_bounded_mode: str | None = None
    # Full joint selector: no protected Prompt core. All final K tokens are
    # selected by the harmonic mean of Prompt and Action rank support in Top-2K.
    prompt_action_full_joint: bool = False

    def _forward_scores(self, inputs, unnorm_key, **kwargs):
        """Generate one logit vector for every fixed action dimension.

        OpenVLA action decoding is seven-dimensional.  The upstream
        ``generate`` helper otherwise honors the language-model EOS token and
        may stop after fewer than seven action positions on some WidowX
        observations.  That produces an invalid partial action rather than a
        valid policy decision, so EOS must not terminate this fixed-length
        action-token loop.
        """
        kwargs = dict(kwargs)
        # ``min_new_tokens`` alone is insufficient for this Prismatic
        # generation path: its multimodal length accounting can still return a
        # truncated score list after emitting the language EOS token.  EOS is
        # not an action bin, so exclude it throughout fixed-length action
        # decoding and let max_new_tokens determine the seven positions.
        eos_id = int(self.vla.generation_config.eos_token_id)
        kwargs["logits_processor"] = LogitsProcessorList([
            SuppressTokensLogitsProcessor([eos_id])
        ])
        return super()._forward_scores(inputs, unnorm_key, **kwargs)

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        if self.selector_mode == "standard_shr":
            return super().step(image, contrast_image, task_description, *args, **kwargs)
        if self.selector_mode not in {"prompt_attention", "random_matched", "sim_region"}:
            raise ValueError(f"unknown Prompt-Attn-SHR selector mode: {self.selector_mode}")

        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError(
                "positive branch did not produce 7 scores and visual features: "
                f"scores_shape={tuple(clean_scores.shape)}, "
                f"visual_available={positive_trace.before is not None}"
            )
        visual = positive_trace.before
        h = visual[0].numpy().astype(np.float32)

        attention_scores = None
        attention_meta = {}
        action_attention_scores = None
        action_attention_meta = {}
        action_attention_arrays = {}
        random_seed = None
        if self.selector_mode == "prompt_attention":
            if self.selector_difference_eta is None:
                sel_instr = self.selector_instruction if (
                    self.selector_instruction and self.selector_instruction != task_description
                ) else task_description
                if sel_instr == task_description:
                    attention_inputs = inputs
                else:
                    attention_inputs = _instruction_inputs(self, image, sel_instr)
                attention_scores, attention_meta = extract_prompt_attention(
                    self, attention_inputs, sel_instr, visual, layers=self.attention_layers
                )
                attention_meta["selector_instruction"] = sel_instr
                attention_meta["selector_matches_task"] = bool(sel_instr == task_description)
            else:
                if self.selector_instruction not in (None, task_description):
                    raise RuntimeError("difference selector requires the real instruction as P")
                contrast_instruction = self.selector_contrast_instruction
                if not contrast_instruction or contrast_instruction == task_description:
                    raise RuntimeError("difference selector requires a distinct contrast instruction")
                eta = float(self.selector_difference_eta)
                epsilon = float(self.selector_difference_epsilon)
                if not np.isfinite(eta) or not np.isfinite(epsilon) or epsilon <= 0:
                    raise RuntimeError("invalid semantic-difference eta/epsilon")
                p_raw, p_meta = extract_prompt_attention(
                    self, inputs, task_description, visual, layers=self.attention_layers
                )
                q_inputs = _instruction_inputs(self, image, contrast_instruction)
                q_raw, q_meta = extract_prompt_attention(
                    self, q_inputs, contrast_instruction, visual, layers=self.attention_layers
                )
                p = np.asarray(p_raw, dtype=np.float64)
                q = np.asarray(q_raw, dtype=np.float64)
                p /= float(p.sum())
                q /= float(q.sum())
                formula = str(self.selector_difference_formula)
                log_p = np.log(p + epsilon)
                log_q = np.log(q + epsilon)
                if formula == "log_boost":
                    difference = log_p - log_q
                    attention_scores = log_p + eta * difference
                elif formula == "target_diff":
                    difference = p - q
                    attention_scores = difference
                elif formula == "target_boost":
                    difference = p - q
                    attention_scores = p + eta * difference
                elif formula == "reverse_diff":
                    difference = q - p
                    attention_scores = difference
                elif formula == "positive_boost":
                    difference = p - q
                    attention_scores = p + eta * np.maximum(difference, 0.0)
                elif formula == "confidence_gated_positive_boost":
                    difference = p - q
                    threshold = self.selector_difference_confidence_threshold
                    if threshold is None or not np.isfinite(float(threshold)) or float(threshold) < 0:
                        raise RuntimeError("confidence-gated boost requires a finite non-negative threshold")
                    relative_confidence = difference / (p + epsilon)
                    confidence_gate = relative_confidence >= float(threshold)
                    attention_scores = p + eta * np.maximum(difference, 0.0) * confidence_gate
                elif formula == "reverse_boost":
                    difference = q - p
                    attention_scores = p + eta * np.maximum(difference, 0.0)
                else:
                    raise RuntimeError(f"unknown selector difference formula: {formula}")
                if not all(np.isfinite(x).all() for x in (p, q, difference, attention_scores)):
                    raise FloatingPointError("non-finite semantic-difference selector score")
                attention_meta = dict(p_meta)
                attention_meta.update({
                    "selector_instruction": task_description,
                    "selector_matches_task": True,
                    "selector_contrast_instruction": contrast_instruction,
                    "selector_difference_kind": self.selector_difference_kind,
                    "selector_difference_formula": formula,
                    "selector_difference_eta": eta,
                    "selector_difference_epsilon": epsilon,
                    "selector_difference_confidence_threshold": (
                        float(self.selector_difference_confidence_threshold)
                        if self.selector_difference_confidence_threshold is not None else None
                    ),
                    "contrast_prompt_query_token_ids": q_meta["prompt_query_token_ids"],
                    "contrast_prompt_query_tokens": q_meta["prompt_query_tokens"],
                    "contrast_prompt_multimodal_query_indices": q_meta["prompt_multimodal_query_indices"],
                    "contrast_attention_sha256": q_meta["attention_sha256"],
                    "correct_attention_mass_on_visual": float(p_raw.sum()),
                    "contrast_attention_mass_on_visual": float(q_raw.sum()),
                    "selector_score_sha256": hashlib.sha256(
                        np.ascontiguousarray(attention_scores).tobytes()
                    ).hexdigest(),
                })
                if formula == "confidence_gated_positive_boost":
                    attention_meta.update({
                        "confidence_gate_eligible_count": int(confidence_gate.sum()),
                        "confidence_gate_eligible_fraction": float(confidence_gate.mean()),
                        "confidence_gate_relative_max": float(relative_confidence.max()),
                    })

        active_pa_modes = sum(value is not None for value in (
            self.complement_arm,
            self.prompt_action_rerank_multiplier,
            self.prompt_action_bounded_mode,
        )) + int(bool(self.prompt_action_full_joint))
        if active_pa_modes > 1:
            raise RuntimeError("Prompt/Action selector modes are mutually exclusive")
        if active_pa_modes:
            if self.selector_mode != "prompt_attention" or self.attention_layers != (11,):
                raise RuntimeError("Prompt/Action selector requires Prompt zero-based L11")
            if self.selector_difference_eta is not None or self.selector_instruction is not None:
                raise RuntimeError("Prompt/Action selector forbids alternate Prompt instructions")
            if self.complement_arm is not None and self.complement_arm not in {
                "original", "prompt_high_action_high", "prompt_high_action_low",
                "prompt_low_action_high",
            }:
                raise RuntimeError(f"unknown complement arm: {self.complement_arm}")
            if self.prompt_action_rerank_multiplier is not None and int(
                self.prompt_action_rerank_multiplier
            ) != 3:
                raise RuntimeError("PA-Rerank v1 locks the candidate multiplier to 3")
            if self.prompt_action_bounded_mode not in {None, "action_only", "joint"}:
                raise RuntimeError(f"unknown bounded Prompt/Action mode: {self.prompt_action_bounded_mode}")
            action_attention_scores, action_attention_meta, action_attention_arrays = (
                extract_action_attention(
                    self, inputs, clean_scores.argmax(dim=-1), clean_scores, visual,
                    layers=self.action_attention_layers,
                )
            )

        m_raw = None
        selected_attention_mass = None
        lower_bound_triggered = False
        upper_bound_triggered = False
        matched_entity_groups = []
        budget_source = None
        budget_entities = None
        if self.selection_top_p is not None:
            if self.selection_count is not None or self.selector_mode != "prompt_attention":
                raise RuntimeError("Top-p selection requires prompt attention and no fixed count")
            threshold = float(self.selection_top_p)
            if not 0.0 < threshold < 1.0:
                raise RuntimeError(f"invalid visual Top-p threshold: {threshold}")
            probability = np.asarray(attention_scores, dtype=np.float64)
            mass = float(probability.sum())
            if not np.isfinite(mass) or mass <= 0:
                raise FloatingPointError("invalid visual attention mass")
            probability /= mass
            order = np.lexsort((np.arange(N_VISUAL), -probability))
            m_raw = int(np.searchsorted(np.cumsum(probability[order]), threshold) + 1)
            lower_bound_triggered = m_raw < int(self.selection_min_count)
            upper_bound_triggered = m_raw > int(self.selection_max_count)
            m = int(np.clip(m_raw, self.selection_min_count, self.selection_max_count))
            reference = []
            labels = None
            per_entity_score = []
            entity_groups = []
            coverage_mode = f"visual_top_p_{threshold:.2f}_clip_{self.selection_min_count}_{self.selection_max_count}"
        elif self.selector_mode == "sim_region":
            if self.selection_top_p is not None or self.selection_count is not None:
                raise RuntimeError("sim_region selection forbids Top-p and fixed count")
            if not self.region_token_ids:
                raise RuntimeError("sim_region requires externally provided region_token_ids")
            m = len(set(self.region_token_ids))
            if not 1 <= m <= N_VISUAL:
                raise RuntimeError(f"invalid sim_region coverage: {m}")
            labels = None
            per_entity_score = []
            entity_groups = []
            reference = []
            coverage_mode = f"sim_region_{self.region_label or 'unlabeled'}"
        elif self.selection_budget_schedule is not None:
            if self.selection_count is not None or self.selector_mode != "prompt_attention":
                raise RuntimeError("scheduled budget requires prompt attention and no fixed count")
            control_step = len(self._episode_trace)
            if control_step >= len(self.selection_budget_schedule):
                raise RuntimeError(
                    f"budget schedule exhausted at step {control_step}: "
                    f"length={len(self.selection_budget_schedule)}"
                )
            m = int(self.selection_budget_schedule[control_step])
            if not 1 <= m <= N_VISUAL:
                raise RuntimeError(f"invalid scheduled token count: {m}")
            labels = None
            per_entity_score = []
            entity_groups = []
            reference = []
            coverage_mode = f"frozen_budget_schedule_{self.selection_budget_label or 'unlabeled'}"
        elif self.selection_count is None:
            labels, _, per_entity_score = self._semantic_clusters(h)
            matched_entity_groups = []
            for group in self._entity_groups(h, labels):
                if group not in matched_entity_groups:
                    matched_entity_groups.append(int(group))
            budget_source = str(self.selection_budget_source)
            if budget_source == "matched":
                entity_groups = matched_entity_groups
            elif budget_source == "wrong_entity":
                budget_entities = tuple(self.selection_budget_entities or ())
                if not budget_entities:
                    raise RuntimeError("wrong-entity budget requires selection_budget_entities")
                group_vectors = np.stack([
                    h[labels == group].mean(axis=0) for group in range(self.kmeans_K)
                ])
                group_vectors /= np.linalg.norm(group_vectors, axis=1, keepdims=True) + 1e-8
                entity_groups = []
                for entity in budget_entities:
                    embedding = self._embed_phrase(entity)
                    embedding /= np.linalg.norm(embedding) + 1e-8
                    group = int(np.argmax(group_vectors @ embedding))
                    if group not in entity_groups:
                        entity_groups.append(group)
            elif budget_source == "random_cluster":
                random_seed = int(np.random.SeedSequence([
                    self.task_index,
                    self._episode_seed,
                    len(self._episode_trace),
                    int(self.selection_budget_random_salt),
                ]).generate_state(1, dtype=np.uint32)[0])
                entity_groups = sorted(int(value) for value in np.random.default_rng(
                    random_seed
                ).choice(
                    self.kmeans_K,
                    size=len(matched_entity_groups),
                    replace=False,
                ))
            else:
                raise RuntimeError(f"unknown selection budget source: {budget_source}")
            reference = sorted(set(
                int(index) for group in entity_groups for index in np.flatnonzero(labels == group)
            ))
            m = len(reference)
            if not 1 <= m <= N_VISUAL:
                raise RuntimeError(f"invalid standard-SHR reference coverage: {m}")
            scale = float(self.selection_budget_scale)
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError(f"invalid matched budget scale: {scale}")
            unscaled_m = m
            m = int(np.clip(round(scale * unscaled_m), 1, N_VISUAL))
            coverage_mode = (
                f"own_state_{budget_source}_budget" if scale == 1.0
                else f"own_state_{budget_source}_scaled_{scale:.2f}"
            )
        else:
            m = int(self.selection_count)
            if m not in {16, 24, 32, 48, 64}:
                raise RuntimeError(f"invalid fixed token count: {m}")
            # Fixed-count selection must not call or otherwise depend on KMeans.
            labels = None
            per_entity_score = []
            entity_groups = []
            reference = []
            coverage_mode = f"fixed_top_{m}"

        complement_meta = {}
        rerank_meta = {}
        bounded_meta = {}
        full_joint_meta = {}
        if self.complement_arm is not None:
            masks = construct_prompt_action_masks(attention_scores, action_attention_scores, m)
            spec = masks[self.complement_arm]
            selected = spec["selected"]
            complement_meta = {
                "complement_arm": self.complement_arm,
                "supplement_rule": self.complement_arm,
                "supplement_count_r": spec["supplement_count_r"],
                "core_token_ids": spec["core"],
                "candidate_prompt_high_token_ids": spec["candidate_high"],
                "candidate_prompt_low_token_ids": spec["candidate_low"],
                "supplement_token_ids": spec["supplement"],
                "original_prompt_token_ids": masks["original"]["selected"],
                **{key: spec[key] for key in (
                    "actual_replacements_vs_original", "supplement_prompt_scores",
                    "supplement_action_scores", "supplement_prompt_global_ranks",
                    "supplement_action_global_ranks", "core_exact",
                    "candidate_pool_membership_exact",
                )},
            }
        elif self.prompt_action_rerank_multiplier is not None:
            spec = construct_prompt_action_rerank(
                attention_scores,
                action_attention_scores,
                m,
                candidate_multiplier=int(self.prompt_action_rerank_multiplier),
            )
            selected = spec["selected"]
            rerank_meta = {
                "prompt_action_rerank": True,
                "prompt_action_rerank_rule": "Prompt-L11 Top-3K then Action-L16-31 Top-K",
                "candidate_multiplier": spec["candidate_multiplier"],
                "candidate_count_raw": spec["raw_candidate_count"],
                "candidate_pool_size": spec["candidate_count"],
                "candidate_pool_saturated": spec["candidate_pool_saturated"],
                "candidate_token_ids": spec["candidates"],
                "original_prompt_token_ids": spec["original_prompt"],
                "rerank_overlap_count": spec["overlap_count"],
                "rerank_overlap_ratio": spec["overlap_ratio"],
                "rerank_jaccard": spec["jaccard"],
                "mean_selected_prompt_attention": spec["mean_selected_prompt_attention"],
                "mean_selected_action_attention": spec["mean_selected_action_attention"],
                "mean_original_l11_action_attention": spec["mean_original_action_attention"],
                "rerank_exact_m": len(selected) == m,
                "rerank_subset_of_candidate_pool": set(selected).issubset(spec["candidates"]),
            }
        elif self.prompt_action_bounded_mode is not None:
            spec = construct_prompt_action_bounded(
                attention_scores,
                action_attention_scores,
                m,
                mode=self.prompt_action_bounded_mode,
            )
            selected = spec["selected"]
            bounded_meta = {
                "prompt_action_bounded": True,
                "prompt_action_bounded_mode": spec["mode"],
                "prompt_action_bounded_rule": "protect Prompt Top-ceil(0.9K); fill inside Prompt Top-2K",
                "candidate_multiplier": 2,
                "protected_fraction": 0.9,
                "candidate_pool_size": spec["candidate_count"],
                "candidate_pool_saturated": spec["candidate_pool_saturated"],
                "candidate_token_ids": spec["candidates"],
                "protected_core_token_ids": spec["core"],
                "supplement_token_ids": spec["tail"],
                "original_prompt_token_ids": spec["original_prompt"],
                "protected_core_count": spec["core_count"],
                "supplement_count": spec["tail_count"],
                "bounded_overlap_count": spec["overlap_count"],
                "bounded_overlap_ratio": spec["overlap_ratio"],
                "bounded_jaccard": spec["jaccard"],
                "mean_selected_prompt_attention": spec["mean_selected_prompt_attention"],
                "mean_selected_action_attention": spec["mean_selected_action_attention"],
                "mean_original_l11_action_attention": spec["mean_original_action_attention"],
                "selected_prompt_rank_percentiles": spec["selected_prompt_percentiles"],
                "selected_action_rank_percentiles": spec["selected_action_percentiles"],
                "selected_joint_scores": spec["selected_joint_scores"],
                "bounded_exact_m": len(selected) == m,
                "bounded_core_preserved": set(spec["core"]).issubset(selected),
                "bounded_subset_of_candidate_pool": set(selected).issubset(spec["candidates"]),
                "bounded_max_replacements_respected": (m - spec["overlap_count"]) <= spec["tail_count"],
            }
        elif self.prompt_action_full_joint:
            spec = construct_prompt_action_full_joint(
                attention_scores, action_attention_scores, m, candidate_multiplier=2
            )
            selected = spec["selected"]
            full_joint_meta = {
                "prompt_action_full_joint": True,
                "prompt_action_full_joint_rule": "harmonic mean of Prompt/Action rank support inside Prompt Top-2K",
                "candidate_multiplier": 2,
                "candidate_pool_size": spec["candidate_count"],
                "candidate_pool_saturated": spec["candidate_pool_saturated"],
                "candidate_token_ids": spec["candidates"],
                "original_prompt_token_ids": spec["original_prompt"],
                "full_joint_overlap_count": spec["overlap_count"],
                "full_joint_overlap_ratio": spec["overlap_ratio"],
                "full_joint_jaccard": spec["jaccard"],
                "mean_selected_prompt_attention": spec["mean_selected_prompt_attention"],
                "mean_selected_action_attention": spec["mean_selected_action_attention"],
                "mean_original_l11_action_attention": spec["mean_original_action_attention"],
                "selected_prompt_rank_percentiles": spec["selected_prompt_percentiles"],
                "selected_action_rank_percentiles": spec["selected_action_percentiles"],
                "selected_joint_scores": spec["selected_joint_scores"],
                "full_joint_exact_m": len(selected) == m,
                "full_joint_subset_of_candidate_pool": set(selected).issubset(spec["candidates"]),
            }
        elif self.selector_mode == "prompt_attention":
            selected = stable_top_m(attention_scores, m)
            if self.selection_top_p is not None:
                selected_attention_mass = float(probability[selected].sum())
        elif self.selector_mode == "sim_region":
            selected = sorted(set(int(index) for index in self.region_token_ids))
            if len(selected) != m:
                raise RuntimeError("sim_region duplicate token ids in region")
        else:
            random_seed = int(np.random.SeedSequence([
                self.task_index, self._episode_seed, self._selector_step, RANDOM_SALT,
            ]).generate_state(1, dtype=np.uint32)[0])
            selected = sorted(int(index) for index in np.random.default_rng(random_seed).choice(
                N_VISUAL, size=m, replace=False
            ))
        if len(selected) != m or len(set(selected)) != m:
            raise RuntimeError("selector failed exact requested coverage")

        h_negative = h.copy()
        reconstructed = harmonic_reconstruct(h, np.asarray(selected, dtype=np.int64), beta=0.0)
        h_negative[selected] = reconstructed
        outside = np.asarray([index for index in range(N_VISUAL) if index not in set(selected)])
        non_target_equal = bool(np.array_equal(h_negative[outside], h[outside]))
        if not non_target_equal or not np.isfinite(h_negative).all():
            raise RuntimeError("harmonic reconstruction audit failed")

        clean_token_ids = clean_scores.argmax(dim=-1)
        with projector_merge_intervention(
            self.vla, torch.from_numpy(h_negative).unsqueeze(0)
        ) as negative_trace:
            negative_scores = guided_forward_scores(
                self.vla, inputs, clean_token_ids, visual.shape[1]
            )
        if negative_trace["before"] is None or not torch.equal(visual, negative_trace["before"]):
            raise RuntimeError("negative projector input differs from clean features")

        final_scores, guidance_meta = self._combine_action_scores(clean_scores, negative_scores)
        # EOS is explicitly excluded during fixed-length action decoding.  CD
        # combines its two suppressed (-inf) logits as -inf - (-inf), which is
        # NaN although it is outside the action vocabulary.  Replace only that
        # known excluded position; any other non-finite logit remains fatal.
        eos_id = int(self.vla.generation_config.eos_token_id)
        invalid = ~torch.isfinite(final_scores)
        if invalid.any():
            expected = torch.zeros_like(invalid, dtype=torch.bool)
            expected[:, eos_id] = True
            if (invalid & ~expected).any():
                raise FloatingPointError("non-finite non-EOS Prompt-Attn-SHR scores")
            final_scores = final_scores.clone()
            final_scores[:, eos_id] = torch.finfo(final_scores.dtype).min
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("non-finite Prompt-Attn-SHR scores")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        start = int(self.vla.vocab_size) - ACTION_VOCAB_SIZE
        centered = (clean_scores[:6, start:start + ACTION_VOCAB_SIZE].float()
                    - negative_scores[:6, start:start + ACTION_VOCAB_SIZE].float())
        centered = centered - centered.mean(dim=-1, keepdim=True)
        clean_decoded = self._decode_actions(clean_token_ids, self.unnorm_key)
        guided_decoded = self._decode_actions(token_ids, self.unnorm_key)
        overlap = len(set(selected) & set(reference))
        if self.selector_difference_eta is not None:
            correct_prompt_selected = stable_top_m(p, m)
        elif self.complement_arm is not None:
            correct_prompt_selected = complement_meta["original_prompt_token_ids"]
        elif self.prompt_action_rerank_multiplier is not None:
            correct_prompt_selected = rerank_meta["original_prompt_token_ids"]
        elif self.prompt_action_bounded_mode is not None:
            correct_prompt_selected = bounded_meta["original_prompt_token_ids"]
        elif self.prompt_action_full_joint:
            correct_prompt_selected = full_joint_meta["original_prompt_token_ids"]
        else:
            correct_prompt_selected = selected
        correct_prompt_overlap = len(set(selected) & set(correct_prompt_selected))
        component_count, isolated_token_ratio = mask_spatial_stats(selected)
        perturbation = h_negative - h
        meta = {
            "selection_mode": self.selector_mode,
            "instruction": task_description,
            "selected_entities": list(self._entities),
            "selected_group_ids": entity_groups,
            "matched_budget_group_ids": matched_entity_groups,
            "per_entity_score": per_entity_score,
            "selected_token_ids": selected,
            "reference_shr_token_ids": reference,
            "m_t": m,
            "num_tokens": m,
            "requested_token_count": m,
            "actual_selected_count": len(selected),
            "coverage_mode": coverage_mode,
            "coverage_exact": len(selected) == m,
            "top_p_threshold": float(self.selection_top_p) if self.selection_top_p is not None else None,
            "m_raw": m_raw,
            "selected_attention_mass": selected_attention_mass,
            "lower_bound_triggered": lower_bound_triggered,
            "upper_bound_triggered": upper_bound_triggered,
            "selection_min_count": int(self.selection_min_count) if self.selection_top_p is not None else None,
            "selection_max_count": int(self.selection_max_count) if self.selection_top_p is not None else None,
            "budget_scale": float(self.selection_budget_scale),
            "budget_source": (
                str(self.selection_budget_source)
                if self.selection_budget_schedule is None and self.selection_count is None
                and self.selection_top_p is None and self.selector_mode == "prompt_attention"
                else "frozen_schedule" if self.selection_budget_schedule is not None else None
            ),
            "budget_entities": list(self.selection_budget_entities or ()),
            "budget_schedule_label": self.selection_budget_label,
            "budget_schedule_step": len(self._episode_trace) if self.selection_budget_schedule is not None else None,
            "unscaled_matched_m_t": unscaled_m if self.selection_budget_schedule is None and self.selection_count is None and self.selection_top_p is None and self.selector_mode == "prompt_attention" else None,
            "mask_component_count": component_count,
            "isolated_token_ratio": isolated_token_ratio,
            "prompt_shr_intersection": overlap,
            "prompt_shr_overlap_ratio": overlap / m if reference else None,
            "prompt_shr_jaccard": (
                overlap / max(1, len(set(selected) | set(reference))) if reference else None
            ),
            "correct_prompt_token_ids": correct_prompt_selected,
            "correct_prompt_retention": correct_prompt_overlap / m,
            "correct_prompt_jaccard": correct_prompt_overlap / max(
                1, len(set(selected) | set(correct_prompt_selected))
            ),
            "region_label": self.region_label,
            "region_aux": dict(self.region_aux) if self.region_aux else None,
            "random_seed": random_seed,
            "random_rule": "SeedSequence([task_index, episode_seed, control_step, 0x50A77E11])",
            "kmeans_K": int(self.kmeans_K),
            "kmeans_seed": int(self.kmeans_seed),
            "beta": 0.0,
            "lambda": float(self.lambd),
            "feature_equal": True,
            "non_target_bit_identical": non_target_equal,
            "reconstruction_finite": True,
            "feature_perturbation_norm": float(np.linalg.norm(perturbation)),
            "feature_perturbation_relative": float(
                np.linalg.norm(perturbation) / (np.linalg.norm(h) + EPS)
            ),
            "centered_logit_residual_norm": float(torch.linalg.vector_norm(centered).item()),
            "centered_logit_residual_norm_per_dim": [
                float(value) for value in torch.linalg.vector_norm(centered, dim=-1).cpu()
            ],
            "guided_prefix": True,
            "positive_token_ids": clean_token_ids.detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "guided_changed_dims": int((token_ids[:6] != clean_token_ids[:6]).sum().item()),
            "guided_change_ratio": float((token_ids[:6] != clean_token_ids[:6]).float().mean().item()),
            "clean_action": np.asarray(clean_decoded).tolist(),
            "guided_action": np.asarray(guided_decoded).tolist(),
            "guided_clean_action_l2": float(np.linalg.norm(guided_decoded[:6] - clean_decoded[:6])),
            **attention_meta,
            **action_attention_meta,
            **complement_meta,
            **rerank_meta,
            **bounded_meta,
            **full_joint_meta,
            **guidance_meta,
        }
        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        record = {
            "positive": positive,
            "negative": negative,
            "selected_mask": np.isin(np.arange(N_VISUAL), selected).astype(np.uint8),
            "reference_shr_mask": np.isin(np.arange(N_VISUAL), reference).astype(np.uint8),
            "per_token_perturbation_norm": (
                np.linalg.norm(perturbation, axis=-1).astype(np.float32) if perturbation.ndim == 2
                else np.zeros(N_VISUAL, dtype=np.float32)
            ),
        }
        if attention_scores is not None and self.save_prompt_attention:
            record["prompt_attention"] = attention_scores.astype(np.float32)
            if self.selector_difference_eta is not None:
                record["correct_attention_probability"] = p
                record["contrast_attention_probability"] = q
                record["attention_log_difference"] = difference
                record["attention_difference"] = difference
                record["selector_score"] = attention_scores
                if self.selector_difference_formula == "confidence_gated_positive_boost":
                    record["relative_difference_confidence"] = relative_confidence.astype(np.float32)
                    record["confidence_gate_mask"] = confidence_gate.astype(np.uint8)
                record["correct_prompt_mask"] = np.isin(
                    np.arange(N_VISUAL), correct_prompt_selected
                ).astype(np.uint8)
        if action_attention_arrays:
            record.update(action_attention_arrays)
            if self.complement_arm is not None:
                record["core_mask"] = np.isin(
                    np.arange(N_VISUAL), complement_meta["core_token_ids"]
                ).astype(np.uint8)
                record["supplement_mask"] = np.isin(
                    np.arange(N_VISUAL), complement_meta["supplement_token_ids"]
                ).astype(np.uint8)
            elif self.prompt_action_rerank_multiplier is not None:
                record["candidate_mask"] = np.isin(
                    np.arange(N_VISUAL), rerank_meta["candidate_token_ids"]
                ).astype(np.uint8)
                record["original_prompt_mask"] = np.isin(
                    np.arange(N_VISUAL), rerank_meta["original_prompt_token_ids"]
                ).astype(np.uint8)
            elif self.prompt_action_bounded_mode is not None:
                record["candidate_mask"] = np.isin(
                    np.arange(N_VISUAL), bounded_meta["candidate_token_ids"]
                ).astype(np.uint8)
                record["original_prompt_mask"] = np.isin(
                    np.arange(N_VISUAL), bounded_meta["original_prompt_token_ids"]
                ).astype(np.uint8)
                record["protected_core_mask"] = np.isin(
                    np.arange(N_VISUAL), bounded_meta["protected_core_token_ids"]
                ).astype(np.uint8)
                record["supplement_mask"] = np.isin(
                    np.arange(N_VISUAL), bounded_meta["supplement_token_ids"]
                ).astype(np.uint8)
            else:
                record["candidate_mask"] = np.isin(
                    np.arange(N_VISUAL), full_joint_meta["candidate_token_ids"]
                ).astype(np.uint8)
                record["original_prompt_mask"] = np.isin(
                    np.arange(N_VISUAL), full_joint_meta["original_prompt_token_ids"]
                ).astype(np.uint8)
        self._episode_logits.append(record)
        self._episode_trace.append(meta)
        self._selector_step += 1
        return raw_action, actions, meta
