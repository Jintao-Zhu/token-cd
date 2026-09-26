"""Small, deterministic utilities for the L11 rank-causal study."""
from __future__ import annotations

import hashlib

import numpy as np

N_VISUAL = 256
RANK_BIN_SIZE = 32
N_RANK_BINS = 8


def rank_bins(scores: np.ndarray) -> list[list[int]]:
    """Return B1..B8, stable descending score with token-id tie break."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (N_VISUAL,) or not np.isfinite(values).all():
        raise ValueError(f"expected 256 finite L11 scores, got {values.shape}")
    order = np.lexsort((np.arange(N_VISUAL), -values))
    bins = [order[i * RANK_BIN_SIZE:(i + 1) * RANK_BIN_SIZE].astype(int).tolist()
            for i in range(N_RANK_BINS)]
    if any(len(group) != RANK_BIN_SIZE for group in bins):
        raise RuntimeError("rank partition did not produce eight groups of 32")
    if len({token for group in bins for token in group}) != N_VISUAL:
        raise RuntimeError("rank bins are not a disjoint partition of all visual tokens")
    return bins


def deterministic_random32(benchmark: str, task: str, episode: str, step: int) -> list[int]:
    """State-keyed Random32; reproducible regardless of worker scheduling."""
    key = f"{benchmark}\0{task}\0{episode}\0{int(step)}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return sorted(int(x) for x in np.random.default_rng(seed).choice(
        N_VISUAL, size=RANK_BIN_SIZE, replace=False
    ))


def clean_margin_effect(clean_logits: np.ndarray, negative_logits: np.ndarray) -> dict:
    """Compute |guidance effect| on each clean top-1-vs-top-2 action margin.

    Inputs may be full-vocabulary logits or the already sliced 256-bin action
    logits. Only rows 0..5 are included; the gripper row is intentionally
    excluded. Positive effect means the contrastive residual increases the
    clean winner's margin; ``effect_score`` uses its absolute magnitude.
    """
    clean = np.asarray(clean_logits, dtype=np.float64)
    negative = np.asarray(negative_logits, dtype=np.float64)
    if clean.ndim != 2 or negative.shape != clean.shape or clean.shape[0] < 6:
        raise ValueError(f"expected matching [>=6,vocab] logits, got {clean.shape}, {negative.shape}")
    clean = clean[:6]
    negative = negative[:6]
    if not np.isfinite(clean).all() or not np.isfinite(negative).all():
        raise FloatingPointError("non-finite action logits")
    residual = clean - negative
    signed = []
    magnitude = []
    for dim in range(6):
        order = np.lexsort((np.arange(clean.shape[1]), -clean[dim]))
        winner, runner_up = int(order[0]), int(order[1])
        g = float(residual[dim, winner] - residual[dim, runner_up])
        signed.append(g)
        magnitude.append(abs(g))
    return {
        "guidance_margin_effect_by_dim": signed,
        "absolute_margin_effect_by_dim": magnitude,
        "effect_score": float(np.mean(magnitude)),
        "signed_effect_mean": float(np.mean(signed)),
    }


def evaluate_libero_rank_bins(model, processor, image, instruction: str, unnorm_key: str) -> dict:
    """Evaluate all eight harmonic rank-bin branches on one fixed LIBERO view."""
    import torch
    from research.ar_token_counterfactual.libero_runtime import build_prompt
    from research.semantic_token_cd import libero_matched_rollout as lmr

    inputs = processor(build_prompt(instruction), image).to(model.device, dtype=torch.bfloat16)
    clean_ids, clean_logits, clean_visual = lmr.generate_clean_action(
        model, processor, image, instruction, inputs
    )
    attention, h, attention_meta = lmr.prompt_attention_and_features(
        model, processor, image, instruction, inputs,
        expected_h=clean_visual, query_mode="instruction_only",
        attention_layers=(11,), attention_heads=(), destination_weight=0.0,
    )
    bins = rank_bins(attention)
    vocab_size = int(model.vocab_size)
    start = vocab_size - 256
    clean_action_logits = clean_logits[:, start:start + 256].detach().float().cpu().numpy()
    if clean_action_logits.shape != (7, 256):
        raise RuntimeError(f"unexpected LIBERO clean action logits: {clean_action_logits.shape}")
    rows = []
    for bin_index, selected in enumerate(bins):
        replacement = h.copy()
        replacement[np.asarray(selected, dtype=np.int64)] = lmr.harmonic_reconstruct(
            h, np.asarray(selected, dtype=np.int64)
        )
        negative = lmr.forward_logits(
            model, processor, image, instruction, clean_ids,
            selected=selected, mean=torch.from_numpy(replacement).float(),
        )
        negative_action_logits = negative[:, start:start + 256].detach().float().cpu().numpy()
        effect = clean_margin_effect(clean_action_logits, negative_action_logits)
        rows.append({
            "rank_bin": bin_index + 1,
            "token_ids": selected,
            **effect,
        })
    return {
        "attention_scores": np.asarray(attention, dtype=np.float32),
        "attention_sha256": attention_meta.get("attention_sha256"),
        "clean_action_logits": clean_action_logits,
        "clean_token_ids": clean_ids[0].detach().cpu().tolist(),
        "rank_bins": rows,
    }


def evaluate_simpler_rank_bins(policy, image, instruction: str) -> dict:
    """Evaluate all eight harmonic rank-bin branches on one fixed SIMPLER view."""
    import torch
    from research.ar_token_counterfactual.intervention import (
        projector_intervention,
    )
    from research.semantic_token_cd.global_merge_policy import (
        guided_forward_scores,
        projector_merge_intervention,
    )
    from research.semantic_token_cd.prompt_attn_shr_policy import (
        extract_prompt_attention,
    )
    from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct

    inputs = policy.process_inputs(image, task_description=instruction)
    with projector_intervention(policy.vla) as trace:
        clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
    visual = trace.before
    if visual is None or visual.shape[1] != N_VISUAL or clean_scores.shape[0] != 7:
        raise RuntimeError("invalid SIMPLER clean branch / projector trace")
    h = visual[0].detach().float().cpu().numpy()
    attention, attention_meta = extract_prompt_attention(
        policy, inputs, instruction, visual, layers=(11,)
    )
    bins = rank_bins(attention)
    action_bins = int(policy.vla.vocab_size) - int(policy.vla.vocab_size - 256)
    start = int(policy.vla.vocab_size) - action_bins
    clean_action_logits = clean_scores[:, start:start + action_bins].detach().float().cpu().numpy()
    if clean_action_logits.shape != (7, 256):
        raise RuntimeError(f"unexpected SIMPLER clean action logits: {clean_action_logits.shape}")
    clean_ids = clean_scores.argmax(dim=-1)
    rows = []
    for bin_index, selected in enumerate(bins):
        replacement = h.copy()
        replacement[np.asarray(selected, dtype=np.int64)] = harmonic_reconstruct(
            h, np.asarray(selected, dtype=np.int64), beta=0.0
        )
        with projector_merge_intervention(
            policy.vla, torch.from_numpy(replacement).unsqueeze(0)
        ):
            negative = guided_forward_scores(policy.vla, inputs, clean_ids, visual.shape[1])
        negative_action_logits = negative[:, start:start + 256].detach().float().cpu().numpy()
        effect = clean_margin_effect(clean_action_logits, negative_action_logits)
        rows.append({
            "rank_bin": bin_index + 1,
            "token_ids": selected,
            **effect,
        })
    return {
        "attention_scores": np.asarray(attention, dtype=np.float32),
        "attention_sha256": attention_meta.get("attention_sha256"),
        "clean_action_logits": clean_action_logits,
        "clean_token_ids": clean_ids.detach().cpu().tolist(),
        "rank_bins": rows,
    }
