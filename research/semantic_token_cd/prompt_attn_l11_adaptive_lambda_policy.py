"""Temporal-consistency Adaptive-lambda controller for L11-Matched Prompt-Attn SHR."""
from __future__ import annotations

import numpy as np
import torch

from research.semantic_token_cd.distractor_policy import ACTION_VOCAB_SIZE
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference


PROBE_LAMBDA = 0.5
MIN_LAMBDA = 0.1
FIXED_LOW_LAMBDA = 0.25
EMA_RHO = 0.5
CORRECTION_EPS = 1e-6
N_CONTINUOUS_ACTIONS = 6
CONTROLLER_MODES = ("fixed_025", "adaptive_raw", "adaptive_ema")


def correction_cosine(current: np.ndarray, previous: np.ndarray | None) -> float | None:
    current = np.asarray(current, dtype=np.float64)
    if previous is None:
        return None
    previous = np.asarray(previous, dtype=np.float64)
    current_norm = float(np.linalg.norm(current))
    previous_norm = float(np.linalg.norm(previous))
    if current_norm < CORRECTION_EPS or previous_norm < CORRECTION_EPS:
        return None
    value = float(np.dot(current, previous) / (current_norm * previous_norm))
    return float(np.clip(value, -1.0, 1.0))


def lambda_from_similarity(similarity: float) -> float:
    positive_similarity = float(np.clip(similarity, 0.0, 1.0))
    return MIN_LAMBDA + (PROBE_LAMBDA - MIN_LAMBDA) * positive_similarity


class AdaptiveLambdaPromptAttentionSHRInference(PromptAttentionSHRInference):
    """L11-Matched selector with fixed-low, raw-cosine, or EMA-cosine lambda."""

    controller_mode: str = "adaptive_raw"

    def reset(self, task_description: str, seed=None) -> None:
        super().reset(task_description, seed)
        self._previous_probe_correction: np.ndarray | None = None
        self._ema_similarity: float | None = None
        self._previous_adaptive_lambda = PROBE_LAMBDA

    def _soft_normalized_action(self, scores: torch.Tensor) -> np.ndarray:
        start = int(self.vla.vocab_size) - ACTION_VOCAB_SIZE
        logits = scores[:N_CONTINUOUS_ACTIONS, start : start + ACTION_VOCAB_SIZE].float()
        if logits.shape != (N_CONTINUOUS_ACTIONS, ACTION_VOCAB_SIZE):
            raise RuntimeError(f"invalid action-vocabulary logits: {tuple(logits.shape)}")
        centers = np.asarray(self.vla.bin_centers, dtype=np.float32)
        if centers.ndim != 1 or centers.size < 2:
            raise RuntimeError(f"invalid OpenVLA bin centers: {centers.shape}")
        discretized = np.arange(ACTION_VOCAB_SIZE - 1, -1, -1)
        discretized = np.clip(discretized, 0, centers.size - 1)
        token_values = torch.as_tensor(
            centers[discretized].copy(), device=logits.device, dtype=logits.dtype
        )
        expected = torch.softmax(logits, dim=-1) @ token_values
        return expected.detach().cpu().numpy().astype(np.float64)

    def _choose_lambda(self, probe_correction: np.ndarray) -> tuple[float, dict]:
        similarity = correction_cosine(probe_correction, self._previous_probe_correction)
        effective_similarity = similarity
        if self.controller_mode == "fixed_025":
            final_lambda = FIXED_LOW_LAMBDA
        elif similarity is None:
            final_lambda = self._previous_adaptive_lambda
        elif self.controller_mode == "adaptive_raw":
            final_lambda = lambda_from_similarity(similarity)
        elif self.controller_mode == "adaptive_ema":
            if self._ema_similarity is None:
                self._ema_similarity = similarity
            else:
                self._ema_similarity = (
                    EMA_RHO * self._ema_similarity + (1.0 - EMA_RHO) * similarity
                )
            effective_similarity = self._ema_similarity
            final_lambda = lambda_from_similarity(effective_similarity)
        else:
            raise ValueError(f"unknown Adaptive-lambda controller mode: {self.controller_mode}")

        self._previous_probe_correction = np.asarray(probe_correction, dtype=np.float64).copy()
        self._previous_adaptive_lambda = float(final_lambda)
        return float(final_lambda), {
            "correction_cosine": similarity,
            "effective_correction_cosine": effective_similarity,
            "ema_similarity": self._ema_similarity,
        }

    def _combine_action_scores(
        self, clean_scores: torch.Tensor, negative_scores: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        if clean_scores.shape[0] != N_CONTINUOUS_ACTIONS + 1:
            raise RuntimeError(f"Adaptive-lambda expects 7 action positions: {clean_scores.shape}")
        if self.controller_mode not in CONTROLLER_MODES:
            raise ValueError(f"invalid controller mode: {self.controller_mode}")
        if abs(float(self.lambd) - PROBE_LAMBDA) > 1e-12:
            raise RuntimeError("Adaptive-lambda probe must remain locked at lambda=0.5")

        probe_scores = clean_scores.clone()
        probe_scores[:-1] = (
            (1.0 + PROBE_LAMBDA) * clean_scores[:-1]
            - PROBE_LAMBDA * negative_scores[:-1]
        )
        positive_soft_action = self._soft_normalized_action(clean_scores)
        probe_soft_action = self._soft_normalized_action(probe_scores)
        probe_correction = probe_soft_action - positive_soft_action
        final_lambda, controller_meta = self._choose_lambda(probe_correction)

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1.0 + final_lambda) * clean_scores[:-1]
            - final_lambda * negative_scores[:-1]
        )
        positive_ids = clean_scores.argmax(dim=-1)
        probe_ids = probe_scores.argmax(dim=-1)
        final_ids = final_scores.argmax(dim=-1)
        correction_norm = float(np.linalg.norm(probe_correction))
        return final_scores, {
            "controller_mode": self.controller_mode,
            "probe_lambda": PROBE_LAMBDA,
            "final_lambda": final_lambda,
            "lambda_min": MIN_LAMBDA,
            "lambda_max": PROBE_LAMBDA,
            "fixed_low_lambda": FIXED_LOW_LAMBDA,
            "ema_rho": EMA_RHO if self.controller_mode == "adaptive_ema" else None,
            "correction_epsilon": CORRECTION_EPS,
            "positive_soft_action": positive_soft_action.tolist(),
            "probe_soft_action": probe_soft_action.tolist(),
            "probe_correction": probe_correction.tolist(),
            "probe_correction_norm": correction_norm,
            "correction_norm_valid": correction_norm >= CORRECTION_EPS,
            "positive_action_token_ids": positive_ids.detach().cpu().tolist(),
            "probe_action_token_ids": probe_ids.detach().cpu().tolist(),
            "adaptive_action_token_ids": final_ids.detach().cpu().tolist(),
            "gripper_excluded_from_controller": True,
            "gripper_token_unchanged": bool(final_ids[-1].item() == positive_ids[-1].item()),
            **controller_meta,
        }
