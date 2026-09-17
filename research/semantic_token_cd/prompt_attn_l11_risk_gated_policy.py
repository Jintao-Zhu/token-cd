"""Conservative risk-gated lambda controller for L11-Matched Prompt-Attn SHR."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from research.semantic_token_cd.distractor_policy import ACTION_VOCAB_SIZE
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference


PROBE_LAMBDA = 0.5
LOW_LAMBDA = 0.25
MAGNITUDE_EMA_RHO = 0.9
VECTOR_EPS = 1e-6
WARMUP_STEPS = 3
N_CONTINUOUS_ACTIONS = 6
CONTROLLER_MODES = ("risk_gated", "gate_off", "force_low")


def cosine_or_none(current: np.ndarray, previous: np.ndarray | None) -> float | None:
    current = np.asarray(current, dtype=np.float64)
    if previous is None:
        return None
    previous = np.asarray(previous, dtype=np.float64)
    current_norm = float(np.linalg.norm(current))
    previous_norm = float(np.linalg.norm(previous))
    if current_norm < VECTOR_EPS or previous_norm < VECTOR_EPS:
        return None
    return float(np.clip(np.dot(current, previous) / (current_norm * previous_norm), -1.0, 1.0))


def openvla_normalized_token_values() -> np.ndarray:
    bins = np.linspace(-1.0, 1.0, ACTION_VOCAB_SIZE)
    centers = (bins[:-1] + bins[1:]) / 2.0
    discretized = np.arange(ACTION_VOCAB_SIZE - 1, -1, -1)
    discretized = np.clip(discretized, 0, centers.size - 1)
    return centers[discretized].astype(np.float64)


def soft_normalized_actions(action_logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(action_logits, dtype=np.float64)
    if logits.shape[-2:] != (N_CONTINUOUS_ACTIONS, ACTION_VOCAB_SIZE):
        raise ValueError(f"expected [...,6,256] action logits, got {logits.shape}")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities @ openvla_normalized_token_values()


@dataclass(frozen=True)
class RiskThresholds:
    base_cosine: float
    guide_cosine: float
    magnitude_ratio: float


class RiskSignalTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.previous_positive_action: np.ndarray | None = None
        self.previous_guidance: np.ndarray | None = None
        self.guidance_magnitude_ema: float | None = None
        self.step_index = 0

    def observe(self, positive_action: np.ndarray, probe_action: np.ndarray) -> dict:
        positive_action = np.asarray(positive_action, dtype=np.float64)
        probe_action = np.asarray(probe_action, dtype=np.float64)
        if positive_action.shape != (N_CONTINUOUS_ACTIONS,) or probe_action.shape != (N_CONTINUOUS_ACTIONS,):
            raise ValueError("risk tracker expects six-dimensional continuous actions")
        guidance = probe_action - positive_action
        guidance_norm = float(np.linalg.norm(guidance))
        base_cosine = cosine_or_none(positive_action, self.previous_positive_action)
        guide_cosine = cosine_or_none(guidance, self.previous_guidance)
        previous_magnitude_ema = self.guidance_magnitude_ema
        magnitude_ratio = (
            guidance_norm / max(previous_magnitude_ema, VECTOR_EPS)
            if previous_magnitude_ema is not None and previous_magnitude_ema >= VECTOR_EPS
            else None
        )
        signals_valid = (
            self.step_index >= WARMUP_STEPS
            and base_cosine is not None
            and guide_cosine is not None
            and magnitude_ratio is not None
            and np.isfinite(magnitude_ratio)
        )
        risk_score = (
            max(base_cosine, 0.0)
            * max(-guide_cosine, 0.0)
            * max(magnitude_ratio - 1.0, 0.0)
            if signals_valid else 0.0
        )
        self.previous_positive_action = positive_action.copy()
        self.previous_guidance = guidance.copy()
        self.guidance_magnitude_ema = (
            guidance_norm
            if previous_magnitude_ema is None
            else MAGNITUDE_EMA_RHO * previous_magnitude_ema
            + (1.0 - MAGNITUDE_EMA_RHO) * guidance_norm
        )
        metadata = {
            "step_index": self.step_index,
            "positive_soft_action": positive_action.tolist(),
            "probe_soft_action": probe_action.tolist(),
            "guidance": guidance.tolist(),
            "base_cosine": base_cosine,
            "guide_cosine": guide_cosine,
            "guidance_norm": guidance_norm,
            "previous_guidance_magnitude_ema": previous_magnitude_ema,
            "guidance_magnitude_ema": self.guidance_magnitude_ema,
            "guidance_magnitude_ratio": magnitude_ratio,
            "risk_score": float(risk_score),
            "signals_valid": bool(signals_valid),
        }
        self.step_index += 1
        return metadata


def gate_decision(metadata: dict, thresholds: RiskThresholds) -> bool:
    return bool(
        metadata["signals_valid"]
        and metadata["base_cosine"] > thresholds.base_cosine
        and metadata["guide_cosine"] < thresholds.guide_cosine
        and metadata["guidance_magnitude_ratio"] > thresholds.magnitude_ratio
    )


class RiskGatedPromptAttentionSHRInference(PromptAttentionSHRInference):
    controller_mode: str = "risk_gated"
    risk_thresholds = RiskThresholds(0.7, 0.0, 1.5)

    def reset(self, task_description: str, seed=None) -> None:
        super().reset(task_description, seed)
        self._risk_tracker = RiskSignalTracker()

    def _soft_normalized_action(self, scores: torch.Tensor) -> np.ndarray:
        start = int(self.vla.vocab_size) - ACTION_VOCAB_SIZE
        logits = scores[:N_CONTINUOUS_ACTIONS, start : start + ACTION_VOCAB_SIZE].float()
        if logits.shape != (N_CONTINUOUS_ACTIONS, ACTION_VOCAB_SIZE):
            raise RuntimeError(f"invalid action-vocabulary logits: {tuple(logits.shape)}")
        token_values = torch.as_tensor(
            openvla_normalized_token_values(), device=logits.device, dtype=logits.dtype
        )
        expected = torch.softmax(logits, dim=-1) @ token_values
        return expected.detach().cpu().numpy().astype(np.float64)

    def _combine_action_scores(
        self, clean_scores: torch.Tensor, negative_scores: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        if clean_scores.shape[0] != N_CONTINUOUS_ACTIONS + 1:
            raise RuntimeError(f"Risk-gated lambda expects 7 action positions: {clean_scores.shape}")
        if self.controller_mode not in CONTROLLER_MODES:
            raise ValueError(f"invalid controller mode: {self.controller_mode}")
        if abs(float(self.lambd) - PROBE_LAMBDA) > 1e-12:
            raise RuntimeError("Risk-gated probe must remain locked at lambda=0.5")

        probe_scores = clean_scores.clone()
        probe_scores[:-1] = (
            (1.0 + PROBE_LAMBDA) * clean_scores[:-1]
            - PROBE_LAMBDA * negative_scores[:-1]
        )
        positive_soft_action = self._soft_normalized_action(clean_scores)
        probe_soft_action = self._soft_normalized_action(probe_scores)
        risk_metadata = self._risk_tracker.observe(positive_soft_action, probe_soft_action)
        if self.controller_mode == "gate_off":
            gate_trigger = False
        elif self.controller_mode == "force_low":
            gate_trigger = True
        else:
            gate_trigger = gate_decision(risk_metadata, self.risk_thresholds)
        final_lambda = LOW_LAMBDA if gate_trigger else PROBE_LAMBDA

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1.0 + final_lambda) * clean_scores[:-1]
            - final_lambda * negative_scores[:-1]
        )
        positive_ids = clean_scores.argmax(dim=-1)
        final_ids = final_scores.argmax(dim=-1)
        return final_scores, {
            "controller_mode": self.controller_mode,
            "probe_lambda": PROBE_LAMBDA,
            "final_lambda": final_lambda,
            "low_lambda": LOW_LAMBDA,
            "gate_trigger": gate_trigger,
            "threshold_base_cosine": self.risk_thresholds.base_cosine,
            "threshold_guide_cosine": self.risk_thresholds.guide_cosine,
            "threshold_magnitude_ratio": self.risk_thresholds.magnitude_ratio,
            "magnitude_ema_rho": MAGNITUDE_EMA_RHO,
            "warmup_steps": WARMUP_STEPS,
            "gripper_excluded_from_controller": True,
            "gripper_token_unchanged": bool(final_ids[-1].item() == positive_ids[-1].item()),
            "positive_action_token_ids": positive_ids.detach().cpu().tolist(),
            "risk_gated_action_token_ids": final_ids.detach().cpu().tolist(),
            **risk_metadata,
        }
