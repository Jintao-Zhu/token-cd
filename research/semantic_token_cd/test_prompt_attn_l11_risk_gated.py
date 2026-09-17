import numpy as np
import torch

from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_l11_risk_gated_policy import (
    LOW_LAMBDA,
    PROBE_LAMBDA,
    RiskGatedPromptAttentionSHRInference,
    RiskSignalTracker,
    RiskThresholds,
    gate_decision,
    openvla_normalized_token_values,
    soft_normalized_actions,
)


def test_selector_path_is_inherited_without_override():
    assert RiskGatedPromptAttentionSHRInference._select is PromptAttentionSHRInference._select
    assert RiskGatedPromptAttentionSHRInference.step is PromptAttentionSHRInference.step


def test_openvla_token_values_match_expected_shape_and_order():
    values = openvla_normalized_token_values()
    assert values.shape == (256,)
    assert values[0] == values[1]
    assert values[0] > values[-1]


def test_soft_action_is_finite_and_bounded():
    logits = np.zeros((2, 6, 256), dtype=np.float64)
    actions = soft_normalized_actions(logits)
    assert actions.shape == (2, 6)
    assert np.isfinite(actions).all()
    assert np.max(np.abs(actions)) <= 1.0


def test_tracker_warmup_and_gate_condition():
    tracker = RiskSignalTracker()
    rows = []
    for step in range(5):
        positive = np.array([1.0, 0, 0, 0, 0, 0])
        probe = positive + np.array([(-1.0 if step == 4 else 1.0) * (2.0 if step == 4 else 0.1), 0, 0, 0, 0, 0])
        rows.append(tracker.observe(positive, probe))
    thresholds = RiskThresholds(0.7, 0.0, 1.5)
    assert not any(gate_decision(row, thresholds) for row in rows[:4])
    assert gate_decision(rows[4], thresholds)


def make_policy(mode: str):
    policy = object.__new__(RiskGatedPromptAttentionSHRInference)
    policy.controller_mode = mode
    policy.risk_thresholds = RiskThresholds(0.7, 0.0, 1.5)
    policy.lambd = PROBE_LAMBDA
    policy._risk_tracker = RiskSignalTracker()
    policy.vla = type("VLA", (), {"vocab_size": 32000})()
    return policy


def test_gate_off_is_exact_fixed_lambda_half():
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn((7, 32000), generator=generator)
    negative = torch.randn((7, 32000), generator=generator)
    policy = make_policy("gate_off")
    actual, metadata = policy._combine_action_scores(clean, negative)
    expected = clean.clone()
    expected[:-1] = (1.0 + PROBE_LAMBDA) * clean[:-1] - PROBE_LAMBDA * negative[:-1]
    assert torch.equal(actual, expected)
    assert metadata["final_lambda"] == PROBE_LAMBDA
    assert torch.equal(actual[-1], clean[-1])


def test_force_low_is_exact_fixed_lambda_quarter():
    generator = torch.Generator().manual_seed(11)
    clean = torch.randn((7, 32000), generator=generator)
    negative = torch.randn((7, 32000), generator=generator)
    policy = make_policy("force_low")
    actual, metadata = policy._combine_action_scores(clean, negative)
    expected = clean.clone()
    expected[:-1] = (1.0 + LOW_LAMBDA) * clean[:-1] - LOW_LAMBDA * negative[:-1]
    assert torch.equal(actual, expected)
    assert metadata["final_lambda"] == LOW_LAMBDA
    assert torch.equal(actual[-1], clean[-1])
