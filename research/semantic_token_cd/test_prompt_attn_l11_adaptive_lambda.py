import numpy as np

from research.semantic_token_cd.prompt_attn_l11_adaptive_lambda_policy import (
    PROBE_LAMBDA,
    AdaptiveLambdaPromptAttentionSHRInference,
    correction_cosine,
    lambda_from_similarity,
)


def make_policy(mode: str):
    policy = object.__new__(AdaptiveLambdaPromptAttentionSHRInference)
    policy.controller_mode = mode
    policy._previous_probe_correction = None
    policy._ema_similarity = None
    policy._previous_adaptive_lambda = PROBE_LAMBDA
    return policy


def test_cosine_and_mapping_bounds():
    assert correction_cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert correction_cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == -1.0
    assert correction_cosine(np.zeros(2), np.ones(2)) is None
    assert lambda_from_similarity(1.0) == 0.5
    assert lambda_from_similarity(0.0) == 0.1
    assert lambda_from_similarity(-1.0) == 0.1


def test_raw_controller_uses_current_temporal_consistency():
    policy = make_policy("adaptive_raw")
    first_lambda, first_meta = policy._choose_lambda(np.array([1.0, 0.0]))
    second_lambda, second_meta = policy._choose_lambda(np.array([1.0, 0.0]))
    third_lambda, third_meta = policy._choose_lambda(np.array([-1.0, 0.0]))
    assert first_lambda == 0.5
    assert first_meta["correction_cosine"] is None
    assert second_lambda == 0.5
    assert second_meta["correction_cosine"] == 1.0
    assert third_lambda == 0.1
    assert third_meta["correction_cosine"] == -1.0


def test_ema_controller_smooths_similarity():
    policy = make_policy("adaptive_ema")
    policy._choose_lambda(np.array([1.0, 0.0]))
    aligned_lambda, _ = policy._choose_lambda(np.array([1.0, 0.0]))
    reversed_lambda, meta = policy._choose_lambda(np.array([-1.0, 0.0]))
    assert aligned_lambda == 0.5
    assert meta["ema_similarity"] == 0.0
    assert reversed_lambda == 0.1


def test_fixed_low_controller_is_exact():
    policy = make_policy("fixed_025")
    value, _ = policy._choose_lambda(np.array([1.0, 0.0]))
    assert value == 0.25
