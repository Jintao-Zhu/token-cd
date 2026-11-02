import numpy as np

from research.semantic_token_cd.pi05_matched_flow_cd import (
    N_VISUAL,
    guided_velocity_numpy,
    libero_guided_dim_mask,
    make_harmonic_negative,
    stable_top_m,
)


def test_stable_top_m_uses_token_id_as_tie_breaker():
    scores = np.zeros((N_VISUAL,), dtype=np.float32)
    scores[[9, 3, 7]] = 1.0
    assert stable_top_m(scores, 2) == [3, 7]


def test_harmonic_negative_changes_only_selected_tokens():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(N_VISUAL, 8)).astype(np.float32)
    selected = [17, 18, 33, 34]
    negative = make_harmonic_negative(features, selected)
    untouched = np.ones((N_VISUAL,), dtype=bool)
    untouched[selected] = False
    assert np.array_equal(negative[untouched], features[untouched])
    assert np.isfinite(negative).all()
    assert not np.array_equal(negative[selected], features[selected])


def test_lambda_zero_is_bit_exact():
    rng = np.random.default_rng(11)
    positive = rng.normal(size=(1, 10, 32)).astype(np.float32)
    negative = rng.normal(size=(1, 10, 32)).astype(np.float32)
    guided = guided_velocity_numpy(
        positive,
        negative,
        guidance_scale=0.0,
        guided_dim_mask=libero_guided_dim_mask(32),
    )
    assert np.array_equal(guided, positive)


def test_identical_branches_are_bit_exact():
    rng = np.random.default_rng(13)
    positive = rng.normal(size=(1, 10, 32)).astype(np.float32)
    guided = guided_velocity_numpy(
        positive,
        positive.copy(),
        guidance_scale=0.5,
        guided_dim_mask=libero_guided_dim_mask(32),
    )
    assert np.array_equal(guided, positive)


def test_guidance_never_changes_gripper_or_padding_dimensions():
    positive = np.zeros((1, 2, 32), dtype=np.float32)
    negative = np.ones_like(positive)
    guided = guided_velocity_numpy(
        positive,
        negative,
        guidance_scale=0.5,
        guided_dim_mask=libero_guided_dim_mask(32),
    )
    assert np.array_equal(guided[..., :6], np.full((1, 2, 6), -0.5, dtype=np.float32))
    assert np.array_equal(guided[..., 6:], positive[..., 6:])
