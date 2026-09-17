"""CPU-only selector invariants for PA-Rerank-L11-Matched v1."""
import numpy as np

from research.semantic_token_cd.prompt_attn_shr_policy import (
    construct_prompt_action_bounded,
    construct_prompt_action_full_joint,
    construct_prompt_action_rerank,
    stable_top_m,
)


def test_prompt_score_second_stage_recovers_l11_matched():
    rng = np.random.default_rng(17)
    prompt = rng.normal(size=256)
    for m in (1, 16, 35, 64, 100):
        result = construct_prompt_action_rerank(prompt, prompt, m, 3)
        assert result["selected"] == stable_top_m(prompt, m)


def test_rerank_is_exact_and_stays_inside_prompt_pool():
    prompt = np.linspace(1.0, 0.0, 256)
    action = np.linspace(0.0, 1.0, 256)
    result = construct_prompt_action_rerank(prompt, action, 35, 3)
    assert len(result["selected"]) == 35
    assert len(set(result["selected"])) == 35
    assert set(result["selected"]).issubset(result["candidates"])
    assert result["candidate_count"] == 105
    assert result["candidate_pool_saturated"] is False


def test_saturated_pool_is_explicit_and_deterministic():
    prompt = np.zeros(256)
    action = np.zeros(256)
    result = construct_prompt_action_rerank(prompt, action, 100, 3)
    assert result["candidate_count"] == 256
    assert result["candidate_pool_saturated"] is True
    assert result["candidates"] == list(range(256))
    assert result["selected"] == list(range(100))


def test_bounded_joint_recovers_prompt_when_scores_match():
    rng = np.random.default_rng(23)
    prompt = rng.normal(size=256)
    for m in (16, 35, 64, 100):
        result = construct_prompt_action_bounded(prompt, prompt, m, "joint")
        assert result["selected"] == stable_top_m(prompt, m)


def test_bounded_modes_protect_core_and_cap_replacements():
    prompt = np.linspace(1.0, 0.0, 256)
    action = np.linspace(0.0, 1.0, 256)
    for mode in ("action_only", "joint"):
        result = construct_prompt_action_bounded(prompt, action, 35, mode)
        assert len(result["selected"]) == 35
        assert result["core_count"] == 32
        assert result["tail_count"] == 3
        assert set(result["core"]).issubset(result["selected"])
        assert set(result["selected"]).issubset(result["candidates"])
        assert len(set(result["selected"]) - set(result["original_prompt"])) <= 3


def test_full_joint_uses_all_k_and_recovers_prompt_when_scores_match():
    rng = np.random.default_rng(29)
    prompt = rng.normal(size=256)
    same = construct_prompt_action_full_joint(prompt, prompt, 35)
    assert same["selected"] == stable_top_m(prompt, 35)
    action = -prompt
    changed = construct_prompt_action_full_joint(prompt, action, 35)
    assert len(changed["selected"]) == 35
    assert set(changed["selected"]).issubset(changed["candidates"])
    assert changed["candidate_count"] == 70
