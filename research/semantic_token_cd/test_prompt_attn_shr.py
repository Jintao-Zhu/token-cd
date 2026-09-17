import numpy as np
import torch

from research.semantic_token_cd.prompt_attn_shr_policy import (
    prompt_query_layout,
    stable_top_m,
)


def test_stable_top_m_uses_ascending_index_for_ties():
    scores = np.zeros(256, dtype=np.float32)
    scores[[9, 2, 7]] = 1.0
    assert stable_top_m(scores, 2) == [2, 7]


def test_prompt_layout_excludes_specials_and_applies_visual_shift():
    ids = torch.tensor([[1, 4337, 278, 2, 0]])
    text, multimodal, tokens = prompt_query_layout(ids, {0, 1, 2, 32000})
    assert text == [1, 2]
    assert multimodal == [257, 258]
    assert tokens == [4337, 278]


def test_top_m_is_unique_and_exact():
    scores = np.random.default_rng(4).normal(size=256)
    selected = stable_top_m(scores, 37)
    assert len(selected) == len(set(selected)) == 37
    threshold = np.partition(scores, -37)[-37]
    assert all(scores[index] >= threshold for index in selected)
