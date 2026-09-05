import torch

from research.semantic_token_cd.positive_support_shr_policy import PositiveSupportSHRInference


class _VLA:
    vocab_size = 32000


def test_psc_blocks_unsupported_cd_winner_and_preserves_gripper():
    policy = object.__new__(PositiveSupportSHRInference)
    policy.support_top_k = 10
    policy.vla = _VLA()
    policy._episode_psc_logits = []
    clean = torch.full((7, 32064), -100.0)
    negative = clean.clone()
    start = 31744
    clean[:, start : start + 256] = torch.arange(256).float()
    negative[:, start : start + 256] = clean[:, start : start + 256]
    # Token bin 0 is outside positive Top-10 but raw CD would make it win.
    negative[0, start] = -1000
    final, meta = policy._combine_action_scores(clean, negative)
    assert meta["shr_token_ids_before_constraint"][0] == start
    assert meta["psc_token_ids"][0] == start + 255
    assert meta["changed_by_filter"][0]
    assert meta["gripper_passthrough"]
    assert final[-1].argmax() == clean[-1].argmax()
