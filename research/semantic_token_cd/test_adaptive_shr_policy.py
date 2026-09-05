import torch

from research.semantic_token_cd.adaptive_shr_policy import AdaptiveSHRCDInference


def _policy():
    policy = object.__new__(AdaptiveSHRCDInference)
    policy.lambd = 0.5
    return policy


def test_no_trigger_is_bit_identical_to_locked_shr():
    clean = torch.tensor([[2.0, 0.0]] * 7)
    negative = clean.clone()
    final, meta = _policy()._combine_action_scores(clean, negative)
    expected = clean.clone()
    expected[:-1] = 1.5 * clean[:-1] - 0.5 * negative[:-1]
    assert torch.equal(final, expected)
    assert meta["shift_ratio"] == 0.0
    assert meta["final_lambda"] == 0.5
    assert meta["adapt_trigger"] is False


def test_two_of_six_tokens_trigger_literal_point_33_threshold():
    clean = torch.tensor([[2.0, 0.0]] * 7)
    negative = clean.clone()
    # At lambda=.5 these two rows switch to token 1, while at lambda=.25 they
    # remain vanilla. Other action rows and gripper remain unchanged.
    negative[0] = torch.tensor([2.0, -5.0])
    negative[1] = torch.tensor([2.0, -5.0])
    final, meta = _policy()._combine_action_scores(clean, negative)
    expected = clean.clone()
    expected[:-1] = 1.25 * clean[:-1] - 0.25 * negative[:-1]
    assert torch.equal(final, expected)
    assert meta["shift_count"] == 2
    assert meta["shift_ratio"] == 2 / 6
    assert meta["final_lambda"] == 0.25
    assert meta["adapt_trigger"] is True
    assert meta["gripper_token_unchanged"] is True


def test_one_of_six_tokens_does_not_trigger():
    clean = torch.tensor([[2.0, 0.0]] * 7)
    negative = clean.clone()
    negative[0] = torch.tensor([2.0, -5.0])
    _, meta = _policy()._combine_action_scores(clean, negative)
    assert meta["shift_count"] == 1
    assert meta["shift_ratio"] == 1 / 6
    assert meta["adapt_trigger"] is False


def test_positive_decode_disables_eos_and_requires_seven_scores():
    class FakeVLA:
        def __init__(self):
            self.kwargs = None

        def get_action_dim(self, _unnorm_key):
            return 7

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return {"scores": tuple(torch.zeros(1, 11) for _ in range(7))}

    policy = object.__new__(AdaptiveSHRCDInference)
    policy.vla = FakeVLA()
    inputs = {
        "input_ids": torch.tensor([[1, 29871]]),
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    scores = policy._forward_scores(inputs, "bridge_orig", do_sample=False)
    assert scores.shape == (7, 11)
    assert "eos_token_id" in policy.vla.kwargs
    assert policy.vla.kwargs["eos_token_id"] == []
