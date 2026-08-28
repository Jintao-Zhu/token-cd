from __future__ import annotations

import torch

from research.coreact_exploration.instrumentation import PrefixToken
from research.coreact_value_probe.representation import POOL_ORDER, pool_contextual_prefix


def test_pool_contextual_prefix_uses_only_declared_valid_spans():
    hidden = torch.arange(6 * 2, dtype=torch.float32).reshape(1, 6, 2)
    spans = [[
        PrefixToken(0, "visual", camera_id="camera1", visual_token_index=0, intervention_allowed=True),
        PrefixToken(1, "visual", camera_id="camera1", visual_token_index=1, is_padding=True),
        PrefixToken(2, "visual", camera_id="camera2", visual_token_index=0, intervention_allowed=True),
        PrefixToken(3, "language", language_token_id=1, intervention_allowed=True),
        PrefixToken(4, "state", is_state=True),
        PrefixToken(5, "prefix_padding", is_padding=True),
    ]]
    pooled, indices = pool_contextual_prefix(hidden, spans)
    assert POOL_ORDER == ("camera1", "camera2", "language", "state")
    assert indices == {"camera1": [0], "camera2": [2], "language": [3], "state": [4]}
    assert torch.equal(pooled, torch.cat([hidden[:, i] for i in (0, 2, 3, 4)], dim=-1))


def test_pool_rejects_missing_real_camera():
    hidden = torch.zeros(1, 3, 2)
    spans = [[PrefixToken(0, "visual", camera_id="camera1"), PrefixToken(1, "language"), PrefixToken(2, "state")]]
    try:
        pool_contextual_prefix(hidden, spans)
    except RuntimeError as error:
        assert "no valid tokens" in str(error)
    else:
        raise AssertionError("missing camera2 must be rejected")
