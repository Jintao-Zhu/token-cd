from __future__ import annotations

import torch
import inspect

from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from research.coreact_flow_timestep_compatibility.metrics import exact_training_pair


def test_exact_flow_training_pair_and_integration_direction():
    actions = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    noise = torch.tensor([[[-0.5, 4.0], [2.0, -1.0]]])
    time = torch.tensor([0.7])
    expanded = time[:, None, None]
    source = inspect.getsource(VLAFlowMatching.forward)
    assert "x_t = time_expanded * noise + (1 - time_expanded) * actions" in source
    assert "u_t = noise - actions" in source
    x_t, target = exact_training_pair(actions, noise, time)
    torch.testing.assert_close(x_t, expanded * noise + (1.0 - expanded) * actions)
    torch.testing.assert_close(target, noise - actions)
    dt = -0.1
    torch.testing.assert_close(x_t + dt * target, (expanded + dt) * noise + (1.0 - expanded - dt) * actions)
    assert dt < 0
