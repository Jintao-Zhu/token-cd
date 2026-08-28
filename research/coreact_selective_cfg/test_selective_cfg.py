import numpy as np
import torch

from research.coreact_selective_cfg.features import (
    GEOMETRY_CONSENSUS_FEATURES,
    assert_deployable_feature_names,
    online_geometry_features,
)


def test_consensus_geometry_distinguishes_aligned_directions():
    strong = torch.tensor([[[2.0, 2.0]]])
    aligned = torch.tensor(
        [
            [[[1.0, 1.0]]],
            [[[0.0, 0.0]]],
            [[[1.5, 1.5]]],
            [[[1.25, 1.25]]],
        ]
    )
    valid = torch.ones_like(strong, dtype=torch.bool)
    features = online_geometry_features(strong, aligned, valid)
    assert len(features) == 4
    assert all(np.isclose(row["consensus_concentration"], 1.0) for row in features)
    assert all(np.isclose(row["all_pairwise_cosine_min"], 1.0) for row in features)


def test_feature_contract_rejects_target_leakage():
    assert_deployable_feature_names(GEOMETRY_CONSENSUS_FEATURES)
    try:
        assert_deployable_feature_names(["timestep", "ev_cosine"])
    except ValueError:
        pass
    else:
        raise AssertionError("EV-derived feature was not rejected")
