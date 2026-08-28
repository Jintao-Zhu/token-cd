import math

import torch

from research.coreact_local_ev_confirmation.capture_object import local_features
from research.coreact_local_ev_confirmation.modeling import LOCAL_FEATURES, PRIMARY_BRANCH
from research.coreact_selective_cfg.features import assert_deployable_feature_names


def test_confirmation_contract_is_local_only_and_target_free():
    assert PRIMARY_BRANCH == "W4_half_last_2"
    assert len(LOCAL_FEATURES) == 7
    assert not any("consensus" in name or "pairwise" in name for name in LOCAL_FEATURES)
    assert_deployable_feature_names(LOCAL_FEATURES)


def test_local_feature_formula_matches_frozen_contract():
    strong = torch.tensor([[3.0, 4.0], [9.0, 9.0]])
    weak = torch.tensor([[0.0, 4.0], [8.0, 8.0]])
    valid = torch.tensor([[True, True], [False, False]])
    result = local_features(strong, weak, valid)
    assert set(result) == set(LOCAL_FEATURES) - {"timestep"}
    assert math.isclose(result["relative_correction_norm"], 3.0 / 5.0)
    assert math.isclose(
        result["strong_weak_cosine"], 16.0 / (5.0 * 4.0), rel_tol=1e-6
    )
    assert all(math.isfinite(value) for value in result.values())
