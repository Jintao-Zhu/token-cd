from __future__ import annotations

import torch

from research.coreact_revision.capture_action_consistency import CONDITION_SUBSETS, subset_features


def test_condition_subsets_are_locked_and_valid() -> None:
    assert CONDITION_SUBSETS == {
        "full6": (0, 1, 2, 3, 4, 5),
        "tau05_noise2": (2, 3),
        "tau3_seed1729": (0, 2, 4),
        "tau3_seed9473": (1, 3, 5),
        "extremes4": (0, 1, 4, 5),
    }
    assert all(len(indices) >= 2 and len(set(indices)) == len(indices) for indices in CONDITION_SUBSETS.values())


def test_subset_features_zero_for_identical_clean_and_masked() -> None:
    clean = torch.tensor([[[1.0, 2.0]], [[2.0, 1.0]], [[1.5, 1.5]]])
    features = subset_features(clean, clean.clone(), (0, 2))
    assert features["masked_minus_clean_dispersion"] == 0.0
    assert features["clean_mask_consensus_distance"] == 0.0
    assert features["mean_intervention_norm"] == 0.0
    assert abs(features["intervention_direction_consistency"]) < 1e-7
