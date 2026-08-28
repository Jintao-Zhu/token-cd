from __future__ import annotations

import torch

from research.coreact_self_guidance.sampler import apply_self_guidance_velocity
from research.coreact_self_guidance.run import select_pair_shard_rows
from research.coreact_self_guidance.relative_sampler import relative_reference_coordinates
from research.coreact_self_guidance.timestep_sampler import noisier_timestep


def test_w1_is_native_reference_and_zero_token_change() -> None:
    clean = torch.randn(1, 4, 9)
    negative = torch.randn(1, 4, 9)
    output, _, _, _ = apply_self_guidance_velocity(clean, negative, w=1.0, skipped=False, pure_negative=False, action_dim=7, trust_region_kappa=0.25)
    assert torch.equal(output, clean)


def test_boundary_steps_are_skipped() -> None:
    clean = torch.ones(1, 4, 7)
    negative = torch.zeros_like(clean)
    output, _, applied, _ = apply_self_guidance_velocity(clean, negative, w=1.5, skipped=True, pure_negative=False, action_dim=7, trust_region_kappa=0.25)
    assert torch.equal(output, clean)
    assert torch.equal(applied, torch.zeros_like(applied))


def test_pair_sharding_keeps_all_arms_together() -> None:
    manifest = [
        {"pair_id": f"pair-{pair}", "arm": f"arm-{arm}"}
        for pair in range(11)
        for arm in range(6)
    ]
    ownership = {}
    for shard in range(4):
        rows = select_pair_shard_rows(manifest, shard, 4)
        for row in rows:
            assert ownership.setdefault(row["pair_id"], shard) == shard
    assert len(ownership) == 11


def test_relative_lag_is_valid_and_active_on_nine_of_ten_steps() -> None:
    coordinates = [relative_reference_coordinates(step, 0.3, 10) for step in range(10)]
    assert coordinates[0][:2] == (0.0, 0.0)
    assert sum(negative < progress for progress, negative, *_ in coordinates) == 9
    assert all(0 <= lower <= upper < 10 for _, _, lower, upper, _ in coordinates)


def test_timestep_shift_keeps_state_time_grid_and_moves_noisier() -> None:
    taus = [1.0 - step / 10 for step in range(10)]
    shifted = [noisier_timestep(tau, 0.2) for tau in taus]
    assert shifted[0] == 1.0
    assert sum(after > before for before, after in zip(taus, shifted, strict=True)) == 9
    assert all(0.0 <= value <= 1.0 for value in shifted)
