from __future__ import annotations

import numpy as np

from .analyze import exact_mcnemar, hierarchical_sample_indices


def test_hierarchical_bootstrap_samples_only_whole_task_memberships() -> None:
    tasks = np.asarray([2, 2, 3, 3, 9, 9])
    indices = hierarchical_sample_indices(tasks, np.random.default_rng(11))
    assert indices.shape == (6,)
    assert set(indices).issubset(set(range(6)))
    for start in range(0, 6, 2):
        assert tasks[indices[start]] == tasks[indices[start + 1]]


def test_exact_mcnemar_discordance() -> None:
    result = exact_mcnemar(np.asarray([1, 1, 0, 0]), np.asarray([1, 0, 1, 0]))
    assert result["first_only"] == 1
    assert result["second_only"] == 1
    assert result["discordant"] == 2
    assert result["exact_p"] == 1.0
