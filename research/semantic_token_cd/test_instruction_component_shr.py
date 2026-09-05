import numpy as np

from research.semantic_token_cd.instruction_component_shr_policy import (
    connected_components_4,
    select_instruction_components,
)


def test_four_neighbor_components_do_not_join_diagonals():
    components = connected_components_4([0, 1, 17, 34, 35])
    assert [c.tolist() for c in components] == [[0, 1, 17], [34, 35]]


def test_component_selection_is_atomic_and_topk():
    features = np.zeros((256, 4), dtype=np.float32)
    features[:, 0] = 1.0
    original = [0, 1, 119, 120, 135, 136, 255]
    selected, meta = select_instruction_components(
        features, original, np.array([1.0, 0, 0, 0]), n_keep=2
    )
    selected_set = set(selected.tolist())
    assert meta["num_components"] == 3
    assert meta["selected_num_components"] == 2
    assert meta["component_atomic"] is True
    assert set([119, 120, 135, 136]).issubset(selected_set)
    for component in connected_components_4(original):
        comp = set(component.tolist())
        assert comp.issubset(selected_set) or comp.isdisjoint(selected_set)


def test_single_component_is_noop():
    features = np.ones((256, 3), dtype=np.float32)
    original = [17, 18, 33, 34]
    selected, meta = select_instruction_components(features, original, np.ones(3), n_keep=1)
    assert selected.tolist() == original
    assert meta["mask_tokens_before"] == meta["mask_tokens_after"]
    assert meta["removed_token_fraction"] == 0.0
