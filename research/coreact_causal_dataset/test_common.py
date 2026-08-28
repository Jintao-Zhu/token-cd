from types import SimpleNamespace

from research.coreact_causal_dataset.common import paired_effect, select_phase_index, stratified_token_groups


def test_paired_effect_sign_and_threshold():
    assert paired_effect([1, 1, 1], [0, 0, 1])["tau"] == -2 / 3
    assert paired_effect([1, 1, 1], [0, 0, 1])["strong_anchor"]
    assert paired_effect([0, 0, 0], [1, 1, 0])["strong_nuisance"]


def test_stratified_groups_are_deterministic_disjoint():
    scores = [float(i) for i in range(20)]
    first = stratified_token_groups(list(range(20)), scores, 123)
    second = stratified_token_groups(list(range(20)), scores, 123)
    assert first == second
    assert len(first) == 8
    assert len({row["token_indices"][0] for row in first}) == 8


def test_phase_selection_uses_grasp_metadata():
    rows = [
        {"grasped": False, "predicate": False, "object_goal_distance": 1.0},
        {"grasped": False, "predicate": False, "object_goal_distance": 0.9},
        {"grasped": True, "predicate": False, "object_goal_distance": 0.8},
        {"grasped": True, "predicate": False, "object_goal_distance": 0.3},
        {"grasped": True, "predicate": True, "object_goal_distance": 0.1},
    ]
    assert select_phase_index(rows, "pre_grasp") == (0, "pre_grasp", False)
    assert select_phase_index(rows, "grasp_contact") == (2, "grasp_contact", False)
    assert select_phase_index(rows, "pre_place") == (3, "pre_place", False)
