"""Instance-segmentation image interventions before SmolVLA preprocessing."""

from __future__ import annotations

import copy
from collections import deque

import cv2
import numpy as np

from research.coreact_region.effect_existence import array_sha256, canonical_json_sha256
from research.coreact_region.region_mapping import instance_label_map, protected_instance_names
from research.coreact_region.segmented_runtime import task_entities


CAMERAS = (
    ("camera1", "image", "agentview_image", "agentview_segmentation_instance"),
    ("camera2", "image2", "robot0_eye_in_hand_image", "robot0_eye_in_hand_segmentation_instance"),
)
SEMANTICS = ("target", "goal", "background_match_target", "background_match_goal")
REPLACEMENTS = ("blur", "inpaint", "black")
MASK_DILATION_RADIUS = 5
BLUR_KERNEL = 31
INPAINT_RADIUS = 7


def _segmentation_2d(value: np.ndarray) -> np.ndarray:
    if value.shape == (256, 256, 1):
        value = value[..., 0]
    if value.shape != (256, 256):
        raise ValueError(f"expected 256x256 segmentation, got {value.shape}")
    return value


def resolve_segmentation_entities(env) -> tuple[str, str]:
    target, logical_goal, _ = task_entities(env)
    instances = set(env._env.env.model.instances_to_ids)
    if target not in instances:
        raise ValueError(f"target instance {target} is absent from segmentation metadata")
    goal_candidates = [logical_goal, logical_goal.removesuffix("_contain_region")]
    goal_candidates.extend(name for name in env._env.obj_of_interest if name != target)
    goal = next((name for name in goal_candidates if name in instances), None)
    if goal is None:
        raise ValueError(f"cannot resolve logical goal {logical_goal} to a segmentation instance")
    return target, goal


def _dilated_instance_mask(
    segmentation: np.ndarray,
    *,
    instance_label: int,
    protected_labels: set[int],
) -> np.ndarray:
    exact = segmentation == instance_label
    kernel_size = MASK_DILATION_RADIUS * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    expanded = cv2.dilate(exact.astype(np.uint8), kernel, iterations=1).astype(bool)
    if protected_labels:
        expanded &= ~np.isin(segmentation, list(protected_labels))
    if not np.all(expanded[exact]):
        raise RuntimeError("dilation removed exact instance pixels")
    return expanded


def connected_background_mask(
    eligible: np.ndarray,
    count: int,
    *,
    random_seed: int,
) -> np.ndarray:
    """Select exact-area eligible pixels, preferring one connected component."""
    output = np.zeros_like(eligible, dtype=bool)
    if count == 0:
        return output
    component_count, labels = cv2.connectedComponents(eligible.astype(np.uint8), connectivity=4)
    rng = np.random.default_rng(random_seed)
    component_sizes = {
        label: int(np.sum(labels == label)) for label in range(1, component_count)
    }
    if sum(component_sizes.values()) < count:
        raise ValueError(f"only {sum(component_sizes.values())} eligible background pixels for {count}")
    single = [label for label, size in component_sizes.items() if size >= count]
    if single:
        component_order = [single[int(rng.integers(0, len(single)))]]
    else:
        component_order = [int(item) for item in rng.permutation(list(component_sizes))]
    selected: list[tuple[int, int]] = []
    height, width = eligible.shape
    for label in component_order:
        coordinates = np.argwhere(labels == label)
        start = tuple(coordinates[int(rng.integers(0, len(coordinates)))])
        queue = deque([start])
        visited = {start}
        while queue and len(selected) < count:
            row, column = queue.popleft()
            selected.append((row, column))
            neighbors = [(row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)]
            for position in rng.permutation(4):
                next_row, next_column = neighbors[int(position)]
                item = (next_row, next_column)
                if (
                    0 <= next_row < height and 0 <= next_column < width
                    and labels[next_row, next_column] == label and item not in visited
                ):
                    visited.add(item)
                    queue.append(item)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"background traversal selected {len(selected)} of {count} pixels")
    rows, columns = zip(*selected, strict=True)
    output[np.asarray(rows), np.asarray(columns)] = True
    return output


def build_image_masks(env, raw: dict, *, random_seed: int) -> tuple[dict, dict]:
    target_name, goal_name = resolve_segmentation_entities(env)
    instances = env._env.env.model.instances_to_ids
    labels = instance_label_map(instances)
    protected_labels = {labels[name] for name in protected_instance_names(instances)}
    camera_data = {}
    for camera_index, (camera_id, _, _, segmentation_key) in enumerate(CAMERAS):
        segmentation = _segmentation_2d(raw[segmentation_key])
        target = _dilated_instance_mask(
            segmentation, instance_label=labels[target_name], protected_labels=protected_labels
        )
        goal = _dilated_instance_mask(
            segmentation, instance_label=labels[goal_name], protected_labels=protected_labels
        )
        excluded_labels = protected_labels | {labels[target_name], labels[goal_name]}
        eligible = ~np.isin(segmentation, list(excluded_labels))
        eligible &= ~target & ~goal
        camera_data[camera_id] = {
            "camera_index": camera_index, "segmentation": segmentation,
            "target": target, "goal": goal, "eligible": eligible,
        }

    background_masks = {camera_id: {} for camera_id in camera_data}
    for semantic_index, semantic in enumerate(("target", "goal"), 1):
        requested = {
            camera_id: int(data[semantic].sum()) for camera_id, data in camera_data.items()
        }
        capacity = {
            camera_id: int(data["eligible"].sum()) for camera_id, data in camera_data.items()
        }
        allocation = {
            camera_id: min(requested[camera_id], capacity[camera_id]) for camera_id in camera_data
        }
        deficit = sum(requested.values()) - sum(allocation.values())
        for camera_id in camera_data:
            available = capacity[camera_id] - allocation[camera_id]
            transferred = min(deficit, available)
            allocation[camera_id] += transferred
            deficit -= transferred
        if deficit:
            raise ValueError(
                f"two-view eligible background is short by {deficit} pixels for {semantic}"
            )
        for camera_id, data in camera_data.items():
            background_masks[camera_id][f"background_match_{semantic}"] = connected_background_mask(
                data["eligible"], allocation[camera_id],
                random_seed=random_seed + data["camera_index"] * 100 + semantic_index,
            )

    bundle, camera_audits = {}, {}
    for camera_id, data in camera_data.items():
        segmentation = data["segmentation"]
        target, goal = data["target"], data["goal"]
        background_target = background_masks[camera_id]["background_match_target"]
        background_goal = background_masks[camera_id]["background_match_goal"]
        eligible = data["eligible"]
        bundle[camera_id] = {
            "target": target,
            "goal": goal,
            "background_match_target": background_target,
            "background_match_goal": background_goal,
        }
        camera_audits[camera_id] = {
            "target_pixels": int(target.sum()), "goal_pixels": int(goal.sum()),
            "background_match_target_pixels": int(background_target.sum()),
            "background_match_goal_pixels": int(background_goal.sum()),
            "target_exact_pixels": int(np.sum(segmentation == labels[target_name])),
            "goal_exact_pixels": int(np.sum(segmentation == labels[goal_name])),
            "background_target_overlap_excluded": int(np.sum(background_target & ~eligible)),
            "background_goal_overlap_excluded": int(np.sum(background_goal & ~eligible)),
            "background_target_label_zero_fraction": float(np.mean(segmentation[background_target] == 0)) if np.any(background_target) else 1.0,
            "background_goal_label_zero_fraction": float(np.mean(segmentation[background_goal] == 0)) if np.any(background_goal) else 1.0,
        }
    camera_audits["two_view_totals"] = {
        "target_pixels": sum(int(data["target"].sum()) for data in camera_data.values()),
        "goal_pixels": sum(int(data["goal"].sum()) for data in camera_data.values()),
        "background_match_target_pixels": sum(int(mask["background_match_target"].sum()) for mask in background_masks.values()),
        "background_match_goal_pixels": sum(int(mask["background_match_goal"].sum()) for mask in background_masks.values()),
    }
    serializable = {
        camera: {semantic: array_sha256(mask) for semantic, mask in masks.items()}
        for camera, masks in bundle.items()
    }
    audit = {
        "target_instance": target_name, "goal_instance": goal_name,
        "mask_dilation_radius_pixels": MASK_DILATION_RADIUS,
        "camera_counts": camera_audits, "mask_hashes": serializable,
        "bundle_sha256": canonical_json_sha256(serializable),
    }
    return bundle, audit


def replace_pixels(image: np.ndarray, mask: np.ndarray, replacement: str) -> np.ndarray:
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB 256x256x3, got {image.dtype} {image.shape}")
    if mask.shape != image.shape[:2] or mask.dtype != bool:
        raise ValueError("image mask shape/dtype mismatch")
    output = image.copy()
    if not np.any(mask):
        return output
    if replacement == "blur":
        source = cv2.GaussianBlur(image, (BLUR_KERNEL, BLUR_KERNEL), sigmaX=0)
    elif replacement == "inpaint":
        source = cv2.inpaint(image, mask.astype(np.uint8) * 255, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    elif replacement == "black":
        source = np.zeros_like(image)
    else:
        raise ValueError(f"unsupported image replacement: {replacement}")
    output[mask] = source[mask]
    if not np.array_equal(output[~mask], image[~mask]):
        raise RuntimeError("replacement changed pixels outside the locked mask")
    return output


def apply_image_condition(observation: dict, bundle: dict, condition: str) -> tuple[dict, dict]:
    if condition == "vanilla":
        return copy.deepcopy(observation), {"condition": condition, "changed_pixels": 0}
    semantic, replacement = condition.rsplit("_", 1)
    if semantic not in SEMANTICS or replacement not in REPLACEMENTS:
        raise ValueError(f"invalid image condition: {condition}")
    output = copy.deepcopy(observation)
    trace = {"condition": condition, "semantic": semantic, "replacement": replacement, "cameras": {}}
    for camera_id, observation_key, _, _ in CAMERAS:
        image = np.asarray(observation["pixels"][observation_key])
        mask = bundle[camera_id][semantic]
        replaced = replace_pixels(image, mask, replacement)
        output["pixels"][observation_key] = replaced
        changed = np.any(replaced != image, axis=-1)
        trace["cameras"][camera_id] = {
            "input_sha256": array_sha256(image), "output_sha256": array_sha256(replaced),
            "mask_sha256": array_sha256(mask), "mask_pixels": int(mask.sum()),
            "changed_pixels": int(changed.sum()),
            "changed_outside_mask": int(np.sum(changed & ~mask)),
        }
    trace["trace_sha256"] = canonical_json_sha256(trace)
    return output, trace
