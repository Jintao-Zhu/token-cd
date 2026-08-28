from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .core import array_sha256, file_sha256, read_jsonl, token_ids_from_mask, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture first-observation SAM2 masks; never steps an environment.")
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Qualification-only prefix limit; formal runner omits it.")
    args = parser.parse_args()
    pcd_root, artifact = args.pcd_root.resolve(), args.artifact.resolve()
    source = pcd_root / "source/PCD"
    sys.path.insert(0, str(source))
    os.chdir(source)
    import simpler_env
    from contrast_utils.contrast_image_generator import mask_to_bbox, name_to_alias
    from contrast_utils.instruction_templates import get_objects_from_instruction
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "pretrained/sam2.1_hiera_large.pt")
    sam_predictor = SAM2ImagePredictor(sam_model)

    def wrapped_observation(env):
        observation = env.unwrapped.get_obs()
        wrappers, current = [], env
        while hasattr(current, "env"):
            if hasattr(current, "observation"):
                wrappers.append(current)
            current = current.env
        for wrapper in reversed(wrappers):
            observation = wrapper.observation(observation)
        return observation

    rows = read_jsonl(artifact / "states.lock.jsonl")
    selected_rows = rows[:args.limit] if args.limit is not None else rows
    output_rows = []
    for ordinal, row in enumerate(selected_rows, 1):
        output_path = artifact / "masks" / f"{row['state_id']}.npz"
        record_path = artifact / "masks" / f"{row['state_id']}.json"
        if record_path.exists():
            existing = json.loads(record_path.read_text())
            if "frozen_rgb_exact_match" in existing and existing.get("mask_npz_sha256") == file_sha256(output_path):
                output_rows.append(existing)
                continue
        env = simpler_env.make(row["task"])
        try:
            obs, _ = env.reset(seed=row["seed"])
            header = json.loads((pcd_root / row["header_path"]).read_text())
            historical_state = np.asarray(header["initial_state"])
            env.unwrapped.set_state(historical_state.copy())
            env.unwrapped._elapsed_steps = 0
            obs = wrapped_observation(env)
            restored_state = np.asarray(env.unwrapped.get_state())
            expected_state = historical_state.astype(restored_state.dtype)
            state_max_abs_error = float(np.max(np.abs(restored_state - expected_state)))
            if state_max_abs_error > 1e-6:
                raise RuntimeError(f"Historical simulator state value mismatch for {row['state_id']}: {state_max_abs_error}")
            rendered = np.asarray(get_image_from_maniskill2_obs_dict(env, obs))
            frozen_image = cv2.cvtColor(cv2.imread(str(pcd_root / row["clean_path"])), cv2.COLOR_BGR2RGB)
            if array_sha256(frozen_image) != row["rgb_array_sha256"]:
                raise RuntimeError(f"Frozen clean PNG hash mismatch for {row['state_id']}")
            image = frozen_image
            observation_source = "frozen_clean_rgb_plus_exact_historical_state_geometry"
            rerender_rgb_exact_match = array_sha256(rendered) == row["rgb_array_sha256"]
            camera = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"
            segmentation = np.asarray(obs["image"][camera]["Segmentation"])[..., 1]
            actor_ids = {name_to_alias(actor.name): actor.id for actor in env.unwrapped.get_actors()}
            articulation_ids = {}
            for articulation in env.unwrapped.get_articulations():
                if articulation.name == "cabinet":
                    articulation_ids.update({name_to_alias(link.name): link.id for link in articulation.get_links()})
            name_to_id = {**actor_ids, **articulation_ids}
            objects = get_objects_from_instruction(row["instruction"])
            boxes = []
            for name in objects:
                gt_mask = segmentation == name_to_id[name] if name in name_to_id else np.zeros_like(segmentation, dtype=bool)
                box = mask_to_bbox(gt_mask)
                if box is not None:
                    boxes.append(box)
            if not boxes:
                raise RuntimeError(f"No official GT box prompt for {row['state_id']}")
            sam_predictor.set_image(image.copy())
            masks, _, _ = sam_predictor.predict(point_coords=None, point_labels=None, box=np.asarray(boxes), multimask_output=False)
            target = np.any(masks.squeeze(1) if masks.ndim == 4 else masks, axis=0)
        finally:
            env.close()
        if not target.any():
            raise RuntimeError(f"Empty SAM target mask for {row['state_id']}")
        ids25, overlaps = token_ids_from_mask(target, 0.25)
        ids50, _ = token_ids_from_mask(target, 0.50)
        if not ids25:
            raise RuntimeError(f"No object token at primary threshold for {row['state_id']}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, sam_target_mask=target, preprocessed_patch_overlap=overlaps)
        record = {"state_id": row["state_id"], "sam_target_mask_sha256": array_sha256(target),
                  "mask_npz_sha256": file_sha256(output_path), "object_token_ids_25": ids25,
                  "object_token_ids_50": ids50, "token_overlap_ratio": overlaps.tolist(),
                  "original_mask_shape": list(target.shape), "preprocessed_mask_grid_shape": [16, 16],
                  "observation_source": observation_source, "frozen_rgb_exact_match": True,
                  "historical_state_geometry_within_1e_6": True,
                  "restored_state_sha256": array_sha256(restored_state),
                  "historical_header_state_sha256": header["initial_state_sha256"],
                  "state_max_abs_error": state_max_abs_error,
                  "rerender_rgb_exact_match": rerender_rgb_exact_match}
        write_json(record_path, record)
        output_rows.append(record)
        print(json.dumps({"ordinal": ordinal, "state_id": row["state_id"], "tokens25": len(ids25)}), flush=True)
    if args.limit is None and len(output_rows) == 270:
        write_json(artifact / "mask_capture_report.json", {"status": "PASS", "states": 270,
                                                            "empty_masks": 0, "environment_steps": 0})


if __name__ == "__main__":
    main()
