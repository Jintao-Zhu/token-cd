from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from .core import TASKS, file_sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    pcd_root, artifact = args.pcd_root.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=False)
    state_rows, calibration_rows = [], []
    for task in TASKS:
        for seed in range(30):
            stem = pcd_root / "method_box/mini" / task / "pcd_box" / f"episode_{seed:03d}"
            paths = {name: Path(str(stem) + suffix) for name, suffix in {
                "clean": "_clean.png", "pixel": "_contrast.png", "changed": "_changed.png",
                "header": "_header.json", "audit": "_visual_audit.json"}.items()}
            if not all(path.is_file() for path in paths.values()):
                raise FileNotFoundError(f"Incomplete frozen state: {task} seed {seed}")
            header, audit = json.loads(paths["header"].read_text()), json.loads(paths["audit"].read_text())
            clean = cv2.cvtColor(cv2.imread(str(paths["clean"])), cv2.COLOR_BGR2RGB)
            from .core import array_sha256
            if array_sha256(clean) != header["initial_rgb_sha256"] or audit["image_sha256"] != header["initial_rgb_sha256"]:
                raise RuntimeError(f"Frozen RGB hash mismatch: {task} seed {seed}")
            state_rows.append({
                "state_id": f"{task}__seed{seed:03d}", "task": task, "seed": seed,
                "instruction": header["instruction"],
                **{f"{name}_path": str(path.relative_to(pcd_root)) for name, path in paths.items()},
                **{f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
                "rgb_array_sha256": header["initial_rgb_sha256"],
                "gt_audit_mask_sha256": audit["intended_target_mask_sha256"],
            })
        for seed in range(30, 50):
            stem = pcd_root / "directional" / task / "pcd_box" / f"episode_{seed:03d}"
            image, header_path = Path(str(stem) + "_clean.png"), Path(str(stem) + "_header.json")
            if not image.is_file() or not header_path.is_file():
                raise FileNotFoundError(f"Missing independent calibration state: {task} seed {seed}")
            header = json.loads(header_path.read_text())
            calibration_rows.append({"calibration_id": f"{task}__seed{seed:03d}", "task": task, "seed": seed,
                                     "instruction": header["instruction"], "image_path": str(image.relative_to(pcd_root)),
                                     "image_sha256": file_sha256(image)})
    for name, rows in (("states.lock.jsonl", state_rows), ("calibration.lock.jsonl", calibration_rows)):
        (artifact / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    protocol = {
        "protocol_id": "TOKEN-PCD-OPENVLA-SIMPLER-STAGE-A-R1", "scope": "OFFLINE_ONLY_NO_ROLLOUT",
        "checkpoint": "PCD pinned OpenVLA-7B", "tasks": list(TASKS), "state_seeds": list(range(30)),
        "calibration_seeds": list(range(30, 50)), "states": 270, "calibration_states": 180,
        "branches": ["clean", "pixel_cf", "object_token_cf", "random_token_cf"],
        "teacher_forcing": "same clean seven-token prefix", "alpha": 0.8,
        "primary_mask_overlap_threshold": 0.25, "secondary_threshold": 0.50,
        "random_rule": "same count, sampled without replacement from non-object tokens; sha256(state_id) seed",
        "primary_gate": {"median_object_pixel_cosine_min": 0.50, "median_object_minus_random_min": 0.20,
                         "paired_bootstrap_ci_lower_gt": 0.0, "tasks_object_gt_random_min": 7,
                         "median_effect_norm_ratio": [0.5, 1.5]},
        "no_go": "object approximately random or median object-pixel cosine below 0.2",
    }
    write_json(artifact / "protocol.lock.json", protocol)
    write_json(artifact / "prepare_report.json", {"status": "PASS", "states": len(state_rows),
                                                   "calibration_states": len(calibration_rows),
                                                   "protocol_sha256": file_sha256(artifact / "protocol.lock.json")})


if __name__ == "__main__":
    main()

