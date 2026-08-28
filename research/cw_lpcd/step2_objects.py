#!/usr/bin/env python3
"""CW-LPCD STEP 2: determine the object visual-token set per state.

Derives object tokens from the Pixel-PCD `_changed.png` mask (the region the
image-level PCD actually removed) by projecting it onto the 16x16 patch grid and
thresholding overlap >= 0.25. Cross-validates against the frozen stage_a
`object_token_ids_25` (derived from the raw SAM mask) as a sanity check.

Run: env/venv/bin/python -m research.cw_lpcd.step2_objects --artifact artifacts/cw_lpcd_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from research.cw_lpcd.core import PCD_ROOT, mask_patch_overlaps, token_ids_from_mask

THRESHOLD = 0.25


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, default=PCD_ROOT)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    pcd_root = args.pcd_root.resolve()
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    masks_dir = Path(
        "artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/masks"
    )
    states_lock = Path(
        "artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl"
    )
    states = [json.loads(line) for line in states_lock.read_text().splitlines() if line]

    manifest = []
    n_exact = 0
    jaccards = []
    n_object = []
    for ordinal, row in enumerate(states):
        state_id = row["state_id"]
        changed_path = pcd_root / row["changed_path"]
        changed = cv2.imread(str(changed_path), cv2.IMREAD_GRAYSCALE)
        if changed is None:
            raise RuntimeError(f"Cannot read changed mask {changed_path}")
        mask01 = (changed > 127).astype(np.float32)
        object_ids, overlaps = token_ids_from_mask(mask01, args.threshold)

        frozen_json = masks_dir / f"{state_id}.json"
        frozen_ids = json.loads(frozen_json.read_text())["object_token_ids_25"] if frozen_json.exists() else None
        jac = jaccard(object_ids, frozen_ids) if frozen_ids is not None else None
        if jac is not None:
            jaccards.append(jac)
        n_object.append(len(object_ids))
        exact = (object_ids == frozen_ids) if frozen_ids is not None else None
        if exact:
            n_exact += 1

        sha = hashlib.sha256(cv2.imread(str(changed_path), cv2.IMREAD_UNCHANGED).tobytes()).hexdigest()
        manifest.append({
            "state_id": state_id, "task": row["task"], "seed": row["seed"],
            "object_token_ids": object_ids, "n_object_tokens": len(object_ids),
            "frozen_object_token_ids_25": frozen_ids,
            "jaccard_vs_frozen": jac, "exact_match_vs_frozen": exact,
            "changed_mask_sha256": sha,
            "max_overlap": float(overlaps.max()) if len(overlaps) else 0.0,
        })
        if ordinal % 30 == 0:
            print(json.dumps({"done": ordinal, "total": len(states)}), flush=True)

    out_manifest = artifact / "object_token_manifest.jsonl"
    with out_manifest.open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "threshold": args.threshold,
        "n_states": len(manifest),
        "object_token_count": {"min": min(n_object), "max": max(n_object),
                               "mean": float(np.mean(n_object)), "median": float(np.median(n_object))},
        "cross_validation_vs_frozen_object_token_ids_25": {
            "n_exact_match": n_exact, "exact_match_rate": n_exact / len(manifest),
            "mean_jaccard": float(np.mean(jaccards)), "min_jaccard": float(np.min(jaccards)),
            "states_with_zero_jaccard": int(sum(1 for j in jaccards if j == 0.0)),
        },
        "manifest_path": str(out_manifest.relative_to(artifact)),
    }
    (artifact / "step2_objects_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
