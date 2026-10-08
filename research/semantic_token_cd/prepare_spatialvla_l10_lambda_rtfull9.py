"""Assemble and audit the frozen RT-full L10-Matched references for lambda scan."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
SNAP = ART / "vanilla_recon_shr_canonical_0_299_v2" / "snapshots"
OVERRIDE = ART / "final_closure_snapshot_overrides_20261003" / "snapshots"
OUT = ART / "spatialvla_l10_lambda_rtfull9_100_299_v1" / "reference_l10_matched"

DRAWERS = {"google_robot_open_drawer", "google_robot_close_drawer"}
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
)


def array_sha(value) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(str(tuple(arr.shape)).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def snapshot_sha(snapshot) -> str:
    digest = hashlib.sha256()
    digest.update(array_sha(snapshot["sim_state"]).encode())
    digest.update(repr(snapshot["agent_state"]).encode())
    digest.update(repr(snapshot["rng_state"]).encode())
    digest.update(snapshot["instruction"].encode())
    return digest.hexdigest()


def source_for(task: str, seed: int) -> Path:
    if task in DRAWERS:
        if 100 <= seed <= 199:
            root = "spatialvla_l10_vs_l11_4task_n100"
        else:
            root = "spatialvla_l10_vanilla_expand9_to300_v1"
    else:
        if 100 <= seed <= 199:
            root = "spatialvla_l10_matched_rtfull7_n100_199"
        else:
            root = "spatialvla_l10_vanilla_rtfull7_to300_v1"
    return ART / root / "episodes" / task / "l10_matched" / f"episode_{seed:03d}_summary.json"


def snapshot_for(task: str, seed: int) -> Path:
    main = SNAP / task / f"seed_{seed:03d}.pkl"
    if not main.is_file():
        raise FileNotFoundError(main)
    ref = json.loads(source_for(task, seed).read_text())
    if snapshot_sha(pickle.loads(main.read_bytes())) == ref["canonical_snapshot_sha256"]:
        return main
    alt = OVERRIDE / task / f"seed_{seed:03d}.pkl"
    if alt.is_file() and snapshot_sha(pickle.loads(alt.read_bytes())) == ref["canonical_snapshot_sha256"]:
        return alt
    raise RuntimeError(f"No snapshot matches the locked L10 reference: {task} seed={seed}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "REFERENCE_MANIFEST.json"
    expected = {
        "protocol_id": "SPATIALVLA_L10_MATCHED_RTFULL_REFERENCE_9TASK_SEEDS100_299_V1",
        "renderer": "SAPIEN RT depth=8 spp=32 denoiser=true",
        "tasks": list(TASKS),
        "seeds": [100, 299],
        "reference_arm": "l10_matched lambda=0.5",
        "pairing": "canonical snapshot sha must match; runtime checks simulator state and records hardware-specific RGB match",
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != expected:
        raise RuntimeError(f"Existing reference manifest conflicts: {manifest_path}")

    checked = 0
    for task in TASKS:
        for seed in range(100, 300):
            source = source_for(task, seed)
            if not source.is_file():
                raise FileNotFoundError(source)
            row = json.loads(source.read_text())
            if row.get("arm") != "l10_matched" or row.get("seed") != seed or row.get("task") != task:
                raise RuntimeError(f"Wrong reference row: {source}")
            if row.get("renderer", {}).get("depth") != 8:
                raise RuntimeError(f"Reference is not RT-full depth 8: {source}")
            if row.get("attention_layer") != 10:
                raise RuntimeError(f"Reference is not L10: {source}")
            snap = snapshot_for(task, seed)
            if snapshot_sha(pickle.loads(snap.read_bytes())) != row["canonical_snapshot_sha256"]:
                raise RuntimeError(f"Snapshot mismatch after source selection: {task} seed={seed}")
            link = OUT / "episodes" / task / "l10_matched" / f"episode_{seed:03d}_summary.json"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != source.resolve():
                    raise RuntimeError(f"Reference symlink points elsewhere: {link}")
            elif link.exists():
                raise RuntimeError(f"Refusing to overwrite non-symlink reference: {link}")
            else:
                link.symlink_to(source.resolve())
            checked += 1

    if not manifest_path.exists():
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, manifest_path)
    print(json.dumps({"reference_rows": checked, "tasks": len(TASKS), "seeds_per_task": 200,
                      "renderer": "RT-full depth 8", "snapshot_hashes": "PASS",
                      "reference_root": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
