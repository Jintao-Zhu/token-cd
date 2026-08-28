from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ARMS = ("Strong", "P1_plus", "P1_minus", "P2_plus", "P2_minus", "P3_plus", "P3_minus", "P4_plus", "P4_minus")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    gate = json.loads((artifact / "snapshot_gate.json").read_text())
    snapshots = sorted((artifact / "snapshots").glob("*.pt"))
    if gate["decision"] != "LOCAL_SUCCESS_SNAPSHOT_GATE_PASS" or len(snapshots) != 100 or not (artifact / "pca_basis.npz").exists():
        raise RuntimeError("U1 preparation gate failed")
    rows = []
    for ordinal, path in enumerate(snapshots):
        meta = torch.load(path, weights_only=False, map_location="cpu")["metadata"]
        for seed in range(5):
            unit_id = f"{meta['snapshot_id']}__seed{seed}"
            for arm in ARMS:
                rows.append({"episode_id": f"{unit_id}__{arm}", "unit_id": unit_id, "snapshot_id": meta["snapshot_id"], "task_id": meta["task_id"], "init_state_id": meta["init_state_id"], "progress": meta["target_progress"], "continuation_seed": seed, "noise_seed": 960_000_000 + ordinal * 100 + seed, "arm": arm})
    target = artifact / "u1_manifest.jsonl"
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (artifact / "u1_manifest.sha256").write_text(hashlib.sha256(target.read_bytes()).hexdigest() + "  u1_manifest.jsonl\n")
    (artifact / "status.json").write_text(json.dumps({"status": "U1_READY", "causal_units": 500, "episodes_planned": 4500}, indent=2) + "\n")
    print(len(rows))


if __name__ == "__main__":
    main()
