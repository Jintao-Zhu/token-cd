from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from research.coreact_trained_weak.runtime import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    checkpoint = Path(json.loads((artifact / "protocol.lock.json").read_text())["strong_checkpoint"])
    _, _, preprocessor, _ = load_policy(checkpoint)
    normalizer = preprocessor.steps[-1]
    source_manifest = workspace / "artifacts/coreact_slg_self_weak_low_noise_v1_20260816_234129/confirmation_manifold/state_manifest.jsonl"
    source_rows = [json.loads(line) for line in source_manifest.read_text().splitlines()]
    chunks, sources = [], []
    for task_id in range(10):
        demo_path = Path(next(row["demo_path"] for row in source_rows if row["task_id"] == task_id))
        if str(demo_path).startswith("/root/code/"):
            demo_path = workspace / Path(str(demo_path).removeprefix("/root/code/"))
        task_count = 0
        with h5py.File(demo_path, "r") as handle:
            for demo_id in sorted(handle["data"], key=lambda value: int(value.split("_")[-1])):
                actions = np.asarray(handle["data"][demo_id]["actions"], dtype=np.float32)
                if len(actions) < 10:
                    continue
                raw = np.stack([actions[index:index + 10, :7] for index in range(len(actions) - 9)])
                tensor = torch.as_tensor(raw, device="cuda", dtype=torch.float32)
                normalized = normalizer._normalize_action(tensor, inverse=False).cpu().numpy()
                chunks.append(normalized.reshape(len(normalized), -1))
                task_count += len(normalized)
        sources.append({"task_id": task_id, "demo_path": str(demo_path), "samples": task_count})
    matrix = np.concatenate(chunks, axis=0).astype(np.float64)
    mean = matrix.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    components = vt[:4].astype(np.float32).reshape(4, 10, 7)
    explained = singular_values[:4] ** 2 / np.sum(singular_values ** 2)
    np.savez_compressed(artifact / "pca_basis.npz", mean=mean.astype(np.float32).reshape(10, 7), components=components)
    audit = {"samples": int(len(matrix)), "dimensions": 70, "axes": 4, "explained_variance_ratio": explained.tolist(), "sources": sources, "outcome_used": False}
    (artifact / "pca_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
