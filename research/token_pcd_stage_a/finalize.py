from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from .core import TASKS, file_sha256, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    artifact, workspace = args.artifact.resolve(), args.workspace.resolve()
    rows = read_jsonl(artifact / "results.jsonl")
    decision = json.loads((artifact / "stage_a_decision.json").read_text())
    required = ["protocol.lock.json", "states.lock.jsonl", "calibration.lock.jsonl", "mask_capture_report.json",
                "model_preflight.json", "position_conditioned_visual_mean.pt", "position_mean.json",
                "integrity_sentinel.json", "results.jsonl", "stage_a_decision.json"]
    if len(rows) != 270 or len({row["state_id"] for row in rows}) != 270:
        raise RuntimeError("Final result cardinality failure")
    if any(row.get("logits_storage_dtype") != "float32" for row in rows):
        raise RuntimeError("Final logits are not float32")
    task_counts = {task: sum(row["task"] == task for row in rows) for task in TASKS}
    if any(count != 30 for count in task_counts.values()):
        raise RuntimeError(f"Task cardinality failure: {task_counts}")
    for row in rows:
        path = artifact / row["logits_path"]
        if file_sha256(path) != row["logits_sha256"]:
            raise RuntimeError(f"Logits hash failure: {row['state_id']}")
        with np.load(path) as payload:
            if set(payload.files) != {"clean", "pixel_cf", "object_token_cf", "random_token_cf", "object_token_cf_50", "random_token_cf_50"}:
                raise RuntimeError("Logits branch set failure")
            if any(payload[name].shape != (7, 32064) or payload[name].dtype != np.float32 for name in payload.files):
                raise RuntimeError(f"Logits shape/dtype failure: {row['state_id']}")
    code_files = sorted((workspace / "research/token_pcd_stage_a").glob("*.py"))
    manifest = {
        "status": "PASS", "scope": "STAGE_A_OFFLINE_ONLY", "rollout_episodes": 0,
        "decision": decision["status"], "states": len(rows), "task_counts": task_counts,
        "all_integrity_checks_pass": all(all(value for value in row["checks"].values() if isinstance(value, bool)) for row in rows),
        "artifact_sha256": {name: file_sha256(artifact / name) for name in required},
        "code_sha256": {str(path.relative_to(workspace)): file_sha256(path) for path in code_files},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__, "numpy": np.__version__,
                    "cuda": torch.version.cuda},
    }
    write_json(artifact / "final_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
