from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from .common import file_sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    output = artifact / "environment.lock.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source = workspace / "artifacts/ar_token_counterfactual_qualification_v1_20260808_231044"
    source_environment = json.loads((source / "environment.json").read_text())
    code_paths = sorted((workspace / "research/ar_token_closed_loop").glob("*.py"))
    code_paths.extend([
        workspace / "research/ar_token_counterfactual/intervention.py",
        workspace / "research/ar_token_counterfactual/libero_runtime.py",
        workspace / "research/ar_token_counterfactual/hf_loader.py",
    ])
    payload = {
        "locked_at": datetime.now().astimezone().isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip(),
        "checkpoint_revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
        "checkpoint_config_sha256": source_environment["checkpoint_config_sha256"],
        "checkpoint_index_sha256": source_environment["checkpoint_index_sha256"],
        "checkpoint_shards": source_environment["checkpoint_shards"],
        "calibration_mean_file_sha256": file_sha256(source / "position_conditioned_visual_mean.pt"),
        "source_environment_sha256": file_sha256(source / "environment.json"),
        "code_sha256": {str(path.relative_to(workspace)): file_sha256(path) for path in code_paths},
        "workspace_git": "unavailable_at_workspace_root",
    }
    if payload["calibration_mean_file_sha256"] != "6e9ae0f39c59b4e93c05955d67c335f21981c8cded1d68e12969888df95203b5":
        raise RuntimeError("Calibration mean hash mismatch")
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
