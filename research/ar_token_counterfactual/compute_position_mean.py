"""Compute position-conditioned post-projector visual means from calibration images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from .intervention import projector_intervention, tensor_sha256
from .libero_runtime import build_prompt, load_policy, set_determinism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    rows = [json.loads(line) for line in (artifact / "calibration_manifest.jsonl").read_text().splitlines() if line]
    if len(rows) != 150 or len({row["calibration_id"] for row in rows}) != 150:
        raise RuntimeError(f"Expected 150 unique calibration images, got {len(rows)}")
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)
    total = None
    with torch.inference_mode():
        for row in rows:
            image = Image.open(artifact / row["image"]).convert("RGB")
            inputs = processor(build_prompt(row["task_description"]), image).to(model.device, dtype=torch.bfloat16)
            with projector_intervention(model) as trace:
                model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"], use_cache=False, return_dict=True,
                )
            features = trace.before[0].double()
            total = features if total is None else total + features
    mean = (total / len(rows)).float()
    output = artifact / "position_conditioned_visual_mean.pt"
    torch.save({"mean": mean, "count": len(rows), "shape": list(mean.shape), "dtype": str(mean.dtype)}, output)
    file_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {"count": len(rows), "shape": list(mean.shape), "dtype": str(mean.dtype), "tensor_sha256": tensor_sha256(mean), "file_sha256": file_hash}
    (artifact / "position_mean.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
