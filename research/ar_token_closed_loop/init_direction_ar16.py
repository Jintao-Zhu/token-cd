from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from .common import file_sha256, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, out = args.workspace.resolve(), args.artifact.resolve()
    source = workspace / "artifacts/ar_token_every_replan_direction_v1_20260809_100702"
    if out.exists():
        raise FileExistsError(out)
    for name in ("snapshots", "groups", "episodes", "logs"):
        (out / name).mkdir(parents=True)
    plans = read_jsonl(source / "snapshot_plan.jsonl")[:50]
    protocol = {
        "experiment": "ar_token_every_replan_direction_calibration16_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_k8_artifact": str(source),
        "checkpoint_revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
        "snapshot_selection": "first 50 rows of locked k8 plan, fixed before rollouts",
        "snapshots": 50,
        "episodes_per_condition": 50,
        "episodes": 200,
        "conditions": ["vanilla", "top16_mask_only", "away", "toward"],
        "group_size": 16,
        "selection": "locked top8 effect group from k8 group artifact",
        "replacement": "position_conditioned_visual_mean",
        "mask_frequency": "every replan through full remaining horizon",
        "guidance_scale": 0.5,
        "away_formula": "clean + 0.5 * (clean - masked)",
        "toward_formula": "clean - 0.5 * (clean - masked)",
        "bootstrap": {"clusters": ["task", "snapshot"], "replicates": 2000, "seed": 20260809},
        "development_replication": True,
    }
    (out / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    with (out / "snapshot_plan.jsonl").open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps(plan, sort_keys=True) + "\n")
            shutil.copytree(source / "snapshots" / plan["snapshot_id"], out / "snapshots" / plan["snapshot_id"])
    write_json(out / "source_provenance.json", {"source_artifact": str(source), "source_protocol_sha256": file_sha256(source / "protocol.lock.yaml"), "source_summary_sha256": file_sha256(source / "summary.json"), "selected_snapshots": len(plans)})
    print(out)


if __name__ == "__main__":
    main()
