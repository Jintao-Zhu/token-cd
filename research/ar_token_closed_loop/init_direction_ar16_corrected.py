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
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    source = args.candidate_artifact.resolve()
    out = args.artifact.resolve()
    if out.exists():
        raise FileExistsError(out)

    for name in ("snapshots", "groups", "episodes", "logs"):
        (out / name).mkdir(parents=True)
    plans = read_jsonl(source / "snapshot_plan.jsonl")
    if len(plans) != 50:
        raise RuntimeError(f"Expected 50 locked snapshots, found {len(plans)}")
    for plan in plans:
        snapshot_id = plan["snapshot_id"]
        shutil.copytree(source / "snapshots" / snapshot_id, out / "snapshots" / snapshot_id)
        shutil.copy(source / "groups" / f"{snapshot_id}.json", out / "groups" / f"{snapshot_id}.json")
    shutil.copy(source / "snapshot_plan.jsonl", out / "snapshot_plan.jsonl")

    protocol = {
        "experiment": "ar_token_every_replan_direction16_corrected_v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "candidate_source_artifact": str(source),
        "candidate_source_usage": "pre-rollout snapshots and locked top-16 groups only; no episode outcomes reused",
        "checkpoint_revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
        "snapshots": 50,
        "episodes_per_condition": 50,
        "episodes": 200,
        "conditions": ["vanilla", "top16_mask_only", "away", "toward"],
        "group_size": 16,
        "selection": "same locked mask_top16_effect group for all three intervention conditions",
        "replacement": "position_conditioned_visual_mean",
        "mask_frequency": "every replan through full remaining horizon",
        "guidance_scale": 0.5,
        "away_formula": "clean + 0.5 * (clean - masked)",
        "toward_formula": "clean - 0.5 * (clean - masked)",
        "bootstrap": {"clusters": ["task", "snapshot"], "replicates": 2000, "seed": 20260809},
        "development_replication": True,
        "supersedes_invalid_artifact": str(source),
        "invalid_artifact_reason": "manifest conditions did not match the locked direction protocol",
    }
    (out / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    write_json(
        out / "source_provenance.json",
        {
            "source_artifact": str(source),
            "source_protocol_sha256": file_sha256(source / "protocol.lock.yaml"),
            "source_snapshot_plan_sha256": file_sha256(source / "snapshot_plan.jsonl"),
            "candidate_files": {
                plan["snapshot_id"]: file_sha256(source / "groups" / f"{plan['snapshot_id']}.json")
                for plan in plans
            },
            "selected_snapshots": len(plans),
        },
    )
    print(json.dumps({"artifact": str(out), "snapshots": len(plans)}))


if __name__ == "__main__":
    main()
