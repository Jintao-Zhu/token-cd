from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import file_sha256, read_jsonl, write_json


CONDITIONS = ("vanilla", "top16_mask_only", "away", "toward")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    output = artifact / "rollout_manifest.lock.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    plans = read_jsonl(artifact / "snapshot_plan.jsonl")
    rows: list[dict] = []
    candidate_hashes: dict[str, str] = {}
    for plan in plans:
        path = artifact / "groups" / f"{plan['snapshot_id']}.json"
        candidate = json.loads(path.read_text())
        selected = candidate["selected"]["mask_top16_effect"]
        indices = selected["token_indices"]
        if len(indices) != 16 or len(set(indices)) != 16 or not candidate["all_finite"]:
            raise RuntimeError(f"Top-16 group qualification failed: {plan['snapshot_id']}")
        candidate_hashes[plan["snapshot_id"]] = file_sha256(path)
        for condition in CONDITIONS:
            rows.append(
                {
                    **plan,
                    "episode_id": f"{plan['snapshot_id']}__{condition}",
                    "condition": condition,
                    "visual_token_indices": [] if condition == "vanilla" else indices,
                    "offline_effect": 0.0 if condition == "vanilla" else selected["mean_teacher_forced_js"],
                    "candidate_file_sha256": candidate_hashes[plan["snapshot_id"]],
                }
            )

    if len(plans) != 50 or len(rows) != 200 or len({row["episode_id"] for row in rows}) != 200:
        raise RuntimeError("Rollout manifest cardinality failure")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise RuntimeError("Rollout manifest condition failure")
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(
        artifact / "candidate_lock.json",
        {"candidate_files": candidate_hashes, "rollout_manifest_sha256": file_sha256(output)},
    )
    print(json.dumps({"snapshots": len(plans), "episodes": len(rows), "manifest_sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
