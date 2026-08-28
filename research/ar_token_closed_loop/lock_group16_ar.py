from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import file_sha256, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    output = artifact / "rollout_manifest.lock.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    plans = read_jsonl(artifact / "snapshot_plan.jsonl")
    rows, candidate_hashes = [], {}
    for plan in plans:
        path = artifact / "groups" / f"{plan['snapshot_id']}.json"
        candidate = json.loads(path.read_text())
        selected = candidate["selected"]
        indices = [selected[name]["token_indices"] for name in ("mask_top16_effect", "mask_bottom16_effect", "mask_random16")]
        if any(len(indexes) != 16 for indexes in indices):
            raise RuntimeError(f"Group size failure: {plan['snapshot_id']}")
        flat_indices = [index for indexes in indices for index in indexes]
        if len({tuple(indexes) for indexes in indices}) != 3 or any(len(set(indexes)) != 16 for indexes in indices) or not candidate["all_finite"]:
            raise RuntimeError(f"Candidate qualification failed: {plan['snapshot_id']}")
        candidate_hashes[plan["snapshot_id"]] = file_sha256(path)
        for condition in ("vanilla", "mask_top16_effect", "mask_bottom16_effect", "mask_random16"):
            rows.append({
                "episode_id": f"{plan['snapshot_id']}__{condition}",
                **plan,
                "condition": condition,
                "visual_token_indices": [] if condition == "vanilla" else selected[condition]["token_indices"],
                "offline_effect": 0.0 if condition == "vanilla" else selected[condition]["mean_teacher_forced_js"],
                "candidate_file_sha256": candidate_hashes[plan["snapshot_id"]],
            })
    if len(rows) != 200 or len({row["episode_id"] for row in rows}) != 200:
        raise RuntimeError("Rollout manifest cardinality failure")
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(artifact / "candidate_lock.json", {"candidate_files": candidate_hashes, "rollout_manifest_sha256": file_sha256(output)})
    print(json.dumps({"snapshots": len(plans), "episodes": len(rows), "manifest_sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
