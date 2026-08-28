#!/usr/bin/env python3
"""Audit every preregistered qualification episode before guided development."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_frames(path: Path) -> int | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    expected = {
        row["episode_id"]: row
        for row in read_jsonl(artifact / "episode_manifest.jsonl")
        if row["split"] == "qualification"
    }
    failures = []
    successes = Counter()
    records = []
    checksums = []
    for episode_id, spec in sorted(expected.items()):
        episode_dir = artifact / "episodes" / episode_id
        record_path = episode_dir / "episode.json"
        video_path = episode_dir / "rollout.mp4"
        steps_path = episode_dir / "steps.jsonl"
        if not all(path.exists() for path in (record_path, video_path, steps_path)):
            failures.append({"episode_id": episode_id, "reason": "missing artifact"})
            continue
        record = json.loads(record_path.read_text())
        steps = read_jsonl(steps_path)
        decoded_frames = video_frames(video_path)
        identity_keys = (
            "episode_id",
            "pair_id",
            "suite",
            "task_id",
            "init_state_id",
            "condition",
            "language",
            "reset_seed",
            "action_noise_seed",
        )
        mismatched = [key for key in identity_keys if record.get(key) != spec.get(key)]
        if mismatched:
            failures.append({"episode_id": episode_id, "reason": "identity mismatch", "keys": mismatched})
        if len(steps) != record["control_steps"]:
            failures.append({"episode_id": episode_id, "reason": "step-log count mismatch"})
        if decoded_frames is None or decoded_frames < record["control_steps"]:
            failures.append({"episode_id": episode_id, "reason": "video decode/count failure"})
        if record["nonfinite_action_count"] != 0:
            failures.append({"episode_id": episode_id, "reason": "nonfinite action"})
        if not all(
            all(abs(float(value)) < float("inf") for value in step[key])
            for step in steps
            for key in ("model_action", "physical_action", "legal_action")
        ):
            failures.append({"episode_id": episode_id, "reason": "nonfinite step log"})
        successes[record["suite"]] += int(record["success"])
        records.append(record)
        checksums.append(
            {
                "episode_id": episode_id,
                "episode_json_sha256": sha256(record_path),
                "steps_jsonl_sha256": sha256(steps_path),
                "video_sha256": sha256(video_path),
                "decoded_video_frames": decoded_frames,
            }
        )

    aggregate_successes = sum(successes.values())
    gate_pass = (
        len(records) == 40
        and not failures
        and aggregate_successes / 40 >= 0.10
        and all(successes[suite] >= 1 for suite in ("libero_spatial", "libero_object"))
    )
    gate = {
        "gate": "canonical_two_suite_backbone_qualification",
        "pass": gate_pass,
        "expected_episodes": 40,
        "complete_episodes": len(records),
        "successes_by_suite": dict(sorted(successes.items())),
        "success_rate_by_suite": {
            suite: successes[suite] / 20 for suite in ("libero_spatial", "libero_object")
        },
        "aggregate_successes": aggregate_successes,
        "aggregate_success_rate": aggregate_successes / 40,
        "failure_count": len(failures),
        "failures": failures,
        "all_videos_fully_decoded": not any(
            failure["reason"] == "video decode/count failure" for failure in failures
        ),
        "interpretation": "Eligibility sanity check only; qualification tasks are excluded from development and confirmation.",
    }
    (artifact / "qualification_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    )
    (artifact / "qualification_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    raise SystemExit(0 if gate_pass else 2)


if __name__ == "__main__":
    main()
