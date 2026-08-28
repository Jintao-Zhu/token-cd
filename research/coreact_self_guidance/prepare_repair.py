from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    source = args.source_artifact.resolve()
    artifact = args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(artifact)

    failure = json.loads((source / "analysis_integrity_failure.json").read_text())
    expected_failure = [{"episode_id": "task04__init35", "reason": "state/preprocessing"}]
    if failure.get("failures") != expected_failure or failure.get("pairs") != 99:
        raise RuntimeError("source artifact does not have the expected isolated integrity failure")
    manifest = (source / "episode_manifest.jsonl").read_text()
    if len(manifest.splitlines()) != 600:
        raise RuntimeError("source manifest is not the locked 600-episode manifest")

    for directory in ("episodes", "logs", "status"):
        (artifact / directory).mkdir(parents=True)
    for name in (
        "requested_protocol.yaml",
        "protocol.phase0.yaml",
        "phase0_calibration.json",
        "phase0_state_manifest.jsonl",
        "task_manifest.json",
        "environment.json",
    ):
        shutil.copy2(source / name, artifact / name)
    (artifact / "episode_manifest.jsonl").write_text(manifest, encoding="utf-8")

    protocol = yaml.safe_load((source / "protocol.lock.yaml").read_text())
    protocol["experiment_name"] = args.experiment_name
    protocol["created_at"] = datetime.now().astimezone().isoformat()
    protocol["repair"] = {
        "source_artifact": str(source),
        "source_protocol_sha256": sha256(source / "protocol.lock.yaml"),
        "source_failure_sha256": sha256(source / "analysis_integrity_failure.json"),
        "reason": "one paired prepared-input identity mismatch in task04 init35 W15",
        "outcomes_used_for_parameter_selection": False,
        "old_episode_results_reused": False,
        "scheduler": "pair-sharded; all six arms for a pair run serially on one worker",
        "pre_action_gate": "exact simulator-state and prepared-input SHA-256 identity",
        "initial_observation_rule": "cache the first arm's real reset observation and reuse it for the first replan of all six paired arms",
    }
    code_paths = [
        workspace / "research/coreact_closed_loop/guidance.py",
        workspace / "research/coreact_closed_loop/runtime.py",
        workspace / "research/coreact_self_guidance/sampler.py",
        workspace / "research/coreact_self_guidance/prepare.py",
        workspace / "research/coreact_self_guidance/prepare_repair.py",
        workspace / "research/coreact_self_guidance/calibrate.py",
        workspace / "research/coreact_self_guidance/integrity.py",
        workspace / "research/coreact_self_guidance/run.py",
        workspace / "research/coreact_self_guidance/analyze.py",
    ]
    protocol["hashes"]["code"] = {
        str(path.relative_to(workspace)): sha256(path) for path in code_paths
    }
    (artifact / "protocol.lock.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps({"artifact": str(artifact), "episodes": 600, "locked_delta": protocol["locked_delta"]}, indent=2))


if __name__ == "__main__":
    main()
