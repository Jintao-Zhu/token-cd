#!/usr/bin/env python3
"""Audit development artifacts and freeze the confirmation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from research.coreact_closed_loop.audit_qualification import read_jsonl, video_frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    expected = {
        row["episode_id"]: row
        for row in read_jsonl(artifact / "episode_manifest.jsonl")
        if row["split"] == "development"
    }
    if any(
        (artifact / "episodes" / row["episode_id"] / "episode.json").exists()
        for row in read_jsonl(artifact / "episode_manifest.jsonl")
        if row["split"] == "confirmation"
    ):
        raise RuntimeError("confirmation result exists before protocol lock")
    failures = []
    successes = Counter()
    pair_success = {}
    records = []
    for episode_id, spec in sorted(expected.items()):
        episode_dir = artifact / "episodes" / episode_id
        paths = [episode_dir / name for name in ("episode.json", "steps.jsonl", "rollout.mp4")]
        if not all(path.exists() for path in paths):
            failures.append({"episode_id": episode_id, "reason": "missing artifact"})
            continue
        record = json.loads(paths[0].read_text())
        steps = read_jsonl(paths[1])
        if len(steps) != record["control_steps"]:
            failures.append({"episode_id": episode_id, "reason": "step count mismatch"})
        if video_frames(paths[2]) is None:
            failures.append({"episode_id": episode_id, "reason": "video decode failure"})
        if record["nonfinite_action_count"] or not record["all_guidance_outputs_finite"]:
            failures.append({"episode_id": episode_id, "reason": "nonfinite output"})
        if record["condition"] == "coreact_top8":
            for trace in record["replan_traces"]:
                if len(trace["selected_indices"]) != 8:
                    failures.append({"episode_id": episode_id, "reason": "selection count"})
                if any(step["virtual_guidance_norm"] != 0 for step in trace["step_traces"]):
                    failures.append({"episode_id": episode_id, "reason": "virtual guidance"})
        elif record["replan_traces"]:
            failures.append({"episode_id": episode_id, "reason": "vanilla has guidance trace"})
        successes[record["condition"]] += int(record["success"])
        pair_success[(record["pair_id"], record["condition"])] = bool(record["success"])
        records.append(record)
    pair_ids = sorted({row["pair_id"] for row in records})
    paired = [
        {
            "pair_id": pair_id,
            "vanilla": pair_success.get((pair_id, "vanilla")),
            "coreact_top8": pair_success.get((pair_id, "coreact_top8")),
        }
        for pair_id in pair_ids
    ]
    base_only = sum(row["vanilla"] and not row["coreact_top8"] for row in paired)
    guided_only = sum(row["coreact_top8"] and not row["vanilla"] for row in paired)
    gate = {
        "gate": "development_execution_integrity",
        "pass": len(records) == 40 and len(paired) == 20 and not failures,
        "expected_episodes": 40,
        "complete_episodes": len(records),
        "expected_pairs": 20,
        "complete_pairs": len(paired),
        "failures": failures,
        "diagnostic_successes_not_used_as_confirmation_gate": dict(successes),
        "diagnostic_base_only_successes": base_only,
        "diagnostic_guided_only_successes": guided_only,
        "method_changes_after_development": [],
        "confirmation_results_created_or_viewed": False,
    }
    (artifact / "development_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    if not gate["pass"]:
        print(json.dumps(gate, indent=2, sort_keys=True))
        raise SystemExit(2)

    protocol = yaml.safe_load((artifact / "protocol.yaml").read_text())
    protocol.update(
        {
            "lock_stage": "qualification_and_development_integrity_passed_confirmation_unopened",
            "development_diagnostic_disclosed_before_confirmation": {
                "vanilla_successes": successes["vanilla"],
                "coreact_top8_successes": successes["coreact_top8"],
                "base_only": base_only,
                "guided_only": guided_only,
                "parameters_changed": False,
            },
            "confirmation_runner": {
                "append_only_episode_directories": True,
                "resume_skips_existing_complete_episode_json": True,
                "sharding": "manifest_order_index_modulo_shard_count",
            },
            "code_hashes": {
                name: sha256(workspace / path)
                for name, path in {
                    "guidance": "research/coreact_closed_loop/guidance.py",
                    "runtime": "research/coreact_closed_loop/runtime.py",
                    "runner": "research/coreact_closed_loop/run_pilot.py",
                    "analyzer": "research/coreact_closed_loop/analyze_pilot.py",
                    "qualification_auditor": "research/coreact_closed_loop/audit_qualification.py",
                    "development_finalizer": "research/coreact_closed_loop/finalize_development.py",
                    "attention_hook": "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
                }.items()
            },
            "statistics_seed": 8_675_309,
            "confirmation_results_viewed": False,
        }
    )
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
