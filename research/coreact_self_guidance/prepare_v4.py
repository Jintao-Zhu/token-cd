from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
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
    parser.add_argument("--alpha-grid", default="0.3,0.5,0.7")
    parser.add_argument("--amendment-reason", default=None)
    args = parser.parse_args()
    workspace, source, artifact = args.workspace.resolve(), args.source_artifact.resolve(), args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(artifact)
    if json.loads((source / "decision.json").read_text()).get("integrity_pass") is not True:
        raise RuntimeError("source artifact must have passed integrity")
    for directory in ("episodes", "logs", "status"):
        (artifact / directory).mkdir(parents=True)
    for name in ("requested_protocol.yaml", "phase0_state_manifest.jsonl", "task_manifest.json"):
        shutil.copy2(source / name, artifact / name)
    source_protocol = yaml.safe_load((source / "protocol.lock.yaml").read_text())
    code_paths = [
        workspace / "research/coreact_closed_loop/guidance.py",
        workspace / "research/coreact_closed_loop/runtime.py",
        workspace / "research/coreact_self_guidance/sampler.py",
        workspace / "research/coreact_self_guidance/relative_sampler.py",
        workspace / "research/coreact_self_guidance/prepare.py",
        workspace / "research/coreact_self_guidance/prepare_v4.py",
        workspace / "research/coreact_self_guidance/calibrate_relative.py",
        workspace / "research/coreact_self_guidance/integrity.py",
        workspace / "research/coreact_self_guidance/run.py",
        workspace / "research/coreact_self_guidance/analyze.py",
    ]
    protocol = {
        "experiment_name": "coreact_self_guidance_negative_branch_v4",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "mechanism_development_not_confirmation",
        "confirmation_claim_allowed": False,
        "parameters_tuned": False,
        "source_artifact": str(source),
        "source_protocol_sha256": sha256(source / "protocol.lock.yaml"),
        "outcomes_used_for_selection": False,
        "amendment_reason": args.amendment_reason,
        "tasks": source_protocol["tasks"],
        "phase0_states": 20,
        "phase0_tasks": [2, 9],
        "alpha_grid": [float(x) for x in args.alpha_grid.split(",")],
        "locked_alpha": "TBD_from_trajectory_calibration",
        "negative_branch": {
            "construction": "relative_lag_self_branch",
            "progress_time": "s=1-model_tau",
            "negative_progress": "s_neg=alpha*s",
            "state_interpolation": "linear interpolation of deterministic native reference states",
            "boundary": "step_zero_has_equal_clean_and_negative_velocity; no skipped flow step",
        },
        "checkpoint": source_protocol["checkpoint"],
        "shared": source_protocol["shared"],
        "arms": source_protocol["arms"],
        "w_values": source_protocol["w_values"],
        "total_episodes": 600,
        "trajectory_calibration": {
            "target": "E1 B_toward_top8 applied post-clip velocity delta",
            "metrics": ["mean", "rms", "median", "integrated_l2", "active_step_fraction", "clip_fraction"],
            "selection_metric": "median post-clip action-space velocity delta",
            "minimum_active_step_fraction": 0.8,
            "ratio_gate": [0.7, 1.4],
            "outcome_blind": True,
            "phase0_selection_seed_rule": "action_noise_seed + 17",
        },
        "statistics": source_protocol["statistics"],
        "analysis_after_600_of_600_only": True,
        "parameters_must_not_change_after_outcomes": True,
        "hashes": {
            "checkpoint_config": source_protocol["hashes"]["checkpoint_config"],
            "checkpoint_weights": source_protocol["hashes"]["checkpoint_weights"],
            "calibration_mean": source_protocol["hashes"]["calibration_mean"],
            "code": {str(path.relative_to(workspace)): sha256(path) for path in code_paths},
        },
    }
    (artifact / "protocol.phase0.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    probe = subprocess.run(["git", "-C", str(workspace / "lerobot"), "rev-parse", "HEAD"], capture_output=True, text=True)
    (artifact / "environment.json").write_text(json.dumps({"created_at": protocol["created_at"], "hostname": platform.node(), "lerobot_git_commit": probe.stdout.strip() or None, "workspace_git_status": "unavailable: ownership/safe-directory restriction", "source_artifact": str(source)}, indent=2) + "\n")
    print(json.dumps({"artifact": str(artifact), "phase0_states": 20, "alpha_grid": protocol["alpha_grid"]}, indent=2))


if __name__ == "__main__":
    main()
