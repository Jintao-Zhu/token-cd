"""Lock selected tasks, state manifests, calibration mean, and Phase-1 code hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    output = artifact / "phase1_protocol.lock.yaml"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    states = [json.loads(line) for line in (artifact / "phase1_state_manifest.jsonl").read_text().splitlines() if line]
    calibration = [json.loads(line) for line in (artifact / "calibration_manifest.jsonl").read_text().splitlines() if line]
    selected = json.loads((artifact / "selected_tasks.json").read_text())
    if len(states) != 150 or len({row["snapshot_id"] for row in states}) != 150:
        raise RuntimeError("Phase-1 state manifest must contain 150 unique states")
    if len(calibration) != 150 or len({row["calibration_id"] for row in calibration}) != 150:
        raise RuntimeError("Calibration manifest must contain 150 unique states")
    if {row["task_id"] for row in states} & {row["task_id"] for row in calibration}:
        raise RuntimeError("Phase-1 and calibration task sets must be disjoint")
    code = [
        "research/ar_token_counterfactual/hf_loader.py",
        "research/ar_token_counterfactual/libero_runtime.py",
        "research/ar_token_counterfactual/intervention.py",
        "research/ar_token_counterfactual/run_phase1_integrity.py",
        "research/ar_token_counterfactual/run_phase1.py",
        "research/ar_token_counterfactual/analyze_phase1.py",
    ]
    protocol = {
        "locked_at": datetime.now().astimezone().isoformat(),
        "selected_tasks": selected,
        "phase1_states": 150,
        "states_per_task": 50,
        "phase_source": "trajectory_quartile_proxy",
        "calibration_states": 150,
        "calibration_task_disjoint": True,
        "visual_tokens_per_state": 16,
        "token_categories": {"attention_top": 4, "attention_middle": 4, "attention_bottom": 4, "deterministic_random": 4},
        "attention_aggregation": "mean_heads_action_positions_late_half_llm_layers",
        "replacement": "position_conditioned_post_projector_visual_mean",
        "primary": "teacher_forced_js_with_identical_clean_action_prefix",
        "secondary": "free_running_cascade",
        "random_seed": 20260808,
        "manifests": {
            "phase1_state_manifest_sha256": sha256(artifact / "phase1_state_manifest.jsonl"),
            "calibration_manifest_sha256": sha256(artifact / "calibration_manifest.jsonl"),
            "position_mean_file_sha256": sha256(artifact / "position_conditioned_visual_mean.pt"),
        },
        "code_sha256": {path: sha256(workspace / path) for path in code},
    }
    output.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
