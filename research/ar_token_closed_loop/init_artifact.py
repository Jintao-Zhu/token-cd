from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from .common import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite {artifact}")
    for child in ("logs", "snapshots", "candidates", "episodes"):
        (artifact / child).mkdir(parents=True, exist_ok=True)
    source = workspace / "artifacts/ar_token_counterfactual_qualification_v1_20260808_231044"
    protocol = {
        "experiment": "ar_token_closed_loop_causal_magnitude_calibration_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_phase1_artifact": str(source),
        "source_phase1_decision": "AR_TOKEN_GO",
        "checkpoint": {"repo": "openvla/openvla-7b-finetuned-libero-spatial", "revision": "962318cec55ac10993ff0f5f43eda9a270b4c873"},
        "suite": "libero_spatial",
        "task_ids": [2, 9, 3],
        "heldout_init_indices": list(range(13, 38)),
        "states": 75,
        "state_phase_target": "cycle trajectory fractions [0.125, 0.375, 0.625, 0.875]",
        "conditions": ["vanilla", "mask_max_effect", "mask_near_zero", "mask_random"],
        "rollouts": 300,
        "candidate_universe": "all 256 post-projector visual tokens",
        "offline_effect": "mean teacher-forced JS over 7 action positions with identical clean action prefix",
        "max_effect": "largest offline effect; token-index tie break",
        "near_zero": "smallest offline effect; token-index tie break",
        "random": "deterministic uniform token excluding max and near-zero",
        "random_seed": 20260809,
        "replacement": "locked position-conditioned post-projector visual mean from source Phase 1",
        "intervention_duration": "first control action only, then vanilla",
        "maximum_total_policy_steps": 220,
        "stochasticity": "greedy action decode; simulator seed 0; no sampling noise",
        "primary": "absolute paired binary success discordance versus vanilla",
        "secondary": ["absolute object-goal endpoint distance change", "absolute eef-object endpoint distance change", "grasp discordance"],
        "bootstrap": {"clusters": ["task", "snapshot"], "replicates": 2000, "seed": 20260809},
        "go_criteria": {
            "success_magnitude": "max minus both near-zero and random paired discordance, both bootstrap CI lower > 0",
            "object_goal_magnitude": "max minus both near-zero and random absolute endpoint effect, both bootstrap CI lower > 0",
            "continuous_predictiveness": "offline E versus absolute object-goal endpoint effect Spearman >= 0.30 with cluster bootstrap CI lower > 0",
            "minimum_passed": 2,
        },
        "decisions": ["AR_CLOSED_LOOP_MAGNITUDE_GO", "AR_CLOSED_LOOP_MAGNITUDE_NO_GO"],
        "prohibited": ["VCAD", "toward_or_away_guidance", "sign_classifier", "changing_k_or_replacement_after_outcomes"],
    }
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    fractions = (0.125, 0.375, 0.625, 0.875)
    with (artifact / "snapshot_plan.jsonl").open("x", encoding="utf-8") as handle:
        for task_id in protocol["task_ids"]:
            for ordinal, init_index in enumerate(protocol["heldout_init_indices"]):
                row = {
                    "snapshot_id": f"task{task_id:02d}_init{init_index:02d}",
                    "task_id": task_id,
                    "init_state_index": init_index,
                    "target_fraction": fractions[ordinal % len(fractions)],
                    "env_seed": 0,
                    "split": "closed_loop_calibration_heldout",
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_files = ["decision.json", "phase1_protocol.lock.yaml", "position_conditioned_visual_mean.pt"]
    provenance = {name: file_sha256(source / name) for name in source_files}
    (artifact / "source_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
