#!/usr/bin/env python3
"""Create and lock the causal-label existence pilot before collecting outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from datetime import datetime
from pathlib import Path

import torch
import yaml
from libero.libero import benchmark

from research.coreact_causal_dataset.common import PHASES, derive_seed
from research.coreact_closed_loop.prepare_pilot import sha256_file
from research.coreact_closed_loop.runtime import CHECKPOINT_REVISION, checkpoint_path


TASK_IDS = (4, 7, 9)
INIT_IDS = tuple(range(15))
BASE_SEED = 812_000_000


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    artifact = workspace / "artifacts" / f"coreact_closed_loop_token_causal_dataset_v2_{stamp}"
    artifact.mkdir(parents=True, exist_ok=False)
    for name in ("clean_extraction", "candidate_audits", "episodes", "logs"):
        (artifact / name).mkdir()
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    tasks = [
        {"suite": "libero_spatial", "task_id": task_id, "task_name": suite.get_task(task_id).name,
         "language": suite.get_task(task_id).language, "bddl_file": suite.get_task(task_id).bddl_file}
        for task_id in TASK_IDS
    ]
    protocol = {
        "experiment_name": "closed_loop_token_causal_effect_dataset_qualification_v2",
        "stage": "development_existence_pilot_not_confirmation",
        "question": "Do stable signed token-level effects on closed-loop success exist?",
        "causal_estimand": "tau(s,G)=mean_k[success(mask G,s,k)-success(clean,s,k)]",
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION,
                     "frozen_eval": True},
        "suite": "libero_spatial",
        "tasks": tasks,
        "task_selection_basis": {
            "locked_before_new_mask_outcomes": True,
            "historical_clean_success": {4: 0.62, 7: 0.86, 9: 0.82},
            "reason": "task 4 has useful dynamic range; tasks 7/9 avoid Goal ceiling and task-8-specific signal",
            "limitation": "all are development tasks; task 7/9 exceed preferred 0.80 ceiling",
        },
        "snapshot_collection": {
            "init_state_ids": list(INIT_IDS), "snapshots_per_task": 15,
            "one_snapshot_per_clean_trajectory": True, "trajectory_max_steps": 280,
            "snapshot_only_at_replan_boundary": True, "target_phases": list(PHASES),
            "phase_assignment": "init_state_id modulo five, selected from clean trajectory only",
            "restore_stability_amendment": {
                "reason": "v1 exposed predicate/grasp flips at contact-threshold snapshots",
                "pre_grasp": "two replan boundaries before first observed grasp",
                "pre_place": "nearest grasped boundary with object-goal distance >=0.03 m",
                "required_boolean_parity": ["grasped", "predicate"],
                "geometry_tolerance_m": 0.001,
                "mask_outcomes_used": False,
            },
            "preserved_metadata": ["exact simulator state", "phase", "geometry", "gripper state",
                                   "recent 10 model actions", "normalized progress"],
        },
        "proposal": {
            "attention_role": "proposal_only_not_causal_attribution",
            "score": "late_half_action_to_context_attention_tau_1_native_prefix",
            "eligible": "valid post-connector visual tokens only",
            "groups_per_state": 8, "group_granularity": "one post-connector visual token",
            "strata": {"top": 3, "middle": 2, "low": 2, "random": 1},
            "replacement": "v8 camera-position-conditioned visual mean",
        },
        "rollouts": {
            "matched_noise_seeds_k": 3, "clean_rollouts_per_state": 3,
            "masked_rollouts_per_group": 3, "expected_total": 1215,
            "mask_schedule": "first replan after snapshot only; all later replans native vanilla",
            "flow_steps": 10, "chunk_size": 50, "executed_prefix": 10,
            "max_control_steps_after_snapshot": 280,
            "paired_identity": ["simulator snapshot", "noise seed by replan", "solver", "preprocessing",
                                "language", "future environment conditions"],
        },
        "labels": {"strong_anchor": "tau < -0.4", "strong_nuisance": "tau > 0.4",
                   "neutral": "abs(tau) <= 0.4"},
        "go_gate": {
            "meaningful_effect_fraction": ">=0.15 (target 0.15-0.20)",
            "both_signs": True, "seed_stability": "reported from paired replicate differences",
            "attention_analysis": "Spearman attention versus abs(tau), proposal quality only",
            "predictor_training_in_this_stage": False,
        },
        "no_go": "effects overwhelmingly neutral or dominated by seed-specific discordance",
        "seed_derivation": "base + task*1e6 + init*1e4 + replicate*100 + purpose",
        "seed_base": BASE_SEED,
        "analysis_embargo": "no aggregate causal outcomes until all manifest rows complete or explicitly failed",
    }
    code = [
        "research/coreact_causal_dataset/common.py", "research/coreact_causal_dataset/prepare.py",
        "research/coreact_causal_dataset/extract_snapshots.py", "research/coreact_causal_dataset/audit_candidates.py",
        "research/coreact_causal_dataset/run_rollouts.py", "research/coreact_causal_dataset/analyze.py",
        "research/coreact_region/segmented_runtime.py", "research/coreact_region/fixed_mask_sampler.py",
    ]
    protocol["code_hashes"] = {name: file_hash(workspace / name) for name in code if (workspace / name).exists()}
    checkpoint = checkpoint_path(workspace)
    means = workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "calibration_mean_sha256": sha256_file(means), "workspace_git_status": "not_a_git_repository",
    }
    rows = []
    for task in tasks:
        for init_id in INIT_IDS:
            phase = PHASES[init_id % len(PHASES)]
            rows.append({
                **task, "snapshot_id": f"spatial_t{task['task_id']:02d}_i{init_id:02d}_{phase}",
                "init_state_id": init_id, "target_phase": phase,
                "reset_seed": derive_seed(BASE_SEED, task["task_id"], init_id, 0, 1),
                "extraction_noise_seed": derive_seed(BASE_SEED, task["task_id"], init_id, 0, 2),
                "proposal_seed": derive_seed(BASE_SEED, task["task_id"], init_id, 0, 3),
                "rollout_seeds": [derive_seed(BASE_SEED, task["task_id"], init_id, k, 10) for k in range(3)],
            })
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    with (artifact / "snapshot_plan.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
