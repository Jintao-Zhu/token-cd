#!/usr/bin/env python3
"""Create the immutable preregistration and episode manifests for the closed-loop pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from libero.libero import benchmark

from lerobot.envs.libero import get_task_init_states


CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
CHECKPOINT_CONFIG_SHA256 = "5c9f3ba9f5f37ea7024c9501b0b20b1941f232989c2167853cb46d9071a70dd7"
CHECKPOINT_WEIGHTS_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
SUITES = ("libero_spatial", "libero_object")
QUALIFICATION_TASKS = (8, 9)
DEVELOPMENT_TASKS = (0, 1)
CONFIRMATION_TASKS = (2, 3, 4, 5, 6)
CONDITIONS = ("vanilla", "coreact_top8", "random8", "bottom8")
CODE_VERSION = "coreact_closed_loop_pilot_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def episode_row(
    *, suite: str, task_id: int, init_state_id: int, split: str, condition: str, language: str
) -> dict:
    suite_index = SUITES.index(suite)
    shared_seed = 41_000_000 + suite_index * 1_000_000 + task_id * 10_000 + init_state_id * 100
    return {
        "episode_id": f"{split}__{suite}__task{task_id:02d}__init{init_state_id:02d}__{condition}",
        "pair_id": f"{split}__{suite}__task{task_id:02d}__init{init_state_id:02d}",
        "split": split,
        "suite": suite,
        "task_id": task_id,
        "init_state_id": init_state_id,
        "condition": condition,
        "language": language,
        "reset_seed": shared_seed + 1,
        "action_noise_seed": shared_seed + 2,
        "selection_seed": shared_seed + 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    artifact = workspace / "artifacts" / f"coreact_closed_loop_pilot_v1_{timestamp}"
    artifact.mkdir(parents=True, exist_ok=False)

    task_records = []
    rows = []
    bench = benchmark.get_benchmark_dict()
    split_specs = (
        ("qualification", QUALIFICATION_TASKS, range(0, 10), ("vanilla",)),
        ("development", DEVELOPMENT_TASKS, range(0, 5), ("vanilla", "coreact_top8")),
        ("confirmation", CONFIRMATION_TASKS, range(10, 30), CONDITIONS),
    )
    for suite in SUITES:
        task_suite = bench[suite]()
        for task_id in sorted(set(QUALIFICATION_TASKS + DEVELOPMENT_TASKS + CONFIRMATION_TASKS)):
            task = task_suite.get_task(task_id)
            init_states = np.asarray(get_task_init_states(task_suite, task_id))
            if init_states.shape[0] < 30:
                raise RuntimeError(f"{suite} task {task_id} has only {init_states.shape[0]} init states")
            task_records.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task.name,
                    "canonical_language": task.language,
                    "problem_folder": task.problem_folder,
                    "bddl_file": task.bddl_file,
                    "init_state_count": int(init_states.shape[0]),
                    "selected_init_states_sha256": sha256_array(init_states[:30]),
                }
            )
            for split, task_ids, init_ids, conditions in split_specs:
                if task_id not in task_ids:
                    continue
                for init_state_id in init_ids:
                    for condition in conditions:
                        rows.append(
                            episode_row(
                                suite=suite,
                                task_id=task_id,
                                init_state_id=init_state_id,
                                split=split,
                                condition=condition,
                                language=task.language,
                            )
                        )

    protocol = {
        "experiment_name": CODE_VERSION,
        "created_before_any_new_closed_loop_result": True,
        "primary_question": "Does locked visual-token self-contrast improve paired canonical LIBERO success?",
        "backbone": {
            "repo": "lerobot/smolvla_libero",
            "revision": CHECKPOINT_REVISION,
            "config_sha256": CHECKPOINT_CONFIG_SHA256,
            "weights_sha256": CHECKPOINT_WEIGHTS_SHA256,
            "frozen_eval": True,
        },
        "evaluation_data": {
            "demonstration_snapshot_repo": "HuggingFaceVLA/libero",
            "demonstration_snapshot_revision": DATASET_REVISION,
            "claimed_as_training_revision": False,
            "environment_suites": list(SUITES),
            "canonical_language_from_libero_task_manifest": True,
            "real_cameras": ["agentview_image", "robot0_eye_in_hand_image"],
        },
        "splits": {
            "qualification": {"task_ids_per_suite": list(QUALIFICATION_TASKS), "init_state_ids": list(range(10))},
            "development": {"task_ids_per_suite": list(DEVELOPMENT_TASKS), "init_state_ids": list(range(5))},
            "confirmation": {
                "task_ids_per_suite": list(CONFIRMATION_TASKS),
                "init_state_ids": list(range(10, 30)),
            },
            "task_and_init_state_disjoint": True,
        },
        "qualification_gate": {
            "episodes": 40,
            "minimum_aggregate_success_rate": 0.10,
            "minimum_successes_per_suite": 1,
            "stop_status": "INCONCLUSIVE_BACKBONE",
        },
        "locked_method": {
            "modalities": ["visual"],
            "ranking_proxy": "late_half_action_to_context_attention_at_flow_time_1_per_replan",
            "group": "single_post_connector_visual_token",
            "selected_group_count": 8,
            "replacement": "position_conditioned_visual_mean_from_v8_calibration_only",
            "guidance_scale": 0.5,
            "trust_region_kappa": 0.25,
            "flow_steps": 10,
            "negative_branch_frequency": "every_flow_step",
            "executed_action_prefix": 10,
            "protected": ["state", "language", "special_tokens", "padding", "virtual_action_dimensions"],
            "uncertainty_gate": "not_in_MVP",
        },
        "conditions": list(CONDITIONS),
        "paired_controls": {
            "same_environment_init_state": True,
            "same_reset_seed": True,
            "same_per_replan_action_noise": True,
            "same_flow_solver": True,
            "same_observation_preprocessing": True,
        },
        "primary_endpoint": "paired_task_success_coreact_top8_minus_vanilla",
        "secondary_endpoints": [
            "coreact_top8_minus_random8_success",
            "coreact_top8_minus_bottom8_success",
            "episode_steps",
            "action_total_variation",
            "chunk_boundary_discontinuity",
            "latency_and_peak_memory",
            "guidance_norm_and_fallback_rate",
        ],
        "statistics": {
            "bootstrap_clusters": ["suite_task", "pair_id"],
            "bootstrap_replicates": 2000,
            "paired_binary_test": "McNemar_exact",
            "multiple_comparison": "Holm",
        },
        "confirmation_decision": {
            "proceed": [
                "all_mandatory_integrity_gates_pass",
                "missingness_le_5_percent",
                "coreact_minus_vanilla_point_estimate_gt_0",
                "coreact_minus_vanilla_cluster_bootstrap_ci_lower_gt_0",
                "coreact_minus_random_cluster_bootstrap_ci_lower_gt_0",
                "both_suites_nonnegative",
                "no_increase_in_constraint_or_nonfinite_action_failures",
            ],
            "statuses": [
                "PROCEED_TO_CONFIRMATORY_CLOSED_LOOP",
                "REVISE_GUIDANCE_AND_REPEAT_PILOT",
                "STOP_CLOSED_LOOP_GUIDANCE_HYPOTHESIS",
                "INCONCLUSIVE_BACKBONE",
                "INCONCLUSIVE_DATA_OR_RESOURCES",
                "FAILED_INTEGRITY",
            ],
        },
        "confirmation_results_must_not_be_read_before_protocol_lock": True,
    }
    (artifact / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    with (artifact / "episode_manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (artifact / "task_manifest.json").write_text(json.dumps(task_records, indent=2, sort_keys=True) + "\n")

    checkpoint = workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots" / CHECKPOINT_REVISION
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_path": str(checkpoint),
        "checkpoint_config_sha256_observed": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights_sha256_observed": sha256_file(checkpoint / "model.safetensors"),
        "lerobot_commit": git_value(workspace / "lerobot", "rev-parse", "HEAD"),
        "lerobot_status_at_start": git_value(workspace / "lerobot", "status", "--short"),
        "libero_commit": git_value(workspace / "LIBERO", "rev-parse", "HEAD"),
        "offline_v8_artifact": "artifacts/coreact_exploration_v8_20260807_002951",
        "offline_modality_means_sha256": sha256_file(
            workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
        ),
    }
    if environment["checkpoint_config_sha256_observed"] != CHECKPOINT_CONFIG_SHA256:
        raise RuntimeError("checkpoint config hash mismatch")
    if environment["checkpoint_weights_sha256_observed"] != CHECKPOINT_WEIGHTS_SHA256:
        raise RuntimeError("checkpoint weight hash mismatch")
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
