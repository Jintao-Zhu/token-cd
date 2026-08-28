#!/usr/bin/env python3
"""Create a preregistered, disjoint mask-only direction diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from libero.libero import benchmark

from lerobot.envs.libero import get_task_init_states
from research.coreact_closed_loop.prepare_pilot import sha256_array, sha256_file


SUITES = ("libero_spatial", "libero_object")
TASK_ID = 7
INIT_STATE_IDS = tuple(range(10))
CONDITIONS = ("vanilla", "top8_mask_only", "random8_mask_only", "bottom8_mask_only")
CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    artifact = workspace / "artifacts" / f"coreact_revision_mask_diagnostic_v1_{timestamp}"
    artifact.mkdir(parents=True, exist_ok=False)

    old_artifact = workspace / "artifacts/coreact_closed_loop_pilot_v1_20260807_074218"
    old_manifest = [json.loads(line) for line in (old_artifact / "episode_manifest.jsonl").read_text().splitlines()]
    old_tasks = {(row["suite"], row["task_id"]) for row in old_manifest}
    if any((suite, TASK_ID) in old_tasks for suite in SUITES):
        raise RuntimeError("revision-development task overlaps the prior pilot manifest")

    bench = benchmark.get_benchmark_dict()
    task_records = []
    rows = []
    for suite_index, suite in enumerate(SUITES):
        task_suite = bench[suite]()
        task = task_suite.get_task(TASK_ID)
        init_states = np.asarray(get_task_init_states(task_suite, TASK_ID))
        if init_states.shape[0] < len(INIT_STATE_IDS):
            raise RuntimeError(f"{suite} task {TASK_ID} has insufficient init states")
        task_records.append(
            {
                "suite": suite,
                "task_id": TASK_ID,
                "task_name": task.name,
                "canonical_language": task.language,
                "bddl_file": task.bddl_file,
                "init_state_count": int(init_states.shape[0]),
                "selected_init_states_sha256": sha256_array(init_states[list(INIT_STATE_IDS)]),
            }
        )
        for init_state_id in INIT_STATE_IDS:
            shared_seed = 52_000_000 + suite_index * 1_000_000 + TASK_ID * 10_000 + init_state_id * 100
            pair_id = f"direction_development__{suite}__task{TASK_ID:02d}__init{init_state_id:02d}"
            for condition in CONDITIONS:
                rows.append(
                    {
                        "episode_id": f"{pair_id}__{condition}",
                        "pair_id": pair_id,
                        "split": "direction_development",
                        "suite": suite,
                        "task_id": TASK_ID,
                        "init_state_id": init_state_id,
                        "condition": condition,
                        "language": task.language,
                        "reset_seed": shared_seed + 1,
                        "action_noise_seed": shared_seed + 2,
                        "selection_seed": shared_seed + 3,
                    }
                )

    means = workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    checkpoint = workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots" / CHECKPOINT_REVISION
    protocol = {
        "experiment_name": "coreact_revision_mask_diagnostic_v1",
        "stage": "revision_development_only_not_confirmation",
        "created_before_results": True,
        "question": "Does direct replacement of attention-top8 visual tokens reduce paired closed-loop success more than random8 or bottom8 replacement?",
        "prior_confirmation_reused_for_tuning": False,
        "disjointness": {
            "rule": "suite/task pair absent from every prior pilot split",
            "task_id_per_suite": TASK_ID,
            "verified_against": str(old_artifact / "episode_manifest.jsonl"),
        },
        "backbone": {
            "repo": "lerobot/smolvla_libero",
            "revision": CHECKPOINT_REVISION,
            "frozen_eval": True,
        },
        "conditions": list(CONDITIONS),
        "paired_states": len(SUITES) * len(INIT_STATE_IDS),
        "episodes": len(rows),
        "selection": {
            "ranking_proxy": "late_half_action_to_context_attention_at_flow_time_1_per_replan",
            "group_count": 8,
            "replacement": "position_conditioned_visual_mean_from_v8_calibration_only",
            "mask_execution": "integrate_masked_prefix_velocity_without_self_contrast_guidance",
        },
        "paired_controls": {
            "same_environment_init_state": True,
            "same_reset_seed": True,
            "same_per_replan_action_noise": True,
            "same_flow_solver": True,
            "same_preprocessing": True,
        },
        "integrity_gate_before_rollouts": [
            "frozen_eval",
            "native_determinism",
            "masked_determinism",
            "exactly_eight_legal_visual_tokens_changed",
            "protected_tokens_untouched",
            "same_noise",
            "all_outputs_finite",
        ],
        "analysis": {
            "primary": "paired_success_damage_vanilla_minus_top8_mask_only",
            "controls": ["random8_mask_only", "bottom8_mask_only"],
            "bootstrap": "suite_task_then_paired_state_2000_replicates",
            "paired_binary_test": "exact_McNemar",
            "interpretation": "development diagnostic only; two task clusters cannot support confirmation claims",
        },
        "stop_rule": "any mandatory integrity failure stops rollouts",
    }
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "uv_available": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "modality_means_sha256": sha256_file(means),
        "prior_decision_sha256": sha256_file(old_artifact / "decision.json"),
    }
    code_hashes = {
        name: file_hash(workspace / path)
        for name, path in {
            "masked_sampler": "research/coreact_revision/masked_sampler.py",
            "runner": "research/coreact_revision/run_mask_diagnostic.py",
            "integrity": "research/coreact_revision/run_mask_integrity.py",
            "analysis": "research/coreact_revision/analyze_mask_diagnostic.py",
            "attention_hook": "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
        }.items()
    }
    protocol["code_hashes"] = code_hashes
    (artifact / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (artifact / "task_manifest.json").write_text(json.dumps(task_records, indent=2, sort_keys=True) + "\n")
    with (artifact / "episode_manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
