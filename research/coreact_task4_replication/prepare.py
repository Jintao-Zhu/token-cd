#!/usr/bin/env python3
"""Create the locked task-4 top8 development replication artifact."""

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

from research.coreact_closed_loop.prepare_pilot import sha256_file
from research.coreact_closed_loop.runtime import CHECKPOINT_REVISION, checkpoint_path


SUITE = "libero_spatial"
TASK_ID = 4
INIT_IDS = tuple(range(50))
CONDITIONS = ("vanilla", "coreact_top8", "top8_mask_only")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, default=Path.cwd()); args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = workspace / "artifacts" / ("coreact_spatial_task4_top8_replication_v1_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"))
    artifact.mkdir(parents=True, exist_ok=False)
    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    expected_language = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
    if task.language != expected_language:
        raise RuntimeError(f"task language mismatch: {task.language}")
    checkpoint = checkpoint_path(workspace)
    means_path = workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    code_paths = {
        "guidance": "research/coreact_closed_loop/guidance.py",
        "masked_sampler": "research/coreact_revision/masked_sampler.py",
        "runtime": "research/coreact_closed_loop/runtime.py",
        "runner": "research/coreact_task4_replication/run.py",
        "integrity": "research/coreact_task4_replication/integrity.py",
        "analyzer": "research/coreact_task4_replication/analyze.py",
        "attention_hook": "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    }
    protocol = {
        "experiment_name": "coreact_spatial_task4_top8_replication_v1",
        "stage": "task_level_development_replication_not_independent_confirmation",
        "created_before_results": True,
        "old_task4_outcomes_read_for_parameter_selection": False,
        "benchmark": "LIBERO", "suite": SUITE, "task_id": TASK_ID,
        "task": task.language, "init_state_ids": list(INIT_IDS),
        "paired_states": 50, "episodes": 150, "conditions": list(CONDITIONS),
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION, "frozen_eval": True},
        "shared": {
            "same_simulator_init_state": True, "same_reset_seed": True,
            "same_gaussian_noise_seed_per_replan": True, "same_preprocessing": True,
            "flow_steps": 10, "chunk_size": 50, "executed_actions_per_chunk": 10,
            "maximum_control_steps": 280, "rerank_current_two_camera_observation_every_replan": True,
        },
        "selection": {
            "native_unintervened_prefix": True, "flow_time": 1.0,
            "score": "late_half_action_to_context_attention", "eligible_visual_tokens": 128,
            "top_k": 8, "protected": ["language", "state", "special", "padding"],
            "replacement": "v8_position_conditioned_camera_visual_mean",
        },
        "coreact_top8": {
            "formula": "clean + 0.5 * trust_region_clip(clean - masked)",
            "guidance_scale": 0.5, "trust_region_kappa": 0.25,
            "real_action_dimensions": 7, "clean_and_masked_velocity_each_flow_step": True,
        },
        "top8_mask_only": {"masked_velocity_only_all_flow_steps": True, "lambda": None},
        "vanilla": {"native_prefix_native_sampler": True},
        "statistics": {"paired_bootstrap_replicates": 2000, "bootstrap_seed": 44_004_004, "paired_binary_test": "exact_McNemar"},
        "prohibited": ["parameter_tuning", "random_condition", "bottom_condition", "reading_partial_success_summary"],
    }
    protocol["hashes"] = {
        "checkpoint_config": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights": sha256_file(checkpoint / "model.safetensors"),
        "calibration_mean": sha256_file(means_path),
        "code": {name: hashlib.sha256((workspace / path).read_bytes()).hexdigest() for name, path in code_paths.items() if (workspace / path).exists()},
    }
    rows = []
    for init_id in INIT_IDS:
        shared = 104_000_000 + TASK_ID * 100_000 + init_id * 100
        pair_id = f"development_replication__{SUITE}__task{TASK_ID:02d}__init{init_id:02d}"
        for condition in CONDITIONS:
            rows.append({
                "episode_id": f"{pair_id}__{condition}", "pair_id": pair_id,
                "suite": SUITE, "task_id": TASK_ID, "task_name": task.name,
                "language": task.language, "init_state_id": init_id, "condition": condition,
                "reset_seed": shared + 1, "action_noise_seed": shared + 2,
                "selection_seed": shared + 3,
            })
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "network_required": False,
    }
    (artifact / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False)); (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (artifact / "task_manifest.json").write_text(json.dumps({"suite": SUITE, "task_id": TASK_ID, "task_name": task.name, "language": task.language, "bddl_file": task.bddl_file}, indent=2, sort_keys=True) + "\n")
    with (artifact / "episode_manifest.jsonl").open("x") as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__": main()
