#!/usr/bin/env python3
"""Prepare a locked 50-pair three-condition replication for one LIBERO-Spatial task."""

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


CONDITIONS = ("vanilla", "coreact_top8", "top8_mask_only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    suite_name = "libero_spatial"
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    if len(init_states) != 50:
        raise RuntimeError(f"expected 50 official init states, got {len(init_states)}")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    artifact = workspace / "artifacts" / f"coreact_spatial_task{args.task_id}_top8_replication_v1_{timestamp}"
    artifact.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoint_path(workspace)
    means = workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    code_paths = {
        "guidance": "research/coreact_closed_loop/guidance.py",
        "masked_sampler": "research/coreact_revision/masked_sampler.py",
        "runtime": "research/coreact_closed_loop/runtime.py",
        "runner": "research/coreact_task4_replication/run.py",
        "integrity": "research/coreact_task4_replication/integrity.py",
        "prepare": "research/coreact_task4_replication/prepare_task.py",
        "analyzer": "research/coreact_task4_replication/analyze_task.py",
        "attention_hook": "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    }
    protocol = {
        "experiment_name": f"coreact_spatial_task{args.task_id}_top8_replication_v1",
        "stage": "task_level_development_replication_not_confirmation",
        "task_selected_without_reading_prior_task_outcomes": True,
        "benchmark": "LIBERO",
        "suite": suite_name,
        "task_id": args.task_id,
        "task": task.language,
        "official_init_state_ids": list(range(50)),
        "paired_states": 50,
        "episodes": 150,
        "conditions": list(CONDITIONS),
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION, "frozen_eval": True},
        "shared": {
            "same_simulator_init_state": True,
            "same_reset_seed": True,
            "same_gaussian_noise_per_replan": True,
            "same_preprocessing": True,
            "render_warmup_resets_without_action": 1,
            "official_same_seed_resets": 1,
            "flow_steps": 10,
            "chunk_size": 50,
            "executed_actions_per_chunk": 10,
            "maximum_control_steps": 280,
            "rerank_current_two_camera_observation_every_replan": True,
        },
        "selection": {
            "native_unintervened_prefix": True,
            "flow_time": 1.0,
            "score": "late_half_action_to_context_attention",
            "eligible_visual_tokens": 128,
            "top_k": 8,
            "protected": ["language", "state", "special", "padding"],
            "replacement": "v8_position_conditioned_camera_visual_mean",
        },
        "coreact_top8": {
            "formula": "clean + 0.5 * trust_region_clip(clean - masked)",
            "guidance_scale": 0.5,
            "trust_region_kappa": 0.25,
            "real_action_dimensions": 7,
        },
        "top8_mask_only": {"masked_velocity_only_all_flow_steps": True, "lambda": None},
        "vanilla": {"native_prefix_native_sampler": True},
        "statistics": {"paired_bootstrap_replicates": 2000, "bootstrap_seed": 77_007_007, "paired_binary_test": "exact_McNemar"},
        "prohibited": ["parameter_tuning", "random_condition", "bottom_condition", "reading_partial_success_summary"],
    }
    protocol["hashes"] = {
        "checkpoint_config": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights": sha256_file(checkpoint / "model.safetensors"),
        "calibration_mean": sha256_file(means),
        "code": {name: hashlib.sha256((workspace / path).read_bytes()).hexdigest() for name, path in code_paths.items()},
    }
    rows = []
    for init_id in range(50):
        shared = 307_000_000 + args.task_id * 100_000 + init_id * 100
        pair_id = f"development_replication__{suite_name}__task{args.task_id:02d}__init{init_id:02d}"
        for condition in CONDITIONS:
            rows.append({
                "episode_id": f"{pair_id}__{condition}", "pair_id": pair_id,
                "suite": suite_name, "task_id": args.task_id, "task_name": task.name,
                "language": task.language, "init_state_id": init_id, "condition": condition,
                "reset_seed": shared + 1, "action_noise_seed": shared + 2, "selection_seed": shared + 3,
            })
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "network_required": False,
    }
    dumped = yaml.safe_dump(protocol, sort_keys=False)
    (artifact / "protocol.yaml").write_text(dumped); (artifact / "protocol.lock.yaml").write_text(dumped)
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (artifact / "task_manifest.json").write_text(json.dumps({"suite": suite_name, "task_id": args.task_id, "task_name": task.name, "language": task.language, "bddl_file": task.bddl_file, "official_init_state_count": 50}, indent=2, sort_keys=True) + "\n")
    with (artifact / "episode_manifest.jsonl").open("x") as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
