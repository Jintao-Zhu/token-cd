#!/usr/bin/env python3
"""Create the locked segmentation-grounded region mechanism experiment."""

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
from research.coreact_region.run_region_diagnostic import CONDITIONS


SUITES = ("libero_spatial", "libero_object")
TASK_ID = 7
INIT_IDS = tuple(range(10, 15))
CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"


def code_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    artifact = workspace / "artifacts" / f"coreact_region_mechanism_v1_{timestamp}"
    artifact.mkdir(parents=True, exist_ok=False)

    prior = workspace / "artifacts/coreact_revision_mask_diagnostic_v1_20260807_105706/episode_manifest.jsonl"
    prior_states = {
        (row["suite"], row["task_id"], row["init_state_id"])
        for row in (json.loads(line) for line in prior.read_text().splitlines())
    }
    rows, tasks = [], []
    bench = benchmark.get_benchmark_dict()
    for suite_index, suite_name in enumerate(SUITES):
        suite = bench[suite_name]()
        task = suite.get_task(TASK_ID)
        tasks.append({
            "suite": suite_name, "task_id": TASK_ID, "task_name": task.name,
            "language": task.language, "bddl_file": task.bddl_file,
        })
        for init_id in INIT_IDS:
            if (suite_name, TASK_ID, init_id) in prior_states:
                raise RuntimeError("region state overlaps prior mask diagnostic")
            shared = 63_000_000 + suite_index * 1_000_000 + TASK_ID * 10_000 + init_id * 100
            pair_id = f"region_development__{suite_name}__task{TASK_ID:02d}__init{init_id:02d}"
            for condition in CONDITIONS:
                rows.append({
                    "episode_id": f"{pair_id}__{condition}", "pair_id": pair_id,
                    "suite": suite_name, "task_id": TASK_ID, "init_state_id": init_id,
                    "condition": condition, "language": task.language,
                    "reset_seed": shared + 1, "action_noise_seed": shared + 2,
                    "selection_seed": shared + 3,
                })
    protocol = {
        "experiment_name": "coreact_region_mechanism_v1",
        "stage": "region_mechanism_development_not_confirmation",
        "created_before_results": True,
        "question": "Does task relevance from simulator segmentation identify the sign of high-attention visual token interventions?",
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION, "frozen_eval": True},
        "states": {"suites": list(SUITES), "task_id": TASK_ID, "init_state_ids": list(INIT_IDS), "paired_states": 10, "prior_state_overlap": 0},
        "conditions": list(CONDITIONS),
        "region_mapping": {
            "segmentation": "LIBERO/robosuite instance segmentation from the same raw observation as policy RGB",
            "policy_image": "256x256 resized without padding to 512x512",
            "vision_grid": "32x32 patches from patch_size_16",
            "connector": "no resampler; row-major 4x4 pixel shuffle to 8x8 post-connector tokens",
            "semantic_term": "source footprint, not strict local receptive field after global vision self-attention",
            "relevant": "task obj_of_interest coverage >= 0.05 and no protected pixels",
            "protected": "any Panda, gripper, or mount instance pixel",
            "background": "zero relevant and zero protected pixels; may contain task-unrelated objects",
            "ambiguous": "nonzero relevant coverage below 0.05",
        },
        "group_selection": {
            "groups": ["relevant_high", "background_high", "relevant_low", "background_random"],
            "matched_count": "minimum available across relevant/background capped at 4",
            "minimum_count": 2,
            "score": "late_half_action_to_context_attention_at_flow_time_1",
            "random_seeded": True,
        },
        "intervention": {
            "replacement": "position_conditioned_visual_mean_from_v8_calibration",
            "masked_replans": [0], "executed_actions_from_first_chunk": 10,
            "subsequent_replans": "vanilla", "total_control_steps": 30,
        },
        "paired_controls": {
            "same_initial_sim_state_hash": True, "same_noise_hash_for_all_three_replans": True,
            "same_language_state_preprocessing": True, "same_flow_solver": True,
        },
        "progress": {
            "reach": "(initial eef-object distance - minimum distance) / initial distance clipped to [-1,1]",
            "transport": "(initial object-goal distance - minimum distance) / initial distance clipped to [-1,1]",
            "grasp": "ever grasped within 30 steps", "predicate": "ever satisfied within 30 steps",
            "composite": "equal mean of reach, transport, grasp, predicate",
            "D": "clean progress minus mask progress; D>0 anchor-like, D<0 nuisance-like",
        },
        "decision": {
            "dtp": "background_high mean composite D < -0.05 and median D < 0 in both suites",
            "coreact": "relevant_high mean composite D > 0.05 and median D > 0 in both suites",
            "otherwise": "compare value-weighted attention, integrated gradients, or activation patching",
            "cfg_or_lambda_tuning_prohibited": True,
        },
        "statistics": {"bootstrap_clusters": ["suite_task", "paired_state"], "replicates": 2000, "development_only": True},
    }
    source_paths = {
        "region_mapping": "research/coreact_region/region_mapping.py",
        "segmented_runtime": "research/coreact_region/segmented_runtime.py",
        "fixed_mask_sampler": "research/coreact_region/fixed_mask_sampler.py",
        "qualification": "research/coreact_region/qualify_region_mapping.py",
        "runner": "research/coreact_region/run_region_diagnostic.py",
        "analysis": "research/coreact_region/analyze_region_diagnostic.py",
        "attention_hook": "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    }
    protocol["code_hashes"] = {name: code_hash(workspace / path) for name, path in source_paths.items()}
    checkpoint = workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots" / CHECKPOINT_REVISION
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "uv_available": False,
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "modality_means_sha256": sha256_file(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"),
    }
    (artifact / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (artifact / "task_manifest.json").write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n")
    with (artifact / "episode_manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
