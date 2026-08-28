#!/usr/bin/env python3
"""Create a new locked Effect-Existence Gate artifact before any intervention."""

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


SUITES = ("libero_spatial", "libero_object")
TASK_ID = 7
INIT_IDS = tuple(range(15, 20))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    artifact = workspace / "artifacts" / f"coreact_effect_existence_v1_{timestamp}"
    artifact.mkdir(parents=True, exist_ok=False)
    bench = benchmark.get_benchmark_dict()
    sources = []
    for suite in SUITES:
        task = bench[suite]().get_task(TASK_ID)
        sources.append(
            {"suite": suite, "task_id": TASK_ID, "task_name": task.name,
             "language": task.language, "bddl_file": task.bddl_file}
        )
    protocol = {
        "experiment_name": "coreact_effect_existence_v1",
        "stage": "development_effect_existence_gate",
        "created_before_clean_extraction": True,
        "question": "Can a first-chunk visual intervention produce a measurable task-progress effect from clean-selected critical states?",
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION, "frozen_eval": True},
        "source_states": {"suites": list(SUITES), "task_id": TASK_ID, "init_state_ids": list(INIT_IDS), "clean_horizon": 60},
        "critical_state_selection": {
            "pre_grasp": "last simulator state immediately before first clean-trajectory grasp",
            "pre_place": "grasped, predicate-false clean state with minimum object-goal distance before later predicate completion",
            "near_reach_fallback": "minimum clean eef-object distance only when <=0.08 m",
            "near_place_fallback": "minimum clean object-goal distance while grasped only when <=0.12 m",
            "intervention_outcomes_used_for_selection": False,
        },
        "candidate_audit": {
            "target_group": "all unprotected post-connector source footprints with nonzero target-object pixels",
            "relevant_high_k": [4, 8, 16],
            "unsupported_k": "omit per snapshot; never fill with background",
            "negative_control": "seeded background-random with identical count and replacement",
        },
        "intervention": {
            "masked_replans": [0], "executed_actions_from_first_chunk": 10,
            "subsequent_replans": "vanilla", "one_rollout_horizon": 60,
            "reported_horizons": [10, 30, 60],
            "primary_replacement": "position_conditioned_visual_mean_from_v8_calibration",
            "strong_ood_sensitivity": "zero for full-target and matched background only",
        },
        "paired_controls": {"exact_sim_state": True, "same_noise_by_replan": True, "same_language_and_preprocessing": True},
        "gate": {
            "D": "clean composite progress minus intervention composite progress",
            "practical_threshold": 0.05,
            "primary_horizon": 60,
            "minimum_critical_states": "at least 2 per suite and 6 total",
            "pass": "at H=60, full-target position-mean pooled median D>0.05, each suite median D>0, at least half of states D>0.05, and at least 4 states differ on grasp/predicate or have |reach/transport D|>0.05",
            "failure_status": "EFFECT_EXISTENCE_NOT_ESTABLISHED",
            "attribution_comparison_before_pass": "prohibited",
            "cfg_or_lambda_tuning": "prohibited",
        },
    }
    code_paths = [
        "research/coreact_region/effect_existence.py",
        "research/coreact_region/extract_critical_states.py",
        "research/coreact_region/audit_effect_candidates.py",
        "research/coreact_region/run_effect_existence.py",
        "research/coreact_region/analyze_effect_existence.py",
        "research/coreact_region/fixed_mask_sampler.py",
    ]
    protocol["code_hashes_at_creation"] = {
        path: file_hash(workspace / path) for path in code_paths if (workspace / path).exists()
    }
    checkpoint = checkpoint_path(workspace)
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "uv_available": False,
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights_sha256": sha256_file(checkpoint / "model.safetensors"),
    }
    (artifact / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (artifact / "task_manifest.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
