#!/usr/bin/env python3
"""Lock a new-state confirmation of persistent pre-encoder image interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from datetime import datetime
from pathlib import Path

import cv2
import torch
import yaml
from libero.libero import benchmark

from research.coreact_closed_loop.prepare_pilot import sha256_file
from research.coreact_closed_loop.runtime import CHECKPOINT_REVISION, checkpoint_path


SUITES = ("libero_spatial", "libero_object")
TASK_ID = 7
INIT_IDS = tuple(range(35, 40))
CONDITIONS = {
    "vanilla": {"base_condition": "vanilla", "intervention_replans": 0},
    "target_inpaint_r2": {"base_condition": "target_inpaint", "intervention_replans": 2},
    "target_inpaint_r3": {"base_condition": "target_inpaint", "intervention_replans": 3},
    "target_blur_r3": {"base_condition": "target_blur", "intervention_replans": 3},
    "background_match_target_inpaint_r3": {"base_condition": "background_match_target_inpaint", "intervention_replans": 3},
    "target_black_r3": {"base_condition": "target_black", "intervention_replans": 3},
    "goal_inpaint_r3": {"base_condition": "goal_inpaint", "intervention_replans": 3},
    "background_match_goal_inpaint_r3": {"base_condition": "background_match_goal_inpaint", "intervention_replans": 3},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = workspace / "artifacts" / (
        "coreact_image_persistent_effect_v1_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    artifact.mkdir(parents=True, exist_ok=False)
    tasks = []
    for suite in SUITES:
        task = benchmark.get_benchmark_dict()[suite]().get_task(TASK_ID)
        tasks.append({"suite": suite, "task_id": TASK_ID, "task_name": task.name, "language": task.language, "bddl_file": task.bddl_file})
    protocol = {
        "experiment_name": "coreact_image_persistent_effect_v1",
        "stage": "new_state_confirmation_after_first_chunk_image_gate_failed",
        "created_before_clean_extraction": True,
        "prior_artifact": "artifacts/coreact_image_effect_existence_v1_20260807_125945",
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION, "frozen_eval": True},
        "states": {"suites": list(SUITES), "task_id": TASK_ID, "init_state_ids": list(INIT_IDS), "clean_horizon": 280, "overlap_with_prior_image_gate": 0},
        "conditions": CONDITIONS,
        "image_mask_semantics": "same qualified two-view target/goal masks and exact joint-area task-irrelevant background controls as amendment 004",
        "rollout": {"control_steps": 60, "actions_per_replan": 10, "horizons": [10, 30, 60], "same_exact_state_and_noise": True, "mask_recomputed_from_current_aligned_segmentation_at_each_intervened_replan": True},
        "gate": {
            "primary": "target_inpaint_r3 at H=60",
            "pass": "pooled median D>0.05, each suite median D>0, at least half states D>0.05, at least 4 component-varying states, and target_inpaint_r3 median D exceeds matched-background r3 median D",
            "r2": "dose-duration sensitivity only",
            "blur": "non-OOD sensitivity", "black": "OOD only",
            "on_fail": "suspect cross-view redundancy, short-term recovery, or progress endpoint; do not compare attribution",
        },
    }
    code_paths = (
        "research/coreact_region/image_intervention.py", "research/coreact_region/extract_critical_states.py",
        "research/coreact_region/audit_persistent_image_candidates.py", "research/coreact_region/run_persistent_image_effect.py",
        "research/coreact_region/analyze_persistent_image_effect.py",
    )
    protocol["code_hashes_at_creation"] = {
        path: hashlib.sha256((workspace / path).read_bytes()).hexdigest()
        for path in code_paths if (workspace / path).exists()
    }
    checkpoint = checkpoint_path(workspace)
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "opencv": cv2.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "network_required": False, "uv_available": False,
    }
    (artifact / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (artifact / "task_manifest.json").write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
