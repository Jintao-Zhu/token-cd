#!/usr/bin/env python3
"""Create the locked pre-encoder image-space Effect-Existence artifact."""

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
INIT_IDS = tuple(range(30, 35))
CONDITIONS = tuple(
    ["vanilla"]
    + [
        f"{semantic}_{replacement}"
        for replacement in ("blur", "inpaint", "black")
        for semantic in ("target", "goal", "background_match_target", "background_match_goal")
    ]
)


def code_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = workspace / "artifacts" / (
        "coreact_image_effect_existence_v1_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    artifact.mkdir(parents=True, exist_ok=False)
    task_rows = []
    for suite_name in SUITES:
        task = benchmark.get_benchmark_dict()[suite_name]().get_task(TASK_ID)
        task_rows.append({
            "suite": suite_name, "task_id": TASK_ID, "task_name": task.name,
            "language": task.language, "bddl_file": task.bddl_file,
        })
    protocol = {
        "experiment_name": "coreact_image_effect_existence_v1",
        "stage": "development_image_space_positive_control",
        "created_before_clean_extraction": True,
        "question": "Does pre-encoder full-object image removal create measurable task-progress dynamic range?",
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION, "frozen_eval": True},
        "clean_state_extraction": {
            "suites": list(SUITES), "task_id": TASK_ID, "init_state_ids": list(INIT_IDS),
            "new_relative_to_prior_region_experiments": True, "horizon": 280,
            "pre_grasp": "state immediately before first clean grasp",
            "pre_place": "minimum object-goal distance while grasped before first clean predicate completion",
            "fallbacks": {"near_reach_m": 0.08, "near_place_m": 0.12},
            "minimum_before_intervention": "6 total, 2 per suite, and at least one true pre-place per suite",
        },
        "image_intervention": {
            "location": "raw uint8 256x256 RGB before official environment/policy preprocessing and vision encoder",
            "cameras": ["agentview_image", "robot0_eye_in_hand_image"],
            "instance_source": "aligned LIBERO simulator instance segmentation from the same raw observation",
            "complete_mask": "exact instance pixels plus 5-pixel dilation, excluding protected robot pixels",
            "primary_replacement": "opencv_telea_inpaint_radius_7",
            "non_ood_sensitivity": "gaussian_blur_kernel_31",
            "ood_only": "black_zero_rgb",
            "conditions": list(CONDITIONS),
            "background_control": "seeded four-connected segmentation-label-zero pixels, exact per-camera mask area",
        },
        "rollout": {
            "shared_exact_simulator_state": True, "shared_noise_by_replan": True,
            "intervention_replans": [0], "executed_actions_from_first_chunk": 10,
            "later_replans": "vanilla", "control_steps": 60, "reported_horizons": [10, 30, 60],
        },
        "gate": {
            "primary_condition": "target_inpaint", "primary_horizon": 60,
            "practical_D_threshold": 0.05,
            "pass": "target_inpaint pooled median D>0.05, each suite median D>0, at least half states D>0.05, and at least 4 component-varying states; target_blur median must be positive",
            "black_cannot_establish_gate_alone": True,
            "on_fail": "do not compare attribution; next test persistent intervention for 2-3 replans",
            "attribution_methods_prohibited": ["value_weighted_attention", "integrated_gradients", "activation_patching"],
            "cfg_or_lambda_tuning_prohibited": True,
        },
    }
    code_paths = (
        "research/coreact_region/image_intervention.py",
        "research/coreact_region/extract_critical_states.py",
        "research/coreact_region/audit_image_candidates.py",
        "research/coreact_region/run_image_effect_existence.py",
        "research/coreact_region/analyze_image_effect_existence.py",
    )
    protocol["code_hashes_at_creation"] = {
        path: code_hash(workspace / path) for path in code_paths if (workspace / path).exists()
    }
    checkpoint = checkpoint_path(workspace)
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "opencv": cv2.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "uv_available": False, "network_required": False,
    }
    (artifact / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (artifact / "task_manifest.json").write_text(json.dumps(task_rows, indent=2, sort_keys=True) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
