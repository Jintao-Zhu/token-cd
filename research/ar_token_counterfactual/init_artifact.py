"""Create and lock an AR token qualification artifact before model outcomes are observed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
from datetime import datetime
from pathlib import Path

import torch
import tokenizers
import transformers
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact}")
    artifact.mkdir(parents=True)
    (artifact / "logs").mkdir()
    (artifact / "videos").mkdir()

    revision = "962318cec55ac10993ff0f5f43eda9a270b4c873"
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial" / revision
    expected_shards = {
        "model-00001-of-00004.safetensors": (4925122448, "094ba8d798d0e13152230897e893d4d12ca3fe11ddb218f27545b1e6e7536134"),
        "model-00002-of-00004.safetensors": (4947392496, "086780bb21b61eee9fb54acc0b938017c599e9d741bcdda18da23c4617fa36d7"),
        "model-00003-of-00004.safetensors": (4947417456, "d696e042cf49abd0b5846e73cbd4931d75a5467ae44857bf8e54a90ed6588308"),
        "model-00004-of-00004.safetensors": (262668432, "234f92229d4055749c17e904e4d3f1c9a9af3eb13f96f14eaaa4530c0c376473"),
    }
    shard_audit = {}
    for filename, (expected_size, expected_hash) in expected_shards.items():
        path = checkpoint / filename
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Checkpoint shard missing or wrong size: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checkpoint shard hash mismatch: {path}: {actual_hash}")
        shard_audit[filename] = {"bytes": expected_size, "sha256": actual_hash}
    protocol = {
        "experiment_name": "ar_token_counterfactual_qualification_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "scientific_question": "Whether post-connector single-token deletion in AR OpenVLA yields strong, stable, structured action-logit effects",
        "checkpoint": {"repo_id": "openvla/openvla-7b-finetuned-libero-spatial", "revision": revision},
        "benchmark": {"name": "LIBERO", "suite": "libero_spatial", "task_ids": list(range(10))},
        "baseline": {
            "episodes_per_task": 20,
            "init_state_indices": list(range(20)),
            "seed": 7,
            "settle_steps": 10,
            "max_policy_steps": 220,
            "center_crop": True,
            "action_sampling": "greedy",
            "task_selection_success_interval": [0.40, 0.85],
            "task_selection_tiebreak": "closest_to_0.625_then_task_id",
            "selected_task_count": 3,
        },
        "phase1": {
            "states_per_task": 50,
            "phase_targets": {"approach": 0.25, "pre_grasp": 0.25, "grasp_contact": 0.25, "transport_place": 0.25},
            "visual_tokens_per_state": 16,
            "selection": {"attention_top": 4, "attention_middle": 4, "attention_bottom": 4, "deterministic_random": 4},
            "random_seed": 20260808,
            "attention_score": "action_token_query_to_post_projector_visual_key_attention",
            "replacement": "position_conditioned_post_projector_visual_mean",
            "primary_effect": "teacher_forced_action_token_js_divergence_with_identical_clean_action_prefix",
            "secondary_effect": "free_running_autoregressive_cascade",
            "metrics": ["js_divergence", "kl_clean_mask", "argmax_flip", "logit_margin_change", "decoded_action_delta", "cascade_amplification"],
        },
        "integrity_gates": [
            "frozen_eval_model",
            "no_mask_forward_parity",
            "changed_visual_index_exactness",
            "protected_token_integrity",
            "clean_determinism",
            "teacher_forced_action_prefix_identity",
            "official_decode_parity",
        ],
        "decision": {
            "statuses": ["AR_TOKEN_GO", "AR_TOKEN_NO_GO"],
            "go_requires_majority_of": [
                "state_level_attention_effect_spearman_median_at_least_0.30_with_cluster_bootstrap_ci_lower_above_0",
                "top_attention_js_exceeds_random_and_bottom_with_task_state_cluster_bootstrap_ci_lower_above_0",
                "at_least_0.05_of_token_action_positions_have_argmax_flip",
                "attention_effect_direction_repeats_in_all_three_tasks_and_at_least_four_of_seven_action_positions",
            ],
            "minimum_go_criteria": 3,
            "bootstrap_replicates": 2000,
        },
        "prohibited": ["classifier", "closed_loop_token_calibration", "VCAD", "parameter_tuning_after_phase1_results"],
    }
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")

    with (artifact / "baseline_manifest.jsonl").open("x", encoding="utf-8") as handle:
        for task_id in range(10):
            for init_state_index in range(20):
                row = {
                    "episode_id": f"spatial_task{task_id:02d}_init{init_state_index:02d}",
                    "task_id": task_id,
                    "init_state_index": init_state_index,
                    "env_seed": 0,
                    "global_seed": 7,
                    "status": "pending",
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    code_paths = [
        workspace / "research/ar_token_counterfactual/hf_loader.py",
        workspace / "research/ar_token_counterfactual/libero_runtime.py",
        workspace / "research/ar_token_counterfactual/init_artifact.py",
        workspace / "research/ar_token_counterfactual/intervention.py",
        workspace / "research/ar_token_counterfactual/run_integrity.py",
        workspace / "research/ar_token_counterfactual/run_baseline.py",
    ]
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite": importlib.metadata.version("robosuite"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
        "checkpoint_index_sha256": sha256(checkpoint / "model.safetensors.index.json"),
        "checkpoint_shards": shard_audit,
        "code_sha256": {str(path.relative_to(workspace)): sha256(path) for path in code_paths},
        "openvla_git_commit": "c8f03f48af692657d3060c19588038c7220e9af9",
        "checkpoint_complete_at_lock_time": True,
    }
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(artifact)


if __name__ == "__main__":
    main()
