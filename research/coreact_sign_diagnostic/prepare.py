#!/usr/bin/env python3
"""Prepare the locked Task-4 away-versus-toward sign diagnostic."""

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


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--task-id", type=int, default=4)
    args = parser.parse_args()
    if args.top_k not in (8, 16, 32): raise ValueError("sign diagnostic supports preregistered top-k values 8, 16, or 32")
    conditions = ("vanilla", f"top{args.top_k}_mask_only", f"coreact_away_top{args.top_k}", f"coreact_toward_top{args.top_k}")
    workspace = args.workspace.resolve(); suite_name = "libero_spatial"; task_id = args.task_id
    suite = benchmark.get_benchmark_dict()[suite_name](); task = suite.get_task(task_id)
    if len(suite.get_task_init_states(task_id)) != 50: raise RuntimeError(f"Task {task_id} must have 50 official init states")
    artifact = workspace / "artifacts" / f"coreact_task{task_id}_top{args.top_k}_sign_diagnostic_v1_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    artifact.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoint_path(workspace); means = workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    code_paths = {
        "guidance": "research/coreact_closed_loop/guidance.py", "masked_sampler": "research/coreact_revision/masked_sampler.py",
        "runtime": "research/coreact_closed_loop/runtime.py", "runner": "research/coreact_task4_replication/run.py",
        "prepare": "research/coreact_sign_diagnostic/prepare.py", "integrity": "research/coreact_sign_diagnostic/integrity.py",
        "analyzer": "research/coreact_sign_diagnostic/analyze.py", "attention_hook": "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    }
    protocol = {
        "experiment_name": f"coreact_task{task_id}_top{args.top_k}_guidance_sign_diagnostic_v1", "stage": "mechanism_development_diagnostic_not_confirmation",
        "hypothesis": f"clean-minus-masked may have the wrong control sign on Task {task_id}", "task_id": task_id, "suite": suite_name,
        "task": task.language, "official_init_state_ids": list(range(50)), "paired_states": 50, "episodes": 200,
        "conditions": list(conditions), "prior_same_task_outcomes_may_be_known": task_id == 7, "parameters_tuned": False,
        "backbone": {"repo": "lerobot/smolvla_libero", "revision": CHECKPOINT_REVISION, "frozen_eval": True},
        "shared": {"same_simulator_init_state": True, "same_reset_seed": True, "same_gaussian_noise_per_replan": True,
            "same_preprocessing": True, "render_warmup_resets_without_action": 1, "official_same_seed_resets": 1,
            "flow_steps": 10, "chunk_size": 50, "executed_actions_per_chunk": 10, "maximum_control_steps": 280,
            "rerank_current_two_camera_observation_every_replan": True},
        "selection": {"native_unintervened_prefix": True, "flow_time": 1.0, "score": "late_half_action_to_context_attention",
            "eligible_visual_tokens": 128, "top_k": args.top_k, "protected": ["language", "state", "special", "padding"],
            "replacement": "v8_position_conditioned_camera_visual_mean"},
        "away": {"formula": "clean + 0.5 * clip(clean - masked)", "guidance_scale": 0.5, "trust_region_kappa": 0.25},
        "toward": {"formula": "clean - 0.5 * clip(clean - masked)", "guidance_scale": 0.5, "trust_region_kappa": 0.25},
        "mask_only": {"masked_velocity_only_all_flow_steps": True}, "real_action_dimensions": 7,
        "statistics": {"paired_bootstrap_replicates": 2000, "bootstrap_seed": 84_004_004, "paired_binary_test": "exact_McNemar",
            "primary_comparison": f"coreact_toward_top{args.top_k}_minus_coreact_away_top{args.top_k}"},
        "prohibited": ["changing_scale", "changing_kappa", "changing_top_k", "changing_replacement", "reading_partial_success_summary"],
    }
    protocol["hashes"] = {"checkpoint_config": sha256_file(checkpoint/"config.json"), "checkpoint_weights": sha256_file(checkpoint/"model.safetensors"),
        "calibration_mean": sha256_file(means), "code": {k: hashlib.sha256((workspace/v).read_bytes()).hexdigest() for k,v in code_paths.items()}}
    rows=[]
    for init_id in range(50):
        shared=432_000_000+task_id*100_000+init_id*100; pair=f"sign_diagnostic__{suite_name}__task{task_id:02d}__init{init_id:02d}"
        for condition in conditions:
            rows.append({"episode_id":f"{pair}__{condition}","pair_id":pair,"suite":suite_name,"task_id":task_id,"task_name":task.name,
                "language":task.language,"init_state_id":init_id,"condition":condition,"group_count":args.top_k,"reset_seed":shared+1,"action_noise_seed":shared+2,"selection_seed":shared+3})
    dumped=yaml.safe_dump(protocol,sort_keys=False); (artifact/"protocol.yaml").write_text(dumped); (artifact/"protocol.lock.yaml").write_text(dumped)
    (artifact/"environment.json").write_text(json.dumps({"timestamp":datetime.now().astimezone().isoformat(),"hostname":socket.gethostname(),"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)},indent=2,sort_keys=True)+"\n")
    (artifact/"task_manifest.json").write_text(json.dumps({"suite":suite_name,"task_id":task_id,"task_name":task.name,"language":task.language,"bddl_file":task.bddl_file},indent=2,sort_keys=True)+"\n")
    with (artifact/"episode_manifest.jsonl").open("x") as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
    print(artifact)


if __name__ == "__main__": main()
