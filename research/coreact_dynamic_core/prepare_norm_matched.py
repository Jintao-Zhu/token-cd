#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml


ARMS = ("vanilla", "random8_norm", "attention8_norm", "instability8_norm")
TARGET = 3.0661711077317473


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = (args.output or workspace / "artifacts" / f"coreact_norm_matched_selector_task4_v1_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if output.exists():
        raise FileExistsError(output)
    for name in ("phase0_states", "episodes", "invalid_pairs", "status", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)

    checkpoint = workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de"
    means = workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    source = workspace / "artifacts/coreact_dynamic_core_token_selection_rollout_v2_20260810_144650"
    protocol = {
        "experiment_name": "coreact_norm_matched_selector_task4_v1",
        "stage": "mechanism_development_not_confirmation",
        "task": 4,
        "task_language": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "available_unique_init_states": 50,
        "init_state_ids": list(range(50)),
        "requested_unique_init_states": 100,
        "resource_deviation": "LIBERO-Spatial Task 4 contains exactly 50 official init states; no duplicate init is represented as a new state",
        "arms": list(ARMS),
        "planned_pairs": 50,
        "planned_episodes": 200,
        "phase0_states": 50,
        "selectors": {
            "random8_norm": "deterministic random 8 of 128 eligible visual tokens",
            "attention8_norm": "native tau=1 late-half action-to-context attention top 8",
            "instability8_norm": "10-step adjacent JS-contribution EMA beta=0.9 top 8",
        },
        "operator": {
            "raw": "d_raw = v_clean - v_counterfactual",
            "trust_region": "d_base = -trust_region_clip(d_raw), kappa=0.25, real action dimensions only",
            "normalization": "applied = alpha*d_base; alpha=min(1,target_rms/(||d_base||+eps)) independently at every flow step",
            "target_applied_correction_rms": TARGET,
            "target_source": "median per-replan applied-correction RMS over all 899 Task-4 Attention8 replans in the locked v2 artifact",
            "target_source_artifact": str(source),
            "target_source_summary": {"replans": 899, "median": TARGET, "mean": 3.0479687545253045},
            "no_extrapolation": True,
        },
        "shared": {
            "checkpoint_frozen_eval": True,
            "same_init_reset_observation_preprocessing_noise_per_pair": True,
            "canonical_first_observation_rule": "independent reset per arm; vanilla raw reset observation is byte-identical policy input for first replan; later observations arm-native",
            "flow_steps": 10,
            "chunk_size": 50,
            "executed_actions_per_chunk": 10,
            "max_control_steps": 280,
            "replacement": "v8 position-conditioned camera/token visual mean",
            "selection_recomputed_each_replan": True,
        },
        "phase0_gate": {
            "exactly_8_changed_visual_tokens": True,
            "finite": True,
            "clean_flow_parity": True,
            "deterministic_repeat": True,
            "attention_and_instability_median_ratio_to_attention_range": [0.95, 1.05],
            "random_under_match_is_reported_not_extrapolated": True,
        },
        "seed_rules": {
            "reset": "160000000 + task*1000 + init",
            "noise_base": "202608140000 + task*100000 + init*1000; add replan index",
            "selection": "20260814 + task*10000 + init*100 + replan",
        },
        "analysis": {
            "no_outcome_summary_before_all_planned_episodes": True,
            "primary_pairs": ["instability8_norm-random8_norm", "attention8_norm-random8_norm", "instability8_norm-attention8_norm"],
            "paired_bootstrap_replicates": 10000,
            "exact_mcnemar": True,
        },
        "checkpoint": {
            "repo": "lerobot/smolvla_libero",
            "revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
            "config_sha256": sha(checkpoint / "config.json"),
            "weights_sha256": sha(checkpoint / "model.safetensors"),
        },
        "calibration_mean_sha256": sha(means),
        "code_sha256": sha(workspace / "research/coreact_dynamic_core/dynamic_guidance.py"),
    }
    (output / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    rows = []
    for init in range(50):
        for arm in ARMS:
            rows.append({
                "pair_id": f"task04__init{init:02d}",
                "episode_id": f"task04__init{init:02d}__{arm}",
                "task_id": 4,
                "init_state_id": init,
                "arm": arm,
                "reset_seed": 160004000 + init,
                "noise_seed_base": 202608540000 + init * 1000,
                "selection_seed_base": 20260814 + 40000 + init * 100,
            })
    (output / "episode_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (output / "decision.json").write_text(json.dumps({"decision": "PHASE0_PENDING_NO_ROLLOUT", "planned_pairs": 50, "planned_episodes": 200}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
