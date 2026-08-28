#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import yaml

from research.coreact_quality_negative_branch.quality_branch import BRANCHES, FLOW_TIMES


CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
DATA_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
TASKS = (4, 7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = (
        args.output
        or workspace
        / "artifacts"
        / f"coreact_quality_ordered_negative_branch_phase0_v1_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    if artifact.exists():
        raise FileExistsError(artifact)
    for child in ("raw", "plots", "status", "logs"):
        (artifact / child).mkdir(parents=True, exist_ok=True)

    import sys

    sys.path.insert(0, str(workspace / "LIBERO"))
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    data_root = workspace / "LIBERO/libero/datasets/libero_spatial"
    states = []
    units = []
    sources = []
    for task_id in TASKS:
        language = suite.get_task(task_id).language
        path = data_root / (language.replace(" ", "_") + "_demo.hdf5")
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            demos = sorted(handle["data"], key=lambda value: int(value.split("_")[-1]))
            if len(demos) != 50:
                raise RuntimeError(f"{path}: expected exactly 50 expert demos")
            lengths = []
            for ordinal, demo_id in enumerate(demos):
                episode = handle["data"][demo_id]
                action_count = len(episode["actions"])
                if len(episode["states"]) != action_count or action_count < 2:
                    raise RuntimeError(f"invalid expert episode task={task_id} demo={demo_id}")
                frame = math.floor((action_count - 1) * 0.5)
                state_id = f"task{task_id:02d}__demo{ordinal:02d}__middle"
                state = {
                    "state_id": state_id,
                    "task_id": task_id,
                    "language": language,
                    "demo_id": demo_id,
                    "demo_ordinal": ordinal,
                    "episode_length": action_count,
                    "resolved_frame": frame,
                    "target_progress": 0.5,
                    "demo_path": str(path),
                }
                states.append(state)
                lengths.append(action_count)
                for noise_ordinal in range(3):
                    units.append(
                        {
                            **state,
                            "unit_id": f"{state_id}__noise{noise_ordinal}",
                            "noise_ordinal": noise_ordinal,
                            "noise_seed": 202608160000
                            + task_id * 10000
                            + ordinal * 10
                            + noise_ordinal,
                        }
                    )
        sources.append(
            {
                "task_id": task_id,
                "language": language,
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "demo_count": 50,
                "min_episode_length": min(lengths),
                "max_episode_length": max(lengths),
            }
        )

    checkpoint = (
        workspace
        / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots"
        / CHECKPOINT_REVISION
    )
    protocol = {
        "experiment_name": "coreact_quality_ordered_negative_branch_phase0_v1",
        "stage": "offline_phase0_negative_branch_qualification_no_rollout",
        "scope": {
            "suite": "libero_spatial",
            "tasks": list(TASKS),
            "role": "development_mechanism_qualification",
            "expert_states_per_task": 50,
            "state_selection": "one middle-progress state from each of 50 expert demos; frame=floor((N-1)*0.5)",
            "noise_seeds_per_state": 3,
            "flow_times": list(FLOW_TIMES),
            "state_noise_units": 300,
            "points_per_branch": 3000,
            "outcome_filtering": False,
            "closed_loop_forbidden": True,
        },
        "strong_branch": "unmodified full SmolVLA action expert with identical observation, language, state, x_t, timestep, and noise",
        "flow_target": {
            "source": "SmolVLAPytorch.build_flow_training_pair, called by the checkpoint training forward",
            "x_t": "t*noise + (1-t)*normalized_expert_action",
            "u_star": "noise - normalized_expert_action",
        },
        "weak_branches": [
            {
                "name": branch.name,
                "affected_expert_layers": branch.affected_layers,
                "residual_scale": branch.residual_scale,
                "definition": "h_out = h_in + residual_scale*(full_expert_layer(h_in)-h_in)",
            }
            for branch in BRANCHES
        ],
        "forbidden_interventions": [
            "visual_token_masking",
            "attention_top_k",
            "projector_token_intervention",
            "prompt_perturbation",
            "timestep_shift",
            "observation_corruption",
        ],
        "metrics": {
            "quality_ordering": "delta_Q = mean_squared_error(weak,u_star) - mean_squared_error(strong,u_star)",
            "extrapolation_validity": "EV = dot(strong-weak,u_star-strong)",
            "ev_cosine": "cos(strong-weak,u_star-strong)",
            "lambda_star": "EV/(squared_norm(strong-weak)+1e-12)",
            "geometry": ["cos(strong,weak)", "norm(strong-weak)/(norm(strong)+1e-12)"],
            "secondary_lambdas": [0.1, 0.25, 0.5],
            "valid_dimensions": "non-padded expert action steps and seven real action dimensions",
        },
        "aggregation": {
            "state": "aggregate three noise seeds and ten flow steps before inference",
            "bootstrap_cluster": "expert state",
            "bootstrap_replicates": 10000,
            "tasks_reported_separately": True,
        },
        "go_gate_each_task": {
            "quality_ordering_rate_min": 0.65,
            "quality_ordering_bootstrap_ci_lower_strictly_above": 0.50,
            "ev_positive_rate_min": 0.60,
            "ev_bootstrap_ci_lower_strictly_above": 0.50,
            "median_strong_weak_cosine_strictly_above": 0.8,
            "median_relative_correction_norm_strict_range": [0.02, 0.5],
            "flow_steps_quality_rate_above_half_min": 6,
            "flow_steps_ev_rate_above_half_min": 6,
        },
        "decision_rules": {
            "none_pass": "NEGATIVE_BRANCH_QUALIFICATION_FAILED_NO_ROLLOUT",
            "at_least_one_pass": "QUALITY_ORDERED_NEGATIVE_BRANCH_FOUND_READY_FOR_CONFIRMATION",
            "automatic_closed_loop": False,
            "no_posthoc_tuning": ["block_count", "attenuation", "lambda", "timestep_window"],
        },
        "data": {
            "repo": "yifengzhu-hf/LIBERO-datasets",
            "revision": DATA_REVISION,
            "sources": sources,
        },
        "checkpoint": {
            "repo": "lerobot/smolvla_libero",
            "revision": CHECKPOINT_REVISION,
            "config_sha256": sha256(checkpoint / "config.json"),
            "weights_sha256": sha256(checkpoint / "model.safetensors"),
            "dtype": "fp32",
            "frozen_eval": True,
        },
        "code_sha256": {
            "quality_branch": sha256(Path(__file__).with_name("quality_branch.py")),
            "modeling_smolvla": sha256(
                workspace / "lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py"
            ),
            "smolvlm_with_expert": sha256(
                workspace / "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py"
            ),
        },
    }
    (artifact / "protocol.lock.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=False)
    )
    (artifact / "state_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in states)
    )
    (artifact / "unit_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in units)
    )
    (artifact / "source_audit.json").write_text(
        json.dumps({"sources": sources, "states": len(states), "units": len(units)}, indent=2) + "\n"
    )
    (artifact / "architecture_audit.md").write_text(
        """# SmolVLA Action-Expert Architecture Audit

## Checkpoint facts

- The checkpoint config uses `num_vlm_layers=16`, `num_expert_layers=0`, `attention_mode=cross_attn`, `self_attn_every_n_layers=2`, and `expert_width_multiplier=0.75`.
- In `SmolVLMWithExpertModel`, `num_expert_layers=0` means the expert is constructed with the same 16-layer depth as the truncated VLM. Runtime verification of the realized counts is mandatory before capture.
- The action expert is a separate transformer hidden stream. Each expert layer applies an attention projection plus residual, followed by post-attention normalization, MLP, and a second residual.
- Even-indexed layers use joint self-attention under the configured cadence; odd-indexed layers use explicit expert cross-attention to VLM prefix key/value states.
- After all expert layers, the expert stream passes through its own final RMSNorm. The last 50 action-token hidden states then pass through `action_out_proj` to produce velocity.

## Preregistered structural degradation

The mathematical unit is one complete expert transformer layer, treated as a composite residual block. For an affected layer, the implementation computes the unchanged full layer output and applies:

`h_out = h_in + alpha * (h_full - h_in)`

W1/W2 set `alpha=0` on the last 1/2 expert layers; W3/W4 set `alpha=0.5` on the last 1/2 expert layers. This leaves all VLM layers, attention inputs, observation/language/state tokens, timestep/action embeddings, expert final norm, and action output head unchanged. No alternative branch definition is authorized.
"""
    )
    (artifact / "decision.json").write_text(
        json.dumps(
            {
                "decision": "PHASE0_PROTOCOL_LOCKED_PENDING_INTEGRITY",
                "closed_loop_authorized": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(artifact)


if __name__ == "__main__":
    main()
