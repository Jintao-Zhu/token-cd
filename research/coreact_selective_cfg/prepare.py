#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import sys

import h5py
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "lerobot/src"))

from research.coreact_quality_negative_branch.quality_branch import BRANCHES, FLOW_TIMES
from research.coreact_selective_cfg.features import (
    GEOMETRY_CONSENSUS_FEATURES,
    GEOMETRY_LOCAL_FEATURES,
    assert_deployable_feature_names,
)


TASKS = tuple(range(10))
CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
DATA_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
PRIMARY_BRANCH = "W4_half_last_2"


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
        / f"coreact_selective_cfg_ev_validity_phase1_v1_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    if artifact.exists():
        raise FileExistsError(artifact)
    for child in ("raw", "plots", "status", "logs", "models"):
        (artifact / child).mkdir(parents=True, exist_ok=True)

    assert_deployable_feature_names(GEOMETRY_LOCAL_FEATURES)
    assert_deployable_feature_names(GEOMETRY_CONSENSUS_FEATURES)

    import sys

    sys.path.insert(0, str(workspace / "LIBERO"))
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    data_root = workspace / "LIBERO/libero/datasets/libero_spatial"
    states: list[dict] = []
    units: list[dict] = []
    sources: list[dict] = []
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
                            "noise_seed": 202608210000
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
    variants = {
        "geometry_local": list(GEOMETRY_LOCAL_FEATURES),
        "geometry_consensus": list(GEOMETRY_CONSENSUS_FEATURES),
        "geometry_consensus_language": list(GEOMETRY_CONSENSUS_FEATURES),
    }
    protocol = {
        "experiment_name": "coreact_selective_cfg_ev_validity_phase1_v1",
        "stage": "offline_phase1_ev_validity_audit_no_rollout",
        "research_question": "Can deployable strong/weak geometry predict EV>0 on a completely held-out LIBERO-Spatial task and support high-precision abstention?",
        "scope": {
            "suite": "libero_spatial",
            "tasks": list(TASKS),
            "role": "offline_task_generalization_qualification",
            "expert_states_per_task": 50,
            "state_selection": "one middle-progress state from every expert demo; frame=floor((N-1)*0.5)",
            "noise_seeds_per_state": 3,
            "flow_times": list(FLOW_TIMES),
            "states": 500,
            "state_noise_units": 1500,
            "matched_points": 15000,
            "point_rows": 60000,
            "outcome_filtering": False,
            "closed_loop_forbidden": True,
        },
        "branches": [
            {
                "name": branch.name,
                "affected_expert_layers": branch.affected_layers,
                "residual_scale": branch.residual_scale,
            }
            for branch in BRANCHES
        ],
        "branch_policy": {
            "no_new_branch": True,
            "primary_target": PRIMARY_BRANCH,
            "reason": "W4 was preregistered before Phase-0 and was the only single-task full-gate pass; W1-W3 remain descriptive robustness audits.",
            "no_branch_selection_from_phase1_outcomes": True,
        },
        "label": {
            "name": "ev_positive",
            "definition": "dot(v_strong-v_weak, u_star-v_strong) > 0",
            "use": "training/evaluation label only; forbidden from inference features",
        },
        "features": {
            "deployment_constraint": "all features must be computable from frozen model outputs, branch directions, timestep, and optionally frozen task-language input embeddings; no expert target or outcome",
            "variants": variants,
            "language_variant": "mean of valid frozen SmolVLA input-token embeddings, reduced by PCA fitted only on training-task language vectors inside every fold",
            "multi_branch_requirement": "all W1-W4 use identical observation, language, state, x_t, timestep, and noise",
        },
        "validation": {
            "outer": "leave-one-task-out over all 10 tasks",
            "inner": "for each outer fold, leave-one-task-out over the remaining 9 tasks to create out-of-task predictions used for Platt calibration and threshold selection",
            "model": {
                "class": "sklearn.ensemble.HistGradientBoostingClassifier",
                "learning_rate": 0.05,
                "max_iter": 200,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 100,
                "l2_regularization": 1.0,
                "random_state": 20260811,
            },
            "calibration": "Platt logistic calibration fitted only to inner out-of-task predictions",
            "threshold_candidates": "inner calibrated prediction quantiles [0.50,0.55,0.60,0.65,0.70], corresponding to nominal coverage 50%-30%",
            "threshold_selection": "maximize mean inner-task precision, then minimum inner-task precision, then closeness to 40% coverage; every inner task must retain >=10% coverage",
            "outer_task_never_used_for": ["model fitting", "PCA", "calibration", "threshold selection"],
            "primary_unit": "held-out task",
            "uncertainty": "bootstrap expert states as clusters within task; pooled inference also resamples tasks",
        },
        "go_gate": {
            "decision_variant": "geometry_consensus",
            "decision_branch": PRIMARY_BRANCH,
            "mean_heldout_coverage_range_inclusive": [0.30, 0.50],
            "minimum_each_task_coverage": 0.15,
            "macro_precision_min": 0.70,
            "tasks_precision_at_least_0_70_min": 7,
            "tasks_precision_above_ungated_prevalence_min": 8,
            "task_bootstrap_pooled_precision_ci_lower_min": 0.65,
            "macro_ece_max": 0.15,
            "tasks_brier_better_than_constant_prevalence_min": 8,
            "language_variant_can_trigger_go": False,
            "secondary_branches_can_trigger_go": False,
        },
        "decision_rules": {
            "pass": "EV_VALIDITY_GATE_FOUND_READY_FOR_SELECTIVE_CFG_CONFIRMATION_PROTOCOL",
            "fail": "EV_VALIDITY_GATE_NOT_ESTABLISHED_NO_ROLLOUT",
            "automatic_closed_loop": False,
            "no_posthoc_tuning": [
                "branch definition",
                "feature list",
                "model family",
                "threshold candidate grid",
                "coverage target",
                "go gate",
            ],
        },
        "data": {"repo": "yifengzhu-hf/LIBERO-datasets", "revision": DATA_REVISION, "sources": sources},
        "checkpoint": {
            "repo": "lerobot/smolvla_libero",
            "revision": CHECKPOINT_REVISION,
            "config_sha256": sha256(checkpoint / "config.json"),
            "weights_sha256": sha256(checkpoint / "model.safetensors"),
            "dtype": "fp32",
            "frozen_eval": True,
        },
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("prepare.py", "capture.py", "analyze.py", "features.py")
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
        json.dumps({"sources": sources, "states": len(states), "units": len(units)}, indent=2)
        + "\n"
    )
    (artifact / "decision.json").write_text(
        json.dumps(
            {
                "decision": "PHASE1_PROTOCOL_LOCKED_PENDING_CAPTURE",
                "closed_loop_authorized": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(artifact)


if __name__ == "__main__":
    main()
