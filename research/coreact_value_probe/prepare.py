#!/usr/bin/env python3
"""Lock the success-value probe protocol and build an auditable vanilla replay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml


TRAIN_TASKS = (0, 1, 2, 5, 6, 7, 9)
VALIDATION_TASKS = (3,)
TEST_TASKS = (4,)
SOURCE_GLOBS = {
    0: "coreact_task0_top8_sign_diagnostic_v1_*",
    1: "coreact_task1_top8_sign_diagnostic_v1_*",
    2: "coreact_task2_top8_sign_diagnostic_v1_*",
    3: "coreact_task3_top8_sign_diagnostic_v1_*",
    4: "coreact_task4_sign_diagnostic_v1_*",
    5: "coreact_task5_top8_sign_diagnostic_v1_*",
    6: "coreact_task6_top8_sign_diagnostic_v1_*",
    7: "coreact_task7_top8_sign_diagnostic_v1_*",
    9: "coreact_task9_top8_sign_diagnostic_v1_*",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def newest_source(artifacts: Path, task_id: int) -> Path:
    matches = sorted(artifacts.glob(SOURCE_GLOBS[task_id]))
    if not matches:
        raise FileNotFoundError(f"no source artifact for task {task_id}")
    return matches[-1]


def vanilla_specs(source: Path) -> list[dict]:
    specs = [row for row in read_jsonl(source / "episode_manifest.jsonl") if row["condition"] == "vanilla"]
    if len(specs) != 50 or {row["init_state_id"] for row in specs} != set(range(50)):
        raise RuntimeError(f"{source} does not contain exactly vanilla init states 0-49")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=False)
    (artifact / "captures").mkdir()
    (artifact / "logs").mkdir()

    protocol = {
        "experiment_name": "success_value_token_causal_probe_v1",
        "stage": "single_state_value_probe_qualification",
        "policy": "frozen lerobot/smolvla_libero",
        "checkpoint_revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
        "suite": "libero_spatial",
        "tasks": {"train": list(TRAIN_TASKS), "validation": list(VALIDATION_TASKS), "heldout_test": list(TEST_TASKS)},
        "episodes_per_task": 50,
        "trajectory_source": "vanilla only; deterministic replay of append-only prior rollouts",
        "capture_frequency": "every native replan, before applying the logged action",
        "label": "final success of the same vanilla episode",
        "representation": {
            "layer": "SmolVLM final layer after final RMSNorm",
            "pools": ["camera1 visual", "camera2 visual", "language valid tokens", "state"],
            "aggregation": "separate valid-token means concatenated in the listed order",
            "dtype": "float32",
            "policy_parameters": "frozen eval",
        },
        "models": {
            "linear": "StandardScaler plus logistic regression, C=1, L2",
            "mlp": "StandardScaler plus MLP hidden=64, ReLU, L2=1e-4, seed=817263, early stopping on validation task",
        },
        "sampling": "each episode total weight 1; states within episode share that weight",
        "primary_evaluation": "task-4 held out from all fitting and model selection",
        "negative_controls": ["language-only static context", "task/prevalence-only constant"],
        "metrics": ["AUROC", "average precision", "Brier score", "ECE-10", "phase AUROC"],
        "bootstrap": {"unit": "episode", "replicates": 2000, "seed": 918273},
        "phases": {
            "approach": "not grasped and eef-object distance > 0.08 m",
            "pre_grasp": "not grasped and eef-object distance <= 0.08 m",
            "grasp": "grasped and fewer than 10 control steps since first grasp",
            "transport": "first grasp occurred at least 10 control steps earlier and task predicate is false",
        },
        "qualification_gate": {
            "mandatory_capture": "450/450 episodes reproduce initial prepared hash, control-step count, and final success",
            "primary_auroc": ">=0.65 and episode-bootstrap 95% CI lower bound >0.50",
            "ap_lift_over_prevalence": ">=0.10",
            "stateful_minus_language_only_auroc": ">=0.05",
            "minimum_phase_support": "at least 10 positive and 10 negative episodes represented; otherwise phase metric is descriptive only",
        },
        "post_qualification": "Only after PASS, lock a separate new-state token-intervention protocol; no guidance tuning.",
        "known_limitation": "development replication; prior task-4 outcomes existed, but no task-4 state or latent is used for fitting or tuning",
    }
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))

    manifest = []
    source_audit = []
    split_by_task = {**{t: "train" for t in TRAIN_TASKS}, **{t: "validation" for t in VALIDATION_TASKS}, **{t: "heldout_test" for t in TEST_TASKS}}
    for task_id in sorted(SOURCE_GLOBS):
        source = newest_source(workspace / "artifacts", task_id)
        specs = vanilla_specs(source)
        successes = 0
        for spec in specs:
            episode_path = source / "episodes" / spec["episode_id"] / "episode.json"
            episode = json.loads(episode_path.read_text())
            step_path = source / episode["step_log"]
            successes += int(episode["success"])
            manifest.append({
                "capture_id": f"task{task_id:02d}__init{spec['init_state_id']:02d}",
                "task_id": task_id,
                "init_state_id": spec["init_state_id"],
                "split": split_by_task[task_id],
                "suite": spec["suite"],
                "language": spec["language"],
                "reset_seed": spec["reset_seed"],
                "source_artifact": str(source.relative_to(workspace)),
                "source_episode": str(episode_path.relative_to(workspace)),
                "source_steps": str(step_path.relative_to(workspace)),
                "source_episode_sha256": sha256(episode_path),
                "source_steps_sha256": sha256(step_path),
                "expected_success": bool(episode["success"]),
                "expected_control_steps": episode["control_steps"],
                "expected_initial_prepared_sha256": episode["initial_prepared_input_sha256"],
                "expected_initial_sim_state_sha256": episode["initial_sim_state_sha256"],
            })
        source_audit.append({"task_id": task_id, "artifact": str(source.relative_to(workspace)), "episodes": 50, "successes": successes, "failures": 50 - successes})
    with (artifact / "capture_manifest.jsonl").open("x") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (artifact / "source_audit.json").write_text(json.dumps(source_audit, indent=2, sort_keys=True) + "\n")
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "workspace": str(workspace),
        "git_status": "unavailable: dubious ownership; global git configuration was not changed",
        "manifest_rows": len(manifest),
    }
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(artifact), "captures": len(manifest), "source_audit": source_audit}, indent=2))


if __name__ == "__main__":
    main()
