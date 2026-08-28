from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


TASKS = (4, 7)
PHASE0_TASKS = (2, 9)
ARMS = ("A_vanilla", "N0_pure_negative", "W05_shrink", "W15_extrapolate", "W20_extrapolate", "REF_toward_top8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--requested-protocol", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(artifact)
    for name in ("episodes", "logs", "status"):
        (artifact / name).mkdir(parents=True)
    requested = args.requested_protocol.read_text(encoding="utf-8")
    (artifact / "requested_protocol.yaml").write_text(requested, encoding="utf-8")
    e1 = workspace / "artifacts/coreact_ensemble_vs_contrast_control_v1_20260809_152414"
    e1_summary = json.loads((e1 / "summary.json").read_text())
    if e1_summary.get("decision") != "INCONCLUSIVE":
        raise RuntimeError("self-guidance experiment is gated by E1 INCONCLUSIVE")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_manifest = {
        str(task_id): {"task_id": task_id, "language": suite.get_task(task_id).language, "bddl_file": suite.get_task(task_id).bddl_file}
        for task_id in (*TASKS, *PHASE0_TASKS)
    }
    write_json(artifact / "task_manifest.json", task_manifest)
    phase0 = []
    for task_id in PHASE0_TASKS:
        for init_state_id in range(10):
            base = 96_000_000 + task_id * 100_000 + init_state_id * 10
            phase0.append({
                "state_id": f"phase0_task{task_id:02d}_init{init_state_id:02d}",
                "task_id": task_id,
                "init_state_id": init_state_id,
                "language": task_manifest[str(task_id)]["language"],
                "reset_seed": base + 1,
                "action_noise_seed": base + 2,
                "split": "phase0_outcome_blind_calibration",
            })
    with (artifact / "phase0_state_manifest.jsonl").open("x", encoding="utf-8") as stream:
        for row in phase0:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    checkpoint = workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de"
    mean_path = workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    code_paths = [
        workspace / "research/coreact_closed_loop/guidance.py",
        workspace / "research/coreact_closed_loop/runtime.py",
        workspace / "research/coreact_self_guidance/sampler.py",
        workspace / "research/coreact_self_guidance/prepare.py",
        workspace / "research/coreact_self_guidance/calibrate.py",
        workspace / "research/coreact_self_guidance/integrity.py",
        workspace / "research/coreact_self_guidance/run.py",
        workspace / "research/coreact_self_guidance/analyze.py",
    ]
    if any(not path.exists() for path in code_paths):
        raise RuntimeError("self-guidance modules incomplete")
    protocol = {
        "experiment_name": "coreact_self_guidance_negative_branch_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "mechanism_development_not_confirmation",
        "gated_by": "E1 decision INCONCLUSIVE",
        "confirmation_claim_allowed": False,
        "parameters_tuned": False,
        "tasks": [{"id": 4, "role": "mechanism_reference_burned_task_no_parameter_selection"}, {"id": 7, "role": "dynamic_range_task"}],
        "phase0_states": 20,
        "phase0_tasks": list(PHASE0_TASKS),
        "phase0_sweep_delta": [0.1, 0.2, 0.3, 0.4, 0.5],
        "locked_delta": "TBD_from_phase0",
        "negative_branch": {
            "primary_construction": "deterministic_back_projection",
            "time_convention": "protocol progress s=1-model_tau; model_tau decreases 1 to 0",
            "formula": "v_neg = v(reference_x[s-delta], s-delta), native clean reference trajectory",
            "fallback_used": False,
            "no_token_selection": True,
            "boundary_rule": "s-delta<0 skips guidance and uses v_clean",
        },
        "checkpoint": {"repo": "lerobot/smolvla_libero", "revision": "31d453f7edd78c839a8bbc39744a292686daf0de"},
        "shared": {"flow_steps": 10, "chunk_size": 50, "executed_actions_per_chunk": 10, "maximum_control_steps": 280, "real_action_dimensions": 7, "same_init_reset_noise_preprocessing": True, "frozen_fp32_eval": True, "guidance_scale": 1.0, "trust_region_kappa": 0.25},
        "arms": list(ARMS),
        "w_values": {"A_vanilla": 1.0, "N0_pure_negative": 0.0, "W05_shrink": 0.5, "W15_extrapolate": 1.5, "W20_extrapolate": 2.0, "REF_toward_top8": 0.5},
        "total_episodes": 600,
        "statistics": {"primary": "W15_extrapolate_minus_A_vanilla", "secondary": "W20_extrapolate_minus_A_vanilla", "bootstrap_replicates": 2000, "bootstrap_unit": "init_state_id", "bootstrap_seed": 20260809, "paired_binary_test": "exact_McNemar", "multiplicity": "Holm across primary and secondary"},
        "analysis_after_600_of_600_only": True,
        "parameters_must_not_change_after_outcomes": True,
        "source_artifacts": {"E1": str(e1), "E1_summary_sha256": sha256(e1 / "summary.json"), "E1_protocol_sha256": sha256(e1 / "protocol.lock.yaml")},
        "hashes": {"checkpoint_config": sha256(checkpoint / "config.json"), "checkpoint_weights": sha256(checkpoint / "model.safetensors"), "calibration_mean": sha256(mean_path), "code": {str(path.relative_to(workspace)): sha256(path) for path in code_paths}},
    }
    (artifact / "protocol.phase0.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    git_probe = subprocess.run(["git", "-C", str(workspace / "lerobot"), "rev-parse", "HEAD"], capture_output=True, text=True)
    write_json(artifact / "environment.json", {"created_at": protocol["created_at"], "hostname": platform.node(), "python": platform.python_version(), "workspace_git_status": "unavailable: ownership/safe-directory restriction", "lerobot_git_commit": git_probe.stdout.strip() or None, "lerobot_git_status_limitation": git_probe.stderr.strip() or None, "phase0_manifest_sha256": sha256(artifact / "phase0_state_manifest.jsonl")})
    print(json.dumps({"artifact": str(artifact), "phase0_states": len(phase0), "arms": ARMS, "gate": "E1_INCONCLUSIVE"}, indent=2))


if __name__ == "__main__": main()
