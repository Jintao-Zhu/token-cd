from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import yaml


ARMS = ("A_vanilla", "B_toward_top8", "C_dual_seed_chunk_average", "D_random8_toward")
TASKS = (4, 7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clipping_audit(source: Path) -> dict:
    paths = sorted(source.glob("episodes/*attention_toward/episode.json"))
    replans = active_replans = steps = active_steps = 0
    for path in paths:
        record = json.loads(path.read_text())
        for replan in record["replan_traces"]:
            scales = [float(step["clip_scale"]) for step in replan["step_traces"]]
            replans += 1
            active_replans += int(any(scale < 1.0 - 1e-12 for scale in scales))
            steps += len(scales)
            active_steps += sum(scale < 1.0 - 1e-12 for scale in scales)
    if len(paths) != 50 or replans == 0:
        raise RuntimeError("clip saturation source is incomplete")
    return {
        "source": str(source),
        "episodes": len(paths),
        "replans": replans,
        "active_replans": active_replans,
        "fraction_of_replans_with_trust_region_clipping_active": active_replans / replans,
        "flow_steps": steps,
        "active_flow_steps": active_steps,
        "fraction_of_flow_steps_with_clipping_active": active_steps / steps,
        "matched_step_variant_enabled": active_replans / replans >= 0.80,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--requested-protocol", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(f"refusing to overwrite {artifact}")
    for name in ("episodes", "logs", "status"):
        (artifact / name).mkdir(parents=True)

    clip_source = workspace / "artifacts/coreact_consistency_direction_task4_50_v1_20260808_131410"
    clip = clipping_audit(clip_source)
    second_source = workspace / "artifacts/coreact_spatial_task7_top8_replication_v1_20260807_160702"
    second_summary = json.loads((second_source / "analysis_summary.json").read_text())
    if second_summary["task_id"] != 7 or second_summary["success_rates"]["vanilla"] != 0.78:
        raise RuntimeError("second-task source changed")
    second_task_audit = {
        "rule": "exclude tasks 4 and 8; choose vanilla success in [0.50,0.80] closest to 0.65",
        "selected_task_id": 7,
        "selected_task_vanilla_success_rate": 0.78,
        "source": str(second_source),
        "source_analysis_sha256": sha256(second_source / "analysis_summary.json"),
        "caveat": "Selection used known vanilla success and was not fully blinded.",
    }
    write_json(artifact / "precondition_audit.json", {"clip_saturation": clip, "second_task_selection": second_task_audit})
    (artifact / "requested_protocol.yaml").write_text(args.requested_protocol.read_text(), encoding="utf-8")

    checkpoint = workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de"
    mean_path = workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    code_paths = [
        workspace / "research/coreact_closed_loop/guidance.py",
        workspace / "research/coreact_closed_loop/runtime.py",
        workspace / "research/coreact_ensemble_control/methods.py",
        workspace / "research/coreact_ensemble_control/prepare.py",
        workspace / "research/coreact_ensemble_control/integrity.py",
        workspace / "research/coreact_ensemble_control/run.py",
        workspace / "research/coreact_ensemble_control/analyze.py",
    ]
    missing = [str(path) for path in code_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"experiment modules incomplete: {missing}")

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_manifest = {
        str(task_id): {
            "task_id": task_id,
            "language": suite.get_task(task_id).language,
            "bddl_file": suite.get_task(task_id).problem_folder + "/" + suite.get_task(task_id).bddl_file,
        }
        for task_id in TASKS
    }
    write_json(artifact / "task_manifest.json", task_manifest)
    created_at = datetime.now().astimezone().isoformat()
    protocol = {
        "experiment_name": "coreact_ensemble_vs_contrast_control_v1",
        "created_at": created_at,
        "stage": "mechanism_decision_experiment_not_confirmation",
        "confirmation_claim_allowed": False,
        "prior_task4_outcomes_known": True,
        "tasks": [
            {"id": 4, "role": "mechanism_reference_burned_task_no_parameter_selection"},
            {"id": 7, "role": "dynamic_range_task"},
        ],
        "suite": "libero_spatial",
        "checkpoint": {"repo": "lerobot/smolvla_libero", "revision": "31d453f7edd78c839a8bbc39744a292686daf0de"},
        "init_state_ids": list(range(50)),
        "arms": list(ARMS),
        "total_episodes": 400,
        "shared": {
            "flow_steps": 10,
            "chunk_size": 50,
            "executed_actions_per_chunk": 10,
            "maximum_control_steps": 280,
            "real_action_dimensions": 7,
            "guidance_scale": 0.5,
            "trust_region_kappa": 0.25,
            "flow_time_for_selection": 1.0,
            "rerank_every_replan": True,
            "eligible_visual_tokens": 128,
            "top_k": 8,
            "protected": ["language", "state", "special", "padding"],
            "replacement": "v8_position_conditioned_camera_visual_mean",
            "render_warmup_resets_without_action": 1,
            "official_same_seed_resets": 1,
            "model_frozen_fp32_eval": True,
        },
        "arm_C": {
            "integrations_per_replan": 2,
            "matched_step_variant_enabled": clip["matched_step_variant_enabled"],
            "procedure": "average two native chunks; clip real-dimension offset from first chunk at kappa times first-chunk L2 norm",
            "masked_tokens": 0,
        },
        "arm_D": {
            "selection": "uniform 8 without replacement from 128 eligible visual tokens",
            "resample_every_replan": True,
            "formula": "clean - 0.5 * trust_region_clip(clean - masked, kappa=0.25)",
        },
        "seed_derivation": {
            "pair_base": "94000000 + task_id*100000 + init_state_id*10",
            "reset_seed": "pair_base + 1",
            "first_action_noise_seed": "pair_base + 2; per replan torch seed = value*1000 + replan_index",
            "selection_seed": "pair_base + 3; per replan torch seed = value*1000 + replan_index",
            "second_action_noise_seed": "pair_base + 4; per replan torch seed = value*1000 + replan_index",
        },
        "statistics": {
            "primary": "C_dual_seed_chunk_average_minus_A_vanilla",
            "secondary": "D_random8_toward_minus_A_vanilla",
            "replication": "B_toward_top8_minus_A_vanilla",
            "mechanism": "B_toward_top8_minus_D_random8_toward",
            "pooling": "tasks separately; pooled descriptive only",
            "paired_bootstrap_replicates": 2000,
            "bootstrap_seed": 20260809,
            "paired_binary_test": "exact_McNemar",
            "multiplicity": "Holm across primary and secondary task-level tests",
        },
        "analysis_after_400_of_400_only": True,
        "parameters_must_not_change_after_outcomes": True,
        "precondition_audit": clip,
        "hashes": {
            "checkpoint_config": sha256(checkpoint / "config.json"),
            "checkpoint_weights": sha256(checkpoint / "model.safetensors"),
            "calibration_mean": sha256(mean_path),
            "code": {str(path.relative_to(workspace)): sha256(path) for path in code_paths},
        },
    }
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")

    manifest = []
    for task_id in TASKS:
        language = task_manifest[str(task_id)]["language"]
        for init_state_id in range(50):
            base = 94_000_000 + task_id * 100_000 + init_state_id * 10
            pair_id = f"task{task_id:02d}__init{init_state_id:02d}"
            for arm in ARMS:
                manifest.append(
                    {
                        "episode_id": f"{pair_id}__{arm}",
                        "pair_id": pair_id,
                        "suite": "libero_spatial",
                        "task_id": task_id,
                        "init_state_id": init_state_id,
                        "arm": arm,
                        "language": language,
                        "reset_seed": base + 1,
                        "action_noise_seed": base + 2,
                        "selection_seed": base + 3,
                        "second_action_noise_seed": base + 4,
                    }
                )
    if len(manifest) != 400 or len({row["episode_id"] for row in manifest}) != 400:
        raise RuntimeError("manifest cardinality failure")
    manifest_path = artifact / "episode_manifest.jsonl"
    with manifest_path.open("x", encoding="utf-8") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    git_probe = subprocess.run(
        ["git", "-C", str(workspace / "lerobot"), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    write_json(
        artifact / "environment.json",
        {
            "created_at": created_at,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "workspace_git_status": "unavailable: workspace root is not a git repository",
            "lerobot_git_commit": git_probe.stdout.strip() or None,
            "lerobot_git_status_limitation": git_probe.stderr.strip() or None,
            "manifest_sha256": sha256(manifest_path),
        },
    )
    print(json.dumps({"artifact": str(artifact), "episodes": len(manifest), "tasks": TASKS, "arms": ARMS, "clip": clip}, indent=2))


if __name__ == "__main__":
    main()
