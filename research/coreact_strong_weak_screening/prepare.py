#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import yaml

from research.coreact_closed_loop.runtime import CHECKPOINT_REVISION, checkpoint_path
from research.coreact_strong_weak_screening.sampler import ARMS, GUIDANCE_LAMBDA


TASKS = tuple(range(10))
INIT_STATES = tuple(range(10))


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
        / f"coreact_strong_weak_cfg_closed_loop_screening_v1_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    if artifact.exists():
        raise FileExistsError(artifact)
    for child in ("episodes", "logs", "plots", "status"):
        (artifact / child).mkdir(parents=True, exist_ok=True)

    import sys

    sys.path.insert(0, str(workspace / "LIBERO"))
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_manifest = {
        str(task): {
            "task_id": task,
            "language": suite.get_task(task).language,
            "bddl_file": suite.get_task(task).bddl_file,
        }
        for task in TASKS
    }
    (artifact / "task_manifest.json").write_text(
        json.dumps(task_manifest, indent=2, sort_keys=True) + "\n"
    )
    checkpoint = checkpoint_path(workspace)
    code_paths = sorted((workspace / "research/coreact_strong_weak_screening").glob("*.py"))
    protocol = {
        "experiment_name": "coreact_strong_weak_cfg_closed_loop_screening_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "closed_loop_screening_not_confirmation",
        "question": "Can persistent structural Strong-Weak CFG improve task success over Strong?",
        "suite": "libero_spatial",
        "tasks": list(TASKS),
        "init_state_ids": list(INIT_STATES),
        "selection": "fixed task ids 0-9 and init ids 0-9; no outcome filtering",
        "arms": list(ARMS),
        "states_per_task": len(INIT_STATES),
        "paired_states": len(TASKS) * len(INIT_STATES),
        "total_episodes": len(TASKS) * len(INIT_STATES) * len(ARMS),
        "branch_contract": {
            "W3": "last 1 action-expert residual multiplied by 0.5",
            "W4": "last 2 action-expert residuals multiplied by 0.5",
            "W4_only": "integrate W4 velocity directly at every flow step",
            "W3_CFG": "v_strong + 0.25 * (v_strong - v_W3)",
            "W4_CFG": "v_strong + 0.25 * (v_strong - v_W4)",
            "guidance_lambda": GUIDANCE_LAMBDA,
            "all_model_action_dimensions_guided": True,
            "trust_region_clipping": False,
            "ev_gate": False,
            "consensus": False,
        },
        "runtime": {
            "persistent": "every replan and every one of 10 flow steps",
            "chunk_size": 50,
            "executed_actions_per_chunk": 10,
            "maximum_control_steps": 280,
            "same_reset_and_noise_seed_within_pair": True,
            "arm_native_observations_after_first_action": True,
            "no_lambda_tuning": True,
            "seed_rules": {
                "reset": "410000000 + task*1000 + init*10",
                "flow_noise_base": "420000000 + task*100000 + init*1000; add replan index",
            },
        },
        "analysis": {
            "primary": [
                "success rate by task and arm",
                "paired overall success difference versus Vanilla",
                "positive/tied/negative task counts versus Vanilla",
                "W4-only minus Vanilla",
            ],
            "bootstrap_replicates": 10000,
            "bootstrap_unit": "paired init state",
            "paired_test": "exact McNemar",
            "promising_rule": {
                "best_cfg_overall_gain_min": 0.05,
                "best_cfg_positive_tasks_min": 6,
                "best_cfg_negative_tasks_max": 2,
                "W4_only_macro_below_vanilla": True,
            },
            "screening_only": True,
            "automatic_confirmation_forbidden": True,
        },
        "prohibited_after_lock": [
            "lambda tuning",
            "EV gating",
            "task-specific arm rules",
            "dropping tasks or states",
            "changing W3/W4 residual scales",
        ],
        "checkpoint": {
            "repo": "lerobot/smolvla_libero",
            "revision": CHECKPOINT_REVISION,
            "config_sha256": sha256(checkpoint / "config.json"),
            "weights_sha256": sha256(checkpoint / "model.safetensors"),
        },
        "implementation_sha256": {
            str(path.relative_to(workspace)): sha256(path) for path in code_paths
        },
    }
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    manifest = []
    for task in TASKS:
        for init_state in INIT_STATES:
            pair_id = f"task{task:02d}__init{init_state:02d}"
            reset_seed = 410000000 + task * 1000 + init_state * 10
            noise_seed = 420000000 + task * 100000 + init_state * 1000
            if not 0 <= reset_seed < 2**32 or not 0 <= noise_seed < 2**32:
                raise RuntimeError("seed exceeds uint32 range")
            for arm in ARMS:
                manifest.append(
                    {
                        "episode_id": f"{pair_id}__{arm}",
                        "pair_id": pair_id,
                        "suite": "libero_spatial",
                        "task_id": task,
                        "init_state_id": init_state,
                        "language": task_manifest[str(task)]["language"],
                        "reset_seed": reset_seed,
                        "action_noise_seed": noise_seed,
                        "arm": arm,
                    }
                )
    (artifact / "episode_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    )
    (artifact / "environment.json").write_text(
        json.dumps({"hostname": platform.node(), "workspace": str(workspace)}, indent=2) + "\n"
    )
    (artifact / "status/protocol.locked").write_text("locked before dry-run and rollout\n")
    print(json.dumps({"artifact": str(artifact), "episodes": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
