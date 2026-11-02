"""Paired Pi0 rollout for guidance-form variants (Phase 1 of the repair search).

Same simulator, same snapshots, same noise, same semantic negative branch as the
locked PI0_SHR_5TASK_0_299_V1 protocol; only the way the positive/negative
residual is injected changes.  See `pi0_guidance_variants.py` for why.

New artifact root per protocol so the locked run stays untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.pi0_guidance_variants import (
    GuidanceSpec,
    Pi0VariantInference,
)
from research.semantic_token_cd.pi0_shr_rollout import (
    CONFIG_BY_TASK,
    TASKS,
    capture_snapshot,
    load_model,
    noise_seed,
    parse_seeds,
    restore_snapshot,
    run_episode,
    snapshot_sha,
    write_trace,
)

PROTOCOL = "PI0_GUIDANCE_FORM_5TASK_V1"

# arm -> guidance spec.  `pi0_vanilla` is the paired no-guidance reference.
VARIANTS: dict[str, GuidanceSpec | None] = {
    "pi0_vanilla": None,
    "pi0_dense_l050": GuidanceSpec(lambd=0.5),          # in-run control
    # dense magnitude sweep: is the continuous injection simply too weak?
    "pi0_dense_l150": GuidanceSpec(lambd=1.5),
    "pi0_dense_l300": GuidanceSpec(lambd=3.0),
    # structural sparsity: touch only the strongest dimensions
    "pi0_top2_l050": GuidanceSpec(lambd=0.5, topk_dims=2),
    # sparse and large together
    "pi0_top2_l150": GuidanceSpec(lambd=1.5, topk_dims=2),
    # step-schedule control
    "pi0_last3_l050": GuidanceSpec(lambd=0.5, active_steps=(7, 8, 9)),
    # decisiveness gate: act only where the counterfactual disagrees by a
    # meaningful fraction of the clean velocity (tau=0.15 activates ~24% of cells)
    "pi0_relgate015_l050": GuidanceSpec(lambd=0.5, rel_gate=0.15),
}

ALL_ARMS = tuple(VARIANTS)


def ensure_config(artifact: Path, arms: tuple[str, ...]) -> None:
    config = {
        "protocol_id": PROTOCOL,
        "tasks": list(TASKS),
        "arms": list(arms),
        "pairing": "same simulator snapshot; deterministic shared initial action noise per replan",
        "model": "open-pi-zero Fractal-Beta for Google Robot and Bridge-Beta for WidowX",
        "positive_branch": "unmodified Pi0 flow velocity",
        "negative_branch": "KMeans K=8 semantic entity regions; beta=0 four-neighbor harmonic reconstruction",
        "gripper": "positive velocity unchanged at every flow step",
        "flow_steps": 10,
        "torch_compile": False,
        "guidance_specs": {
            arm: ("no guidance" if spec is None else spec.describe())
            for arm, spec in VARIANTS.items()
        },
        "reference_protocol": "PI0_SHR_5TASK_0_299_V1",
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"CONFIG_LOCK differs: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument(
        "--arms", default=",".join(ALL_ARMS),
        help="comma-separated arm names (default: all)",
    )
    args = parser.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for arm in arms:
        if arm not in VARIANTS:
            raise SystemExit(f"unknown arm {arm!r}; known: {ALL_ARMS}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.gpu):
        raise RuntimeError(
            f"launcher must set CUDA_VISIBLE_DEVICES={args.gpu}; found {visible!r}"
        )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    ensure_config(artifact, arms)

    import simpler_env

    env = simpler_env.make(args.task)
    model_config, model, adapter, checkpoint = load_model(args.task, device)
    policies = {}
    for arm in arms:
        spec = VARIANTS[arm]
        policies[arm] = (
            Pi0VariantInference(model, adapter.tokenizer, spec=spec)
            if spec is not None
            else Pi0VariantInference(model, adapter.tokenizer, spec=GuidanceSpec(lambd=0.0))
        )

    task_root = artifact / "episodes" / args.task
    snapshot_root = artifact / "snapshots" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    for seed in parse_seeds(args.seeds):
        snapshot_path = snapshot_root / f"seed_{seed:03d}.pkl"
        if snapshot_path.exists():
            with snapshot_path.open("rb") as handle:
                snapshot = pickle.load(handle)
        else:
            snapshot = capture_snapshot(env, seed)
            temporary = snapshot_path.with_name(f".{snapshot_path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, snapshot_path)
        canonical = snapshot_sha(snapshot)
        arm_hashes = {}
        for arm in arms:
            arm_root = task_root / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            summary_path = arm_root / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_root / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                summary = json.loads(summary_path.read_text())
                arm_hashes[arm] = (
                    summary["canonical_snapshot_sha256"],
                    summary["initial_state_sha256"],
                    summary["initial_rgb_sha256"],
                )
                continue
            observation, state_hash, rgb_hash = restore_snapshot(env, adapter, seed, snapshot)
            instruction = env.get_language_instruction()
            started = time.monotonic()
            success, steps, traces, actions, last_info = run_episode(
                env, adapter, policies[arm], args.task, seed, arm, observation,
                instruction, device
            )
            runtime = time.monotonic() - started
            write_trace(arrays_path, traces, actions)
            technical_pass = bool(traces) and all(
                trace["shared_action_state_between_branches"]
                and trace["all_gripper_velocity_positive_unchanged"]
                and trace["reconstruction_finite"]
                for trace in traces
            )
            if not technical_pass:
                raise RuntimeError(f"Pi0 variant audit failed for {args.task} seed={seed} arm={arm}")
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "seed": seed,
                "evaluation_seed": seed,
                "arm": arm,
                "success": success,
                "control_steps": steps,
                "replans": len(traces),
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "checkpoint": str(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "config": CONFIG_BY_TASK[args.task],
                "act_steps": int(model_config.act_steps),
                "lambda": 0.0 if arm == "pi0_vanilla" else VARIANTS[arm].lambd,
                "guidance_spec": (
                    "no guidance" if arm == "pi0_vanilla" else VARIANTS[arm].describe()
                ),
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_hash,
                "initial_rgb_sha256": rgb_hash,
                "arrays_file": arrays_path.name,
                "noise_seeds": [trace["noise_seed"] for trace in traces],
                "mean_selected_tokens": None if arm == "pi0_vanilla" else float(np.mean([
                    trace["selector"]["num_tokens"] for trace in traces
                ])),
                "mean_residual_norm": float(np.mean([
                    flow["residual_norm"] for trace in traces for flow in trace["flow_steps"]
                ])),
                "active_flow_steps_per_replan": None if arm == "pi0_vanilla" else int(np.mean([
                    sum(1 for flow in trace["flow_steps"] if flow["guidance_active"])
                    for trace in traces
                ])),
                "selector_trace": [trace["selector"] for trace in traces],
                "flow_residual_norms": [
                    [flow["residual_norm"] for flow in trace["flow_steps"]]
                    for trace in traces
                ],
                "all_gripper_velocity_positive_unchanged": True,
                "shared_action_state_between_branches": True,
                "technical_pass": True,
                "last_info": last_info,
            }
            temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, summary_path)
            arm_hashes[arm] = (canonical, state_hash, rgb_hash)
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": success,
                "steps": steps,
                "runtime_seconds": round(runtime, 2),
            }), flush=True)
        if len(set(arm_hashes.values())) != 1:
            raise RuntimeError(f"Pi0 cross-arm pairing mismatch: {args.task} seed={seed}")


if __name__ == "__main__":
    main()
