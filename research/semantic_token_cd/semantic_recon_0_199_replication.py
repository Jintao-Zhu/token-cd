"""SCR-CD Phase 2 (replication): single-arm Semantic-Recon rollout, seeds 0-199.

NOT a new method. This is a *replication / stability* run that adds the
``semantic_recon_k8_m10`` arm over the SAME initial-state range (seeds 0-199)
where vanilla and semantic-attn already have results, so the three can be
compared per-seed (hash-verified) split into 0-99 and 100-199 blocks.

Only ONE arm is run here: semantic_recon_k8_m10. Frozen config is identical to
Phase 1 (300-399): KMeans K=8 semantic selector, M=10 deterministic-FPS context
basis, affine ridge reconstruction (rho = 1e-3 * tr(B_c^T B_c)/M), negative keeps
only hat{v}_i for i in G, 256 tokens, CD on action dims 0..5 (dim 6 clean),
lambda=0.5, greedy, shared guided prefix.

Tasks: move_near / close_drawer / pick_coke_can (the 3 core tasks).
Seeds: 0-199 (200/task), 600 episodes total.

Every episode summary records ``initial_state_sha256`` / ``initial_rgb_sha256`` /
``canonical_snapshot_sha256`` — the SAME fields the earlier 0-99 and 100-199
vanilla/attn artifacts record — so the analysis can do fail-closed per-seed
initial-state hash matching and drop any non-reproducible seed.

Run (per task; seed-shard across processes/GPUs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/semantic_recon_0_199_replication.py \
      --artifact artifacts/semantic_recon_0_199_core3_replication_v1 \
      --task google_robot_move_near --gpu 1 --seeds 0-24
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.semantic_recon_policy import (
    M_BASIS,
    RHO_SCALE,
    SemanticReconCDInference,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)
from research.semantic_token_cd.semantic_recon_rollout import (
    audit_recon_trace,
    _init_common,
)

PROTOCOL = "SCR_CD_SEMANTIC_RECON_K8_M10_0_199_CORE3_REPLICATION_V1"
TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
TASK_INDEX = {t: i for i, t in enumerate(TASKS)}
ARM = "semantic_recon_k8_m10"
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0


def parse_seeds(spec: str) -> list[int]:
    if not spec.strip():
        return list(range(0, 200))
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def build_recon_policy(base, task: str) -> SemanticReconCDInference:
    recon = copy.copy(base)
    recon.__class__ = SemanticReconCDInference
    _init_common(recon, LAMBDA)
    recon.recon_selection_mode = "semantic"
    recon.selection_mode = "semantic"
    recon.M = M_BASIS
    recon.rho_scale = RHO_SCALE
    recon._task_id = TASK_INDEX[task]
    return recon


def config_lock(task_root: Path, task: str) -> None:
    lock = {
        "experiment": PROTOCOL,
        "phase": "2 replication: single-arm semantic_recon over seeds 0-199",
        "task": task,
        "seeds": [0, 199],  # range marker (inclusive)
        "arms": [ARM],
        "lambda": LAMBDA,
        "guided_prefix": "shared clean greedy prefix; negative branch teacher-forced",
        "kmeans": {"K": KMEANS_K, "seed": KMEANS_SEED, "n_init": 10,
                   "selector": "entity_set KMeans top-1 cos match (source/target), deduped"},
        "reconstruction": {
            "M": M_BASIS,
            "rho_scale": RHO_SCALE,
            "rho": "1e-3 * tr(B_c^T B_c) / M (fixed)",
            "operator": "v_i -> mu_B + B_c alpha_i; alpha=(B_c^T B_c + rho I)^{-1} B_c^T (v_i - mu_B)",
            "basis": "M tokens FPS on cosine from C = V \\ G",
            "token_count": "256 -> 256; positions unchanged; non-selected bit-identical",
            "scope": "source AND target groups excluded from basis; once per control obs",
        },
        "cd_tokens": "action dims 0..5, gripper dim 6 clean",
        "pairing": "per-seed initial_state_sha256 match vs existing vanilla/attn artifacts",
        "note": "frozen config == Phase 1 semantic_recon_k8_m10 (K8 M10 lambda0.5)",
    }
    path = task_root / "CONFIG_LOCK.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != lock:
            raise RuntimeError(f"CONFIG_LOCK differs from requested run: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="", help="comma/range; empty = 0..199")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    seeds = parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    arm_dir = task_root / ARM
    arm_dir.mkdir(parents=True, exist_ok=True)
    config_lock(task_root, args.task)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_recon_policy(OpenVLAInference(**policy_config), args.task)

    pairs = []
    for seed in seeds:
        summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
        arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            print(json.dumps({"skip": args.task, "seed": seed, "arm": ARM}), flush=True)
            continue

        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
        write_arrays(arrays_path, policy._episode_logits, actions)
        trace = jsonable(policy._episode_trace)

        audit = audit_recon_trace(trace, "semantic")
        summary = {
            "protocol_id": PROTOCOL,
            "environment_id": environment_id,
            "task": args.task,
            "seed": seed,
            "episode_id": seed,
            "arm": ARM,
            "instruction": instruction,
            "success": bool(result["success"]),
            "failure_reason": reason,
            "control_steps": steps,
            "action_jitter_index": jitter,
            "lambda": LAMBDA,
            "guided_prefix": True,
            "kmeans_K": KMEANS_K,
            "kmeans_seed": KMEANS_SEED,
            "canonical_snapshot_sha256": canonical,
            "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha,
            "arrays_file": arrays_path.name,
            "selector_trace": trace,
            "result": jsonable(result),
            "reconstruction": {"M": M_BASIS, "rho_scale": RHO_SCALE,
                               "task_id": TASK_INDEX[args.task]},
            "mean_e_rel": float(np.mean([
                s.get("recon_e_rel", {}).get("mean", 0.0) for s in trace
            ])) if trace else 0.0,
            "mean_q_ratio": float(np.mean([
                s.get("recon_q_ratio", {}).get("mean", 0.0) for s in trace
            ])) if trace else 0.0,
            "mean_cos_v_vhat": float(np.mean([
                s.get("recon_cos", {}).get("mean", 0.0) for s in trace
            ])) if trace else 0.0,
        }
        summary.update(audit)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        pairs.append({
            "seed": seed,
            "canonical_snapshot_sha256": canonical,
            "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha,
        })
        print(json.dumps({
            "task": args.task, "seed": seed, "arm": ARM,
            "success": summary["success"], "steps": steps,
            "technical_pass": audit["technical_pass"],
        }, sort_keys=True), flush=True)

    manifest_path = task_root / f"pairing_manifest_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps({
        "task": args.task, "seeds": seeds, "arm": ARM, "pairs": pairs,
        "vanilla_source": "existing_0_199_artifacts",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
