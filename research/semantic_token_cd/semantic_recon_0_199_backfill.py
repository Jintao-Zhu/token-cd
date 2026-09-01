"""SCR-CD Phase 2 (replication): backfill vanilla + semantic_attn for move_near 0-99.

move_near had NO vanilla/attn baseline in the 0-99 range (only 100-199 existed in
``attn_semantic_layer_window_sweep_v1``). This script fills that gap so the 3-way
comparison (vanilla / semantic_attn_k8_l8_15 / semantic_recon_k8_m10) can be done
for move_near split into 0-99 and 100-199 blocks.

Two arms, capture-once-reuse (both restore the SAME canonical snapshot per seed),
hash-verified fail-closed. Config is identical to Phase 1:
    vanilla                    — clean OpenVLA.
    semantic_attn_k8_l8_15     — block Action Query -> selected visual keys, L8-15,
                                 KMeans K=8 semantic selector, mask -1e4, lambda 0.5.

Run:
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/semantic_recon_0_199_backfill.py \
      --artifact artifacts/semantic_recon_0_199_core3_replication_v1 \
      --gpu 1 --seeds 0-19
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
from research.semantic_token_cd.semantic_merge_policy import (
    GuidedSemanticAttentionCDInference,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)
from research.semantic_token_cd.semantic_recon_rollout import (
    audit_attn_trace,
    _init_common,
)

PROTOCOL = "SCR_CD_SEMANTIC_RECON_K8_M10_0_199_CORE3_REPLICATION_V1"
TASK = "google_robot_move_near"
ARMS = ("vanilla", "semantic_attn_k8_l8_15")
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0
LAYER_START, LAYER_END = 8, 16
MASK_VALUE = -1e4


def parse_seeds(spec: str) -> list[int]:
    if not spec.strip():
        return list(range(0, 100))
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


def build_policies(base):
    vanilla = copy.copy(base)
    vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace = []
    vanilla._episode_logits = []

    attn = copy.copy(base)
    attn.__class__ = GuidedSemanticAttentionCDInference
    _init_common(attn, LAMBDA)
    attn.attention_layer_start = LAYER_START
    attn.attention_layer_end = LAYER_END
    attn.attention_mask_value = MASK_VALUE
    return {"vanilla": vanilla, "semantic_attn_k8_l8_15": attn}


def config_lock(task_root: Path) -> None:
    lock = {
        "experiment": PROTOCOL,
        "phase": "2 replication backfill: move_near 0-99 vanilla + attn",
        "task": TASK,
        "seeds": [0, 99],
        "arms": list(ARMS),
        "lambda": LAMBDA,
        "kmeans": {"K": KMEANS_K, "seed": KMEANS_SEED, "n_init": 10},
        "attention": {"layers": [LAYER_START, LAYER_END], "mask_value": MASK_VALUE},
        "cd_tokens": "action dims 0..5, gripper dim 6 clean",
        "pairing": "capture-once-reuse; hash-verified fail-closed across 2 arms",
        "note": "attn arm == semantic_attn_k8_l8_15 (K8, L8-15, lambda0.5, semantic_hard)",
    }
    path = task_root / "CONFIG_LOCK_backfill.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != lock:
            raise RuntimeError(f"CONFIG_LOCK_backfill differs from requested run: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="", help="comma/range; empty = 0..99")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    seeds = parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    task_root = artifact / "episodes" / TASK
    task_root.mkdir(parents=True, exist_ok=True)
    config_lock(task_root)

    env, environment_id = make_environment(TASK)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, TASK, {}, False)
    policies = build_policies(OpenVLAInference(**policy_config))

    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        summaries: dict[str, dict] = {}
        initial_hashes: dict[str, tuple] = {}
        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                summaries[arm] = json.loads(summary_path.read_text())
                print(json.dumps({"skip": TASK, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial_hashes[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(
                env, policy, instruction, obs
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)

            audit = None
            if arm != "vanilla":
                audit = audit_attn_trace(trace)

            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": TASK,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": jitter,
                "lambda": 0.0 if arm == "vanilla" else LAMBDA,
                "guided_prefix": arm != "vanilla",
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
            }
            if arm == "semantic_attn_k8_l8_15":
                summary["attention_layers"] = [LAYER_START, LAYER_END]
                summary["attention_mask_value_requested"] = float(MASK_VALUE)
            if audit is not None:
                summary.update(audit)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            summaries[arm] = summary
            print(json.dumps({
                "task": TASK, "seed": seed, "arm": arm,
                "success": summary["success"], "steps": steps,
                "technical_pass": audit["technical_pass"] if audit else True,
            }, sort_keys=True), flush=True)

        for arm in ARMS:
            if arm not in initial_hashes:
                initial_hashes[arm] = (
                    summaries[arm]["initial_state_sha256"],
                    summaries[arm]["initial_rgb_sha256"],
                )
        if len({v[0] for v in initial_hashes.values()}) != 1 or len({v[1] for v in initial_hashes.values()}) != 1:
            raise RuntimeError(f"Cross-arm initial-state mismatch for {TASK} seed {seed}")
        if len({s["canonical_snapshot_sha256"] for s in summaries.values()}) != 1:
            raise RuntimeError(f"Cross-arm snapshot mismatch for {TASK} seed {seed}")

    manifest_path = task_root / f"pairing_manifest_backfill_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps({
        "task": TASK, "seeds": seeds, "arms": list(ARMS),
        "all_two_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": TASK, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
