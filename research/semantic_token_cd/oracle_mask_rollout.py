"""Oracle Semantic-Mask test: 4-arm paired rollout.

Arms (all share the exact initial state/RGB per seed via capture-once-reuse):
    vanilla            — clean OpenVLA (argmax)
    kmeans_cd          — L8-15 semantic_hard Attention-CD (K=8, lambda=0.5) [= the
                         existing layer-window L8-15 arm; reproduced for pairing]
    oracle_cd          — same Attention-CD, but G_KMeans -> G_Oracle (tau=0.5)
    oracle_budget_cd   — same Attention-CD, but G = Top-B GT-overlap patches,
                         B = |G_KMeans| per step (budget-matched)

Everything else (layers [8,16), lambda=0.5, mask -1e4, CD formula) is identical
across the three CD arms. The GT object mask comes from the simulator's
instance Segmentation (already enabled via prepackaged add_segmentation); the
target actor ids are read directly from the env per task.

Run (per task; seed-shard across processes/GPUs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/oracle_mask_rollout.py \
      --artifact artifacts/attn_semantic_oracle_mask_v1 \
      --task google_robot_move_near --gpu 1 --seeds 100-112
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np

from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.oracle_mask_policy import OracleAttentionCDInference
from research.semantic_token_cd.spatial_grid_policy import SpatialGridAttentionCDInference
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)

PROTOCOL = "ATTN_SEMANTIC_ORACLE_MASK_V1"
TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
ARMS = ("vanilla", "kmeans_cd", "oracle_cd", "oracle_budget_cd")
KMEANS_K = 8
LAMBDA = 0.5
LAYER_START, LAYER_END = 8, 16
ORACLE_TAU = 0.5
CAMERA_NAME = "overhead_camera"  # google_robot policy camera
SEED_START = 100
N_SEEDS = 100


def parse_seeds(spec: str) -> list[int]:
    if not spec.strip():
        return list(range(SEED_START, SEED_START + N_SEEDS))
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


def get_target_actor_ids(env, task: str) -> list[int]:
    """GT target actor ids, read from the env's episode object handles."""
    u = env.unwrapped
    if task == "google_robot_move_near":
        return [int(u.episode_source_obj.id), int(u.episode_target_obj.id)]
    if task == "google_robot_close_drawer":
        # NOTE: u.drawer_obj can point at the wrong link on a freshly-loaded scene
        # (it resolved to middle_drawer even when drawer_id == "top", leaving the
        # target id absent from the overhead Segmentation -> empty oracle mask).
        # drawer_id is deterministic and correct; resolve the link by name instead.
        name = f"{u.drawer_id}_drawer"
        for link in u.art_obj.get_links():
            if link.name == name:
                return [int(link.id)]
        raise RuntimeError(f"drawer link '{name}' not found in cabinet articulation")
    if task == "google_robot_pick_coke_can":
        return [int(u.obj.id)]
    raise ValueError(f"No oracle target mapping for {task}")


def mask_to_overlap(seg: np.ndarray, target_ids: list[int]) -> np.ndarray:
    """GT Segmentation (H,W,4) -> per-patch overlap o_i in [0,1] over the 256 grid.

    Matches the policy's own preprocessing: the RGB is INTER_AREA-resized to
    224x224, then SigLIP splits it into a 16x16 grid (patch 14). We resize the
    binary target mask to 224 with the same interpolation and average each
    14x14 block, giving o_i = |P_i ∩ M| / |P_i| exactly in the model's view.
    """
    actor_seg = seg[..., 1]
    mask = np.isin(actor_seg, target_ids).astype(np.float32)
    m224 = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_AREA)
    overlap = m224.reshape(16, 14, 16, 14).mean(axis=(1, 3)).reshape(-1)
    return overlap.astype(np.float32)


def build_policies(base):
    policies: dict[str, object] = {}
    vanilla = copy.copy(base)
    vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace = []
    vanilla._episode_logits = []
    policies["vanilla"] = vanilla

    for arm in ("kmeans_cd", "oracle_cd", "oracle_budget_cd"):
        policy = copy.copy(base)
        if arm == "kmeans_cd":
            policy.__class__ = SpatialGridAttentionCDInference
            policy.spatial_selection_mode = "semantic_hard"
        else:
            policy.__class__ = OracleAttentionCDInference
            policy.oracle_mode = "full" if arm == "oracle_cd" else "budget"
            policy._oracle_tau = ORACLE_TAU
            policy._oracle_overlap = None
        policy.alpha = LAMBDA
        policy.lambd = LAMBDA
        policy.kmeans_K = KMEANS_K
        policy.kmeans_seed = 0
        policy.selection_mode = "semantic"
        policy.attention_layer_start = LAYER_START
        policy.attention_layer_end = LAYER_END
        policy.attention_mask_value = -1e4
        policy._selector_instr = None
        policy._entities = []
        policy._entity_emb = []
        policy._emb_cache = {}
        policy._episode_trace = []
        policy._episode_logits = []
        policy._episode_seed = 0
        policy._selector_step = 0
        policies[arm] = policy
    return policies


def audit_cd_trace(trace: list[dict], expected_calls: int) -> dict:
    if not trace:
        raise RuntimeError("CD arm produced no trace")
    cd_steps = [s for s in trace if not s.get("oracle_empty", False)]
    degenerate = len(cd_steps) == 0
    feature_equal = all(s["feature_equal"] for s in cd_steps)
    hook_pass = all(s["attention_mask"]["hook_calls"] == expected_calls for s in cd_steps)
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_attention_hook_audits_pass": hook_pass,
        "technical_pass": feature_equal and hook_pass,
        "n_cd_steps": len(cd_steps),
        "n_oracle_empty_steps": len(trace) - len(cd_steps),
        "oracle_degenerate": degenerate,
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"Technical audit failed: {audit}")
    return audit


def run_episode_oracle(env, policy, instruction, obs, target_ids=None):
    """run_episode with per-step oracle-overlap injection."""
    from research.semantic_token_cd.spatial_grid_rollout import run_episode as _run

    if not isinstance(policy, OracleAttentionCDInference):
        return _run(env, policy, instruction, obs)
    # Reimplement the closed-loop with mask injection (oracle arms only).
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize

    predicted_terminated = truncated = False
    timestep = 0
    step_infos = []
    executed_actions = []
    image = _get_image(env, obs)
    while not (predicted_terminated or truncated):
        seg = obs["image"][CAMERA_NAME]["Segmentation"]
        policy._oracle_overlap = mask_to_overlap(seg, target_ids)
        _raw_action, actions, _aux = policy.step(
            image, None, instruction, proprio=obs["agent"]["eef_pos"]
        )
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            executed = _flatten(action)
            if executed.shape != (7,) or not np.isfinite(executed).all():
                raise FloatingPointError(f"Invalid executed action: {executed}")
            executed_actions.append(executed.copy())
            obs, _reward, _success, truncated, info = env.step(executed)
            image = _get_image(env, obs)
            timestep += 1
            step_infos.append(convert_numpy_or_torch_to_python(info))
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not env.unwrapped.is_final_subtask():
                predicted_terminated = False
                env.advance_to_next_subtask()
            instruction = env.unwrapped.get_language_instruction()
    result = summarize(step_infos)
    result.update(stat_first(step_infos))
    result.update(stat_final(step_infos))
    failure_reason = None if result["success"] else (
        "environment_time_limit" if truncated else "policy_terminated_without_success"
    )
    actions_array = np.asarray(executed_actions, dtype=np.float32)
    jitter = float(np.linalg.norm(np.diff(actions_array, axis=0), axis=1).mean()) if len(actions_array) > 1 else 0.0
    return result, timestep, failure_reason, actions_array, jitter


def _get_image(env, obs):
    # google_robot policy camera (matches parallel_inference.get_image_from_maniskill2_obs_dict)
    return obs["image"][CAMERA_NAME]["rgb"]


def _flatten(action):
    from research.semantic_token_cd.rollout_pilot import flatten_action
    return flatten_action(action)


def config_lock(task_root: Path, task: str) -> None:
    lock = {
        "experiment": PROTOCOL,
        "benchmark": "SIMPLER Attention-CD oracle semantic-mask test (K8 lambda0.5 L8-15)",
        "task": task,
        "seeds": list(range(SEED_START, SEED_START + N_SEEDS)),
        "arms": list(ARMS),
        "kmeans_k": KMEANS_K,
        "lambda": LAMBDA,
        "attention_layers": [LAYER_START, LAYER_END],
        "attention_mask_value_requested": -10000.0,
        "oracle": {
            "source": "simulator instance Segmentation (prepackaged add_segmentation)",
            "target_actor_ids": "read from env object handles per task",
            "tau": ORACLE_TAU,
            "overlap": "o_i = |patch_i ∩ M| / |patch_i| in the 224x224 model view",
            "full": "G = {i : o_i > tau}",
            "budget": "G = Top-B patches by o_i, B = |G_KMeans| per step",
            "empty_fallback": "full-mode step with no patch > tau -> clean action (CD off for that step)",
        },
        "pairing": "capture-once-reuse; all 4 arms share the exact initial state/RGB per seed",
    }
    path = task_root / "CONFIG_LOCK.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing != lock:
            raise RuntimeError(f"CONFIG_LOCK differs from requested run: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="", help="comma/range; empty = 100..199")
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
    config_lock(task_root, args.task)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**policy_config))
    expected_hook_calls = 7 * (LAYER_END - LAYER_START)

    pairs = []
    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        summaries: dict[str, dict] = {}
        initial_hashes: dict[str, tuple] = {}
        target_ids: list[int] | None = None
        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                summaries[arm] = json.loads(summary_path.read_text())
                print(json.dumps({"skip": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial_hashes[arm] = (state_sha, rgb_sha)
            if target_ids is None:
                target_ids = get_target_actor_ids(env, args.task)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode_oracle(
                env, policy, instruction, obs, target_ids
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            audit = None if arm == "vanilla" else audit_cd_trace(trace, expected_hook_calls)
            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": args.task,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": jitter,
                "lambda": 0.0 if arm == "vanilla" else LAMBDA,
                "kmeans_k": None if arm == "vanilla" else KMEANS_K,
                "attention_layers": None if arm == "vanilla" else [LAYER_START, LAYER_END],
                "oracle_tau": None if arm not in ("oracle_cd", "oracle_budget_cd") else ORACLE_TAU,
                "oracle_mode": None if not arm.startswith("oracle") else ("full" if arm == "oracle_cd" else "budget"),
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
            }
            if audit is not None:
                summary.update(audit)
                if arm.startswith("oracle"):
                    summary["oracle_mean_num_tokens"] = float(
                        np.mean([s.get("num_tokens", 0) for s in trace if not s.get("oracle_empty", False)])
                    )
                    summary["oracle_empty_rate"] = float(
                        np.mean([1.0 if s.get("oracle_empty", False) else 0.0 for s in trace])
                    )
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            summaries[arm] = summary
            print(json.dumps({
                "task": args.task, "seed": seed, "arm": arm,
                "success": summary["success"], "steps": steps,
                "technical_pass": audit is None or audit["technical_pass"],
            }, sort_keys=True), flush=True)

        for arm in ARMS:
            if arm not in initial_hashes:
                initial_hashes[arm] = (
                    summaries[arm]["initial_state_sha256"],
                    summaries[arm]["initial_rgb_sha256"],
                )
        if len({v[0] for v in initial_hashes.values()}) != 1 or len({v[1] for v in initial_hashes.values()}) != 1:
            raise RuntimeError(f"Cross-arm initial-state mismatch for {args.task} seed {seed}")
        if len({s["canonical_snapshot_sha256"] for s in summaries.values()}) != 1:
            raise RuntimeError(f"Cross-arm snapshot mismatch for {args.task} seed {seed}")
        pairs.append({"seed": seed, "canonical_snapshot_sha256": canonical, "exact_four_arm_pairing": True})

    manifest_path = task_root / f"pairing_manifest_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps({
        "task": args.task, "seeds": seeds, "arms": list(ARMS), "pairs": pairs,
        "all_four_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
