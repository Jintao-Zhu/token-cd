"""Local Feature-Inpainting CD (LF-CD): 3-arm paired rollout.

Arms (all share the exact initial state/RGB per seed via capture-once-reuse):
    vanilla           — clean OpenVLA (argmax)
    oracle_attn_cd    — Oracle GT-mask Attention-CD (tau=0.5, lambda=0.5, L8-15,
                        mask -1e4) = the existing oracle_cd arm, reproduced
                        IN-PROCESS so the pairing guarantee holds.
    local_replace_cd  — same G_Oracle, but the negative branch is produced by
                        overwriting target visual tokens' projector features with
                        the surrounding background-ring mean (feature inpainting)
                        instead of blocking attention.

lambda, seed, mask, tau and per-step coverage are identical between the two CD
arms, so the success/harm delta isolates *how the negative branch is built*
(attention blocking vs feature inpainting).

Run (per task; seed-shard across processes/GPUs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/local_replace_rollout.py \
      --artifact artifacts/attn_local_replace_cd_v1 \
      --task google_robot_move_near --gpu 1 --seeds 100-119
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
from research.semantic_token_cd.local_replace_policy import LocalReplaceCDInference
from research.semantic_token_cd.oracle_mask_policy import OracleAttentionCDInference
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)

PROTOCOL = "ATTN_LOCAL_REPLACE_CD_V1"
TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
ARMS = ("vanilla", "oracle_attn_cd", "local_replace_cd")
LAMBDA = 0.5
LAYER_START, LAYER_END = 8, 16
ORACLE_TAU = 0.5
MIN_RING = 8
RING_MAX_RADIUS = 4
CAMERA_NAME = "overhead_camera"
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


def get_entity_actor_ids(env, task: str) -> list[list[int]]:
    """Per-entity GT target actor ids (source and target separated)."""
    u = env.unwrapped
    if task == "google_robot_move_near":
        return [[int(u.episode_source_obj.id)], [int(u.episode_target_obj.id)]]
    if task == "google_robot_close_drawer":
        name = f"{u.drawer_id}_drawer"
        for link in u.art_obj.get_links():
            if link.name == name:
                return [[int(link.id)]]
        raise RuntimeError(f"drawer link '{name}' not found in cabinet articulation")
    if task == "google_robot_pick_coke_can":
        return [[int(u.obj.id)]]
    raise ValueError(f"No oracle target mapping for {task}")


def mask_to_overlap(seg: np.ndarray, target_ids: list[int]) -> np.ndarray:
    """GT Segmentation (H,W,4) -> per-patch overlap o_i in [0,1] over the 256 grid."""
    actor_seg = seg[..., 1]
    mask = np.isin(actor_seg, target_ids).astype(np.float32)
    import cv2
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

    attn = copy.copy(base)
    attn.__class__ = OracleAttentionCDInference
    attn.oracle_mode = "full"
    attn._oracle_tau = ORACLE_TAU
    attn._oracle_overlap = None
    attn.alpha = LAMBDA
    attn.lambd = LAMBDA
    attn.attention_layer_start = LAYER_START
    attn.attention_layer_end = LAYER_END
    attn.attention_mask_value = -1e4
    attn._selector_instr = None
    attn._entities = []
    attn._entity_emb = []
    attn._emb_cache = {}
    attn._episode_trace = []
    attn._episode_logits = []
    attn._episode_seed = 0
    attn._selector_step = 0
    policies["oracle_attn_cd"] = attn

    lr = copy.copy(base)
    lr.__class__ = LocalReplaceCDInference
    lr._oracle_tau = ORACLE_TAU
    lr._oracle_overlap = None
    lr._oracle_entity_overlaps = None
    lr._min_ring = MIN_RING
    lr._ring_max_radius = RING_MAX_RADIUS
    lr.alpha = LAMBDA
    lr.lambd = LAMBDA
    lr._selector_instr = None
    lr._entities = []
    lr._entity_emb = []
    lr._emb_cache = {}
    lr._episode_trace = []
    lr._episode_logits = []
    lr._episode_seed = 0
    lr._selector_step = 0
    policies["local_replace_cd"] = lr

    return policies


def audit_attn_trace(trace: list[dict], expected_calls: int) -> dict:
    if not trace:
        raise RuntimeError("oracle_attn_cd arm produced no trace")
    cd_steps = [s for s in trace if not s.get("oracle_empty", False)]
    feature_equal = all(s["feature_equal"] for s in cd_steps)
    hook_pass = all(s["attention_mask"]["hook_calls"] == expected_calls for s in cd_steps)
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_attention_hook_audits_pass": hook_pass,
        "technical_pass": feature_equal and hook_pass,
        "n_cd_steps": len(cd_steps),
        "n_oracle_empty_steps": len(trace) - len(cd_steps),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"oracle_attn_cd technical audit failed: {audit}")
    return audit


def audit_local_replace_trace(trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError("local_replace_cd arm produced no trace")
    cd_steps = [s for s in trace if not s.get("oracle_empty", False)]
    feature_equal = all(s.get("feature_equal", False) for s in cd_steps)
    ring_ok = all(s.get("n_ring_total", 0) > 0 for s in cd_steps)
    finite = all(np.isfinite(s.get("residual_norm", 0.0)) for s in cd_steps)
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_steps_have_nonempty_ring": ring_ok,
        "all_residual_norms_finite": finite,
        "technical_pass": feature_equal and ring_ok and finite,
        "n_cd_steps": len(cd_steps),
        "n_oracle_empty_steps": len(trace) - len(cd_steps),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"local_replace_cd technical audit failed: {audit}")
    return audit


def local_replace_metrics(trace: list[dict]) -> dict:
    cd = [s for s in trace if not s.get("oracle_empty", False)]
    if not cd:
        return {"n_local_replace_steps": 0}
    norms = [float(s["residual_norm"]) for s in cd]
    mu_ratios = [p["mu_norm_ratio"] for s in cd for p in s.get("per_entity", []) if "mu_norm_ratio" in p]
    s_before = [p["semantic_sim_before"] for s in cd for p in s.get("per_entity", []) if "semantic_sim_before" in p]
    s_after = [p["semantic_sim_after"] for s in cd for p in s.get("per_entity", []) if "semantic_sim_after" in p]
    n_target = [int(s["n_target_tokens"]) for s in cd]
    return {
        "n_local_replace_steps": len(cd),
        "mean_residual_norm": float(np.mean(norms)),
        "mean_mu_norm_ratio": float(np.mean(mu_ratios)) if mu_ratios else None,
        "mean_semantic_sim_before": float(np.mean(s_before)) if s_before else None,
        "mean_semantic_sim_after": float(np.mean(s_after)) if s_after else None,
        "mean_n_target_tokens": float(np.mean(n_target)),
    }


def run_episode_local_replace(env, policy, instruction, obs, entity_groups=None, target_ids=None):
    from research.semantic_token_cd.spatial_grid_rollout import run_episode as _run

    if not isinstance(policy, (OracleAttentionCDInference, LocalReplaceCDInference)):
        return _run(env, policy, instruction, obs)
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize

    predicted_terminated = truncated = False
    timestep = 0
    step_infos = []
    executed_actions = []
    image = _get_image(env, obs)
    while not (predicted_terminated or truncated):
        seg = obs["image"][CAMERA_NAME]["Segmentation"]
        if isinstance(policy, LocalReplaceCDInference):
            policy._oracle_entity_overlaps = [mask_to_overlap(seg, g) for g in entity_groups]
            policy._oracle_overlap = mask_to_overlap(seg, target_ids)
        else:
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
    return obs["image"][CAMERA_NAME]["rgb"]


def _flatten(action):
    from research.semantic_token_cd.rollout_pilot import flatten_action
    return flatten_action(action)


def config_lock(task_root: Path, task: str) -> None:
    lock = {
        "experiment": PROTOCOL,
        "benchmark": "SIMPLER Local Feature-Inpainting CD (LF-CD) vs Oracle Attention-CD",
        "task": task,
        "seeds": list(range(SEED_START, SEED_START + N_SEEDS)),
        "arms": list(ARMS),
        "lambda": LAMBDA,
        "attention_layers": [LAYER_START, LAYER_END],
        "attention_mask_value_requested": -10000.0,
        "oracle": {
            "source": "simulator instance Segmentation (prepackaged add_segmentation)",
            "tau": ORACLE_TAU,
            "overlap": "o_i = |patch_i ∩ M| / |patch_i| in the 224x224 model view",
            "full": "G = {i : o_i > tau}, per entity (source/target separated)",
        },
        "local_replace": {
            "operator": "v_i <- (1/|R(G)|) Σ_{j∈R(G)} v_j for i∈G, at projector output",
            "ring": "R(G) = Dilate(G, r) \\ G \\ all_target; r grows until |R|>=8 (max 4)",
            "per_entity": "each entity's ring excludes every other entity's target region",
            "min_ring": MIN_RING,
            "ring_max_radius": RING_MAX_RADIUS,
            "cd_tokens": "action dims 0..5, gripper dim 6 keeps clean",
            "empty_fallback": "no patch>tau or empty ring -> clean action (CD off)",
        },
        "pairing": "capture-once-reuse; all 3 arms share the exact initial state/RGB per seed",
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
        entity_groups: list[list[int]] | None = None
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
            if entity_groups is None:
                entity_groups = get_entity_actor_ids(env, args.task)
                target_ids = [i for g in entity_groups for i in g]
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode_local_replace(
                env, policy, instruction, obs, entity_groups, target_ids
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            if arm == "vanilla":
                audit = None
                lr_metrics = None
            elif arm == "oracle_attn_cd":
                audit = audit_attn_trace(trace, expected_hook_calls)
                lr_metrics = None
            else:
                audit = audit_local_replace_trace(trace)
                lr_metrics = local_replace_metrics(trace)
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
                "attention_layers": None if arm != "oracle_attn_cd" else [LAYER_START, LAYER_END],
                "oracle_tau": None if arm == "vanilla" else ORACLE_TAU,
                "oracle_mode": None if arm == "vanilla" else (
                    "full" if arm == "oracle_attn_cd" else "local_replace"
                ),
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
            }
            if audit is not None:
                summary.update(audit)
                summary["oracle_mean_num_tokens"] = float(
                    np.mean([s.get("num_tokens", s.get("n_target_tokens", 0))
                             for s in trace if not s.get("oracle_empty", False)])
                )
                summary["oracle_empty_rate"] = float(
                    np.mean([1.0 if s.get("oracle_empty", False) else 0.0 for s in trace])
                )
            if lr_metrics is not None:
                summary["local_replace"] = lr_metrics
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
        pairs.append({"seed": seed, "canonical_snapshot_sha256": canonical, "exact_three_arm_pairing": True})

    manifest_path = task_root / f"pairing_manifest_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps({
        "task": args.task, "seeds": seeds, "arms": list(ARMS), "pairs": pairs,
        "all_three_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
