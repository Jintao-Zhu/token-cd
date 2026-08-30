"""Semantic-Merge vs Attention-K8 CD (Phase 1B mechanism study) — 2-arm paired rollout.

Both arms share the EXACT same KMeans K=8 semantic selector (source/target entity
groups, computed on the clean projector features), differing only in *how* the
selected groups are degraded in the negative branch:

    semantic_attn_k8_l8_15    — Attention-CD: block Action Query -> selected
                                visual keys on layers [8,16), mask -1e4.
    semantic_merge_k8_eta100  — Merge-CD: collapse each selected group to its own
                                prototype (v_i -> mu_G, eta=1.0); source/target
                                groups merged separately; attention untouched.

Both use the guided autoregressive prefix (negative branch teacher-forced on the
clean branch's greedy tokens) and CD on action dims 0..5 (dim 6 clean), lambda=0.5.

Vanilla is NOT re-run here: it already exists in the Phase 1A artifact
`artifacts/attn_global_merge_v1/episodes/<task>/vanilla/`. Each episode's
canonical/state/RGB hashes are written into the summaries and the analysis script
verifies them fail-closed against the stored vanilla hashes (the capture_snapshot
determinism was already proven across processes for seeds 200-299).

Run (per task; seed-shard across processes/GPUs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/semantic_merge_rollout.py \
      --artifact artifacts/attn_semantic_merge_k8_v1 \
      --task google_robot_close_drawer --gpu 1 --seeds 200-219
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.semantic_merge_policy import (
    GuidedSemanticAttentionCDInference,
    SemanticMergeCDInference,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    write_arrays,
)

PROTOCOL = "ATTN_SEMANTIC_MERGE_K8_V1"
TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
ARMS = ("semantic_attn_k8_l8_15", "semantic_merge_k8_eta100")
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0
LAYER_START, LAYER_END = 8, 16
MASK_VALUE = -1e4
ETA = 1.0
CAMERA_NAME = "overhead_camera"
SEED_START = 200
N_SEEDS = 100
N_ATTENTION_LAYERS = LAYER_END - LAYER_START
VANILLA_ARTIFACT = "attn_global_merge_v1"


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


def build_policies(base):
    policies: dict[str, object] = {}

    attn = copy.copy(base)
    attn.__class__ = GuidedSemanticAttentionCDInference
    attn.alpha = LAMBDA
    attn.lambd = LAMBDA
    attn.selection_mode = "semantic"
    attn.kmeans_K = KMEANS_K
    attn.kmeans_seed = KMEANS_SEED
    attn.attention_layer_start = LAYER_START
    attn.attention_layer_end = LAYER_END
    attn.attention_mask_value = MASK_VALUE
    attn._selector_instr = None
    attn._entities = []
    attn._entity_emb = []
    attn._emb_cache = {}
    attn._episode_trace = []
    attn._episode_logits = []
    attn._episode_seed = 0
    attn._selector_step = 0
    policies["semantic_attn_k8_l8_15"] = attn

    merge = copy.copy(base)
    merge.__class__ = SemanticMergeCDInference
    merge.alpha = LAMBDA
    merge.lambd = LAMBDA
    merge.selection_mode = "semantic"
    merge.kmeans_K = KMEANS_K
    merge.kmeans_seed = KMEANS_SEED
    merge.eta = ETA
    merge._selector_instr = None
    merge._entities = []
    merge._entity_emb = []
    merge._emb_cache = {}
    merge._episode_trace = []
    merge._episode_logits = []
    merge._episode_seed = 0
    merge._selector_step = 0
    policies["semantic_merge_k8_eta100"] = merge

    return policies


def _get_image(env, obs):
    return obs["image"][CAMERA_NAME]["rgb"]


def _flatten(action):
    from research.semantic_token_cd.rollout_pilot import flatten_action
    return flatten_action(action)


def run_episode_semantic(env, policy, instruction, obs):
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize

    predicted_terminated = truncated = False
    timestep = 0
    step_infos = []
    executed_actions = []
    image = _get_image(env, obs)
    while not (predicted_terminated or truncated):
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


def audit_attn_trace(trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError("semantic_attn_k8_l8_15 arm produced no trace")
    feature_equal = all(s.get("feature_equal", False) for s in trace)
    guided = all(s.get("guided_prefix", False) for s in trace)
    finite = all(np.isfinite(s.get("residual_norm", 0.0)) for s in trace)
    hook_pass = all(s.get("attention_mask", {}).get("hook_calls") == N_ATTENTION_LAYERS for s in trace)
    kmeans_ok = all(s.get("kmeans_K") == KMEANS_K for s in trace)
    selection_ok = all(s.get("selection_mode") == "semantic" for s in trace)
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_attention_hook_audits_pass": hook_pass,
        "all_guided_prefix": guided,
        "all_residual_norms_finite": finite,
        "all_kmeans_K": kmeans_ok,
        "all_semantic_selection": selection_ok,
        "technical_pass": feature_equal and hook_pass and guided and finite and kmeans_ok and selection_ok,
        "n_cd_steps": len(trace),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"semantic_attn_k8_l8_15 technical audit failed: {audit}")
    return audit


def audit_merge_trace(trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError("semantic_merge_k8_eta100 arm produced no trace")
    feature_equal = all(s.get("feature_equal", False) for s in trace)
    guided = all(s.get("guided_prefix", False) for s in trace)
    finite = all(np.isfinite(s.get("residual_norm", 0.0)) for s in trace)
    n_tokens_ok = all(s.get("n_tokens_negative", 0) == 256 for s in trace)
    eta_ok = all(abs(s.get("eta", 0.0) - ETA) < 1e-9 for s in trace)
    kmeans_ok = all(s.get("kmeans_K") == KMEANS_K for s in trace)
    # Token-count consistency: num_tokens must equal the sum of the selected
    # group sizes (the selector's selected_token_ids == union of group members).
    count_ok = all(
        s.get("num_tokens") == sum(s.get("selected_group_sizes", [])) for s in trace
    )
    # The eta=1.0 merge must be a *real* detail-removal: at least one step has a
    # multi-member selected group with strictly positive within-group variance.
    real_merge = any(
        any(size >= 2 and var > 0.0 for size, var in zip(s.get("selected_group_sizes", []), s.get("within_group_variance_before", [])))
        for s in trace
    )
    audit = {
        "all_negative_branch_sees_clean_v": feature_equal,
        "all_guided_prefix": guided,
        "all_residual_norms_finite": finite,
        "all_256_tokens": n_tokens_ok,
        "all_eta_100": eta_ok,
        "all_kmeans_K": kmeans_ok,
        "all_token_counts_consistent": count_ok,
        "at_least_one_real_merge": real_merge,
        "technical_pass": feature_equal and guided and finite and n_tokens_ok and eta_ok and kmeans_ok and count_ok and real_merge,
        "n_cd_steps": len(trace),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"semantic_merge_k8_eta100 technical audit failed: {audit}")
    return audit


def config_lock(task_root: Path, task: str) -> None:
    lock = {
        "experiment": PROTOCOL,
        "phase": "1B mechanism study (Semantic-Merge vs Attention-K8)",
        "task": task,
        "seeds": list(range(SEED_START, SEED_START + N_SEEDS)),
        "arms": list(ARMS),
        "lambda": LAMBDA,
        "guided_prefix": "both branches share a_{<q}^* = clean greedy tokens; negative branch teacher-forced",
        "kmeans": {
            "K": KMEANS_K,
            "seed": KMEANS_SEED,
            "n_init": 10,
            "selector": "entity_set KMeans top-1 cos match (source/target), deduped",
        },
        "semantic_attn_k8_l8_15": {
            "intervention": "block Action Query -> selected visual keys",
            "layers": [LAYER_START, LAYER_END],
            "mask_value_requested": float(MASK_VALUE),
        },
        "semantic_merge_k8_eta100": {
            "intervention": "merge each selected group to its own prototype",
            "eta": ETA,
            "operator": "v_i -> (1-eta) v_i + eta * mu_G for i in G; source/target merged separately",
            "token_count": "256 -> 256 (no pruning)",
        },
        "cd_tokens": "action dims 0..5, gripper dim 6 keeps clean",
        "vanilla": "NOT re-run; paired externally from artifacts/attn_global_merge_v1/episodes/<task>/vanilla",
        "pairing": "capture-once-reuse; hash-verified fail-closed against stored vanilla hashes",
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
    parser.add_argument("--seeds", type=str, default="", help="comma/range; empty = 200..299")
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

    pairs = []
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
                print(json.dumps({"skip": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial_hashes[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode_semantic(
                env, policy, instruction, obs
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            audit = audit_attn_trace(trace) if arm == "semantic_attn_k8_l8_15" else audit_merge_trace(trace)
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
                "lambda": LAMBDA,
                "guided_prefix": True,
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "selection_mode": "semantic",
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
                "mean_residual_norm": float(np.mean([s.get("residual_norm", 0.0) for s in trace])),
                "mean_num_tokens": float(np.mean([s.get("num_tokens", 0) for s in trace])),
                "mean_num_groups": float(np.mean([s.get("num_groups", 0) for s in trace])),
                "mean_language_score": float(np.mean([s.get("language_score", 0.0) for s in trace])),
            }
            if arm == "semantic_attn_k8_l8_15":
                summary["attention_layers"] = [LAYER_START, LAYER_END]
                summary["attention_mask_value_requested"] = float(MASK_VALUE)
            else:
                summary["eta"] = ETA
                summary["merge"] = "semantic_selected_group_prototype"
                summary["mean_within_group_variance_before"] = float(np.mean([
                    v for s in trace for v in s.get("within_group_variance_before", [])
                ]))
            summary.update(audit)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            summaries[arm] = summary
            print(json.dumps({
                "task": args.task, "seed": seed, "arm": arm,
                "success": summary["success"], "steps": steps,
                "technical_pass": audit["technical_pass"],
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
        pairs.append({
            "seed": seed,
            "canonical_snapshot_sha256": canonical,
            "initial_state_sha256": initial_hashes[ARMS[0]][0],
            "initial_rgb_sha256": initial_hashes[ARMS[0]][1],
            "exact_two_arm_pairing": True,
        })

    manifest_path = task_root / f"pairing_manifest_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps({
        "task": args.task, "seeds": seeds, "arms": list(ARMS), "pairs": pairs,
        "all_two_arm_exact_pairing": True,
        "vanilla_paired_from": VANILLA_ARTIFACT,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
