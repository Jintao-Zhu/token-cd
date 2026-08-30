"""Global Spatial Merge CD (GSM-CD, Phase 1A) — 5-arm paired rollout.

Arms (all share the exact initial state/RGB per seed via capture-once-reuse):
    vanilla          — clean OpenVLA (argmax)
    semantic_attn_cd — Oracle GT-mask Attention-CD (tau=0.5, lambda=0.5, L8-15,
                       mask -1e4) with a GUIDED autoregressive prefix (baseline)
    gsm_025          — Global 2x2 Spatial Merge CD, eta=0.25 (guided prefix)
    gsm_050          — Global 2x2 Spatial Merge CD, eta=0.50 (guided prefix)
    gsm_100          — Global 2x2 Spatial Merge CD, eta=1.00 (guided prefix)

The four CD arms share identical seed/lambda/guided-prefix, so any success/harm
delta isolates (a) the merge strength eta, and (b) merge-vs-attention-blocking.

Run (per task; seed-shard across processes/GPUs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/global_merge_rollout.py \
      --artifact artifacts/attn_global_merge_v1 \
      --task google_robot_close_drawer --gpu 1 --seeds 200-219
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
from research.semantic_token_cd.global_merge_policy import (
    GuidedAttentionCDInference,
    GuidedMergeCDInference,
)
from research.semantic_token_cd.local_replace_rollout import (
    get_entity_actor_ids,
    mask_to_overlap,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)

PROTOCOL = "ATTN_GLOBAL_MERGE_CD_V1"
TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
ARMS = ("vanilla", "semantic_attn_cd", "gsm_025", "gsm_050", "gsm_100")
ETAS = {"gsm_025": 0.25, "gsm_050": 0.5, "gsm_100": 1.0}
LAMBDA = 0.5
LAYER_START, LAYER_END = 8, 16
ORACLE_TAU = 0.5
CAMERA_NAME = "overhead_camera"
SEED_START = 200
N_SEEDS = 100
N_ATTENTION_LAYERS = LAYER_END - LAYER_START


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

    vanilla = copy.copy(base)
    vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace = []
    vanilla._episode_logits = []
    policies["vanilla"] = vanilla

    attn = copy.copy(base)
    attn.__class__ = GuidedAttentionCDInference
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
    policies["semantic_attn_cd"] = attn

    for arm, eta in ETAS.items():
        g = copy.copy(base)
        g.__class__ = GuidedMergeCDInference
        g.eta = eta
        g.alpha = LAMBDA
        g.lambd = LAMBDA
        g._selector_instr = None
        g._entities = []
        g._entity_emb = []
        g._emb_cache = {}
        g._episode_trace = []
        g._episode_logits = []
        g._episode_seed = 0
        g._selector_step = 0
        policies[arm] = g

    return policies


def run_episode_gsm(env, policy, instruction, obs, target_ids=None):
    if not isinstance(policy, (GuidedAttentionCDInference, GuidedMergeCDInference)):
        return run_episode(env, policy, instruction, obs)
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize

    predicted_terminated = truncated = False
    timestep = 0
    step_infos = []
    executed_actions = []
    image = _get_image(env, obs)
    while not (predicted_terminated or truncated):
        if isinstance(policy, GuidedAttentionCDInference):
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
    return obs["image"][CAMERA_NAME]["rgb"]


def _flatten(action):
    from research.semantic_token_cd.rollout_pilot import flatten_action
    return flatten_action(action)


def audit_attn_trace(trace: list[dict], expected_calls: int) -> dict:
    if not trace:
        raise RuntimeError("semantic_attn_cd arm produced no trace")
    cd_steps = [s for s in trace if not s.get("oracle_empty", False)]
    feature_equal = all(s["feature_equal"] for s in cd_steps)
    hook_pass = all(s["attention_mask"]["hook_calls"] == expected_calls for s in cd_steps)
    guided = all(s.get("guided_prefix", False) for s in cd_steps)
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_attention_hook_audits_pass": hook_pass,
        "all_guided_prefix": guided,
        "technical_pass": feature_equal and hook_pass and guided,
        "n_cd_steps": len(cd_steps),
        "n_oracle_empty_steps": len(trace) - len(cd_steps),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"semantic_attn_cd technical audit failed: {audit}")
    return audit


def audit_merge_trace(trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError("gsm arm produced no trace")
    feature_equal = all(s.get("feature_equal", False) for s in trace)
    guided = all(s.get("guided_prefix", False) for s in trace)
    finite = all(np.isfinite(s.get("residual_norm", 0.0)) for s in trace)
    n_tokens_ok = all(s.get("n_tokens_negative", 0) == 256 for s in trace)
    audit = {
        "all_negative_branch_sees_clean_v": feature_equal,
        "all_guided_prefix": guided,
        "all_residual_norms_finite": finite,
        "all_256_tokens": n_tokens_ok,
        "technical_pass": feature_equal and guided and finite and n_tokens_ok,
        "n_cd_steps": len(trace),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"gsm technical audit failed: {audit}")
    return audit


def config_lock(task_root: Path, task: str) -> None:
    lock = {
        "experiment": PROTOCOL,
        "phase": "1A (Global Spatial Merge strength sweep)",
        "task": task,
        "seeds": list(range(SEED_START, SEED_START + N_SEEDS)),
        "arms": list(ARMS),
        "lambda": LAMBDA,
        "guided_prefix": "both branches share a_{<q}^* = clean greedy tokens; negative branch teacher-forced",
        "attention_cd": {
            "selector": "Oracle GT instance mask, tau=0.5 (full mode)",
            "layers": [LAYER_START, LAYER_END],
            "mask_value_requested": -10000.0,
        },
        "global_merge": {
            "operator": "v_i -> (1-eta) v_i + eta * mu_block, 2x2 non-overlapping blocks on 16x16 grid",
            "token_count": "256 -> 256 (no pruning)",
            "etas": {k: v for k, v in ETAS.items()},
            "cd_tokens": "action dims 0..5, gripper dim 6 keeps clean",
        },
        "pairing": "capture-once-reuse; all 5 arms share the exact initial state/RGB per seed",
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
            result, steps, reason, actions, jitter = run_episode_gsm(
                env, policy, instruction, obs, target_ids
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            if arm == "vanilla":
                audit = None
            elif arm == "semantic_attn_cd":
                audit = audit_attn_trace(trace, N_ATTENTION_LAYERS)
            else:
                audit = audit_merge_trace(trace)
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
                "guided_prefix": arm != "vanilla",
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
            }
            if arm == "semantic_attn_cd":
                summary["attention_layers"] = [LAYER_START, LAYER_END]
                summary["oracle_tau"] = ORACLE_TAU
                summary["oracle_mode"] = "full"
            elif arm in ETAS:
                summary["eta"] = ETAS[arm]
                summary["merge"] = "global_2x2"
            if audit is not None:
                summary.update(audit)
            if arm == "semantic_attn_cd":
                summary["oracle_mean_num_tokens"] = float(
                    np.mean([s.get("num_tokens", 0)
                             for s in trace if not s.get("oracle_empty", False)])
                )
                summary["oracle_empty_rate"] = float(
                    np.mean([1.0 if s.get("oracle_empty", False) else 0.0 for s in trace])
                )
            if arm in ETAS:
                cd = trace
                summary["mean_residual_norm"] = float(
                    np.mean([s.get("residual_norm", 0.0) for s in cd])
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
        pairs.append({"seed": seed, "canonical_snapshot_sha256": canonical, "exact_five_arm_pairing": True})

    manifest_path = task_root / f"pairing_manifest_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps({
        "task": args.task, "seeds": seeds, "arms": list(ARMS), "pairs": pairs,
        "all_five_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
