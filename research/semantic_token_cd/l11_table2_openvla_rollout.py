"""OpenVLA Table 2 component ablations on paired SIMPLER canonical scenes."""
from __future__ import annotations

import argparse
import copy
import cv2
import json
import os
import pickle
import time
from pathlib import Path

from research.semantic_token_cd import prompt_attn_l11_budget_causal_rollout as shared
from research.semantic_token_cd.distractor_rollout import (
    flatten_action, get_image_from_maniskill2_obs_dict, jsonable, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json, finite_mean, write_arrays
from research.semantic_token_cd.semantic_recon_rollout import TASKS as ALL_TASKS, TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode

TASKS = tuple(ALL_TASKS)
FIG4_TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
RT_DEPTH = {
    **shared.RT_DEPTH,
    "google_robot_place_apple_in_closed_top_drawer": 0,
    "widowx_carrot_on_plate": 0,
    "widowx_put_eggplant_in_basket": 0,
    "widowx_spoon_on_towel": 0,
    "widowx_stack_cube": 0,
}
# The shared renderer factory consumes this per-task map. Extend it only in
# this runner process so other studies keep their four-Google-task lock.
shared.RT_DEPTH.update(RT_DEPTH)
RT0_TASKS = set(RT_DEPTH) - {"google_robot_open_drawer", "google_robot_close_drawer"}
CARD_ARMS = {
    "1": {
        "zero": ("zero", "contrast", 0.5),
        "clean_global_mean": ("clean_global_mean", "contrast", 0.5),
        "local_neighbor_mean": ("local_neighbor_mean", "contrast", 0.5),
    },
    "2": {
        "hard_intervention_only": ("zero", "hard_intervention_only", 0.5),
        "harmonic_negative_only": ("harmonic", "negative_only", 0.5),
        "random_ranking": ("harmonic", "contrast", 0.5),
    },
    "3": {
        "layer5": ("harmonic", "contrast", 0.5),
        "layer11": ("harmonic", "contrast", 0.5),
        "geometry_ranking": ("harmonic", "contrast", 0.5),
    },
    "6": {
        "full": ("harmonic", "contrast", 0.5),
        "kmeans_ranking": ("harmonic", "contrast", 0.5),
        "fixed32": ("harmonic", "contrast", 0.5),
        "shuffle_m": ("harmonic", "contrast", 0.5),
        "dose_0p50": ("harmonic", "contrast", 0.5),
        "dose_0p75": ("harmonic", "contrast", 0.5),
        "dose_1p25": ("harmonic", "contrast", 0.5),
        "dose_1p50": ("harmonic", "contrast", 0.5),
    },
    "7": {
        "target_enriched": ("harmonic", "contrast", 0.5),
        "target_sparing": ("harmonic", "contrast", 0.5),
    },
}
DOSE_SCALES = {
    "dose_0p50": 0.50,
    "dose_0p75": 0.75,
    "dose_1p25": 1.25,
    "dose_1p50": 1.50,
}
CARD_LOCATION = {("3", "layer5"): "layer5", ("3", "layer11"): "layer11"}
SCHEDULE_FILE = Path(__file__).resolve().parents[2] / "artifacts/l11_matched_budget_causal_pilot_v1/BUDGET_SCHEDULES.json"


def build_policies(
    base, task: str, card: str, arms: tuple[str, ...], *, partial_l11_attention: bool = False
):
    policies = {}
    for arm in arms:
        constructor, execution, strength = CARD_ARMS[card][arm]
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, strength)
        policy._task_name = task
        policy.beta = 0.0
        policy.selector_mode = "prompt_attention"
        policy.task_index = TASK_INDEX[task]
        policy.attention_layers = (11,)
        policy.use_partial_l11_attention = bool(partial_l11_attention)
        policy.selection_count = 32 if arm == "fixed32" else None
        policy.selection_top_p = None
        policy.selection_budget_schedule = None
        policy.selection_budget_scale = DOSE_SCALES.get(arm, 1.0)
        policy.selection_budget_label = arm
        policy.selection_budget_source = "matched"
        policy.counterfactual_constructor = constructor
        policy.guidance_execution = execution
        policy.intervention_location = CARD_LOCATION.get((card, arm), "projector")
        policy.mask_selector = {
            "random_ranking": "random_matched",
            "geometry_ranking": "geometry_mask",
            "kmeans_ranking": "kmeans_matched",
        }.get(arm, "l11")
        policy.region_mask_mode = arm if arm in {"target_enriched", "target_sparing"} else None
        policy._region_patch_overlap = None
        policy.save_prompt_attention = False
        policies[arm] = policy
    return policies


def audit(trace: list[dict], arm: str, card: str) -> dict:
    constructor, execution, strength = CARD_ARMS[card][arm]
    checks = {
        "nonempty_trace": bool(trace),
        "matched_count_exact": bool(trace) and all(
            int(x.get("actual_selected_count", -1)) == int(x.get("m_t", -2)) for x in trace
        ),
        "clean_features_preserved": bool(trace) and all(x.get("feature_equal") is True for x in trace),
        "unselected_features_unchanged": bool(trace) and all(
            x.get("non_target_bit_identical") is True for x in trace
        ),
        "finite_reconstruction": bool(trace) and all(x.get("reconstruction_finite") is True for x in trace),
        "lambda_exact": bool(trace) and all(abs(float(x.get("lambda", -1)) - strength) < 1e-12 for x in trace),
        "layer_exact": bool(trace) and all(x.get("attention_layers") == [11] for x in trace),
        "constructor_exact": bool(trace) and all(
            x.get("counterfactual_constructor") == constructor for x in trace
        ),
        "execution_exact": bool(trace) and all(x.get("guidance_execution") == execution for x in trace),
        "intervention_location_exact": bool(trace) and all(
            x.get("intervention_location") == CARD_LOCATION.get((card, arm), "projector")
            for x in trace
        ),
        "ranking_selector_exact": bool(trace) and all(
            x.get("mask_selector") == {
                "random_ranking": "random_matched",
                "geometry_ranking": "geometry_mask",
                "kmeans_ranking": "kmeans_matched",
            }.get(arm, "l11")
            for x in trace
        ),
        "seven_action_tokens": bool(trace) and all(
            len(x.get("positive_token_ids", [])) == 7
            and len(x.get("negative_token_ids", [])) == 7
            and len(x.get("final_token_ids", [])) == 7 for x in trace
        ),
    }
    if execution != "contrast":
        checks["direct_negative_decode"] = bool(trace) and all(
            x.get("final_token_ids") == x.get("negative_token_ids") for x in trace
        )
    if card == "2" and arm == "hard_intervention_only":
        checks["zero_constructor"] = bool(trace) and all(
            x.get("counterfactual_constructor") == "zero" for x in trace
        )
    if card == "1" and arm in {"zero", "clean_global_mean"}:
        checks["reconstruction_constructor_exact"] = bool(trace) and all(
            x.get("counterfactual_constructor") == arm for x in trace
        )
    if arm in DOSE_SCALES:
        scale = DOSE_SCALES[arm]
        checks["dose_scale_exact"] = bool(trace) and all(
            abs(float(x.get("budget_scale", -1)) - scale) < 1e-12 for x in trace
        )
        checks["dose_count_exact"] = bool(trace) and all(
            int(x.get("m_t", -1)) == max(1, min(256, int(round(
                scale * int(x.get("unscaled_matched_m_t", -1))
            ))))
            and int(x.get("actual_selected_count", -2)) == int(x.get("m_t", -1))
            for x in trace
        )
    if card == "7":
        checks["gt_region_mask_mode_exact"] = bool(trace) and all(
            x.get("gt_region_mask_mode") == arm for x in trace
        )
        checks["gt_region_count_exact"] = bool(trace) and all(
            int(x.get("actual_selected_count", -1)) == int(x.get("m_t", -2))
            for x in trace
        )
        if arm == "target_sparing":
            checks["no_target_patch_selected"] = bool(trace) and all(
                int(x.get("gt_selected_target_patch_count", -1)) == 0 for x in trace
            )
        elif arm == "target_enriched":
            checks["target_enriched_capacity_exact"] = bool(trace) and all(
                int(x.get("gt_selected_target_patch_count", -1))
                == min(int(x.get("gt_target_patch_count", -2)), int(x.get("m_t", -1)))
                for x in trace
            )
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"seven-card card {card} audit failed for {arm}: {checks}")
    return checks


def load_scan_reference(snapshot_root: Path, task: str, seed: int) -> dict:
    """Use the renderer-matched baseline for all depth-0 scan tasks."""
    if task not in RT0_TASKS:
        return shared.load_task_reference(snapshot_root, task, seed)
    path = shared.RT0_REFERENCE / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing RT-depth0 reference: {path}")
    reference = json.loads(path.read_text())
    if "depth0" not in str(reference.get("renderer", "")).lower():
        raise RuntimeError(f"RT-depth0 reference has wrong renderer: {path}")
    # Keep the canonical seed's instruction, which may be randomized by task.
    canonical = shared.load_reference(snapshot_root, task, seed)
    reference["instruction"] = canonical["instruction"]
    return reference


def gt_target_patch_overlap(env, obs, task: str, instruction: str) -> np.ndarray:
    """Map simulator target instance IDs to the exact 224x224/16x16 model grid."""
    u = env.unwrapped
    if task in {"google_robot_open_drawer", "google_robot_close_drawer"}:
        drawer = next((name for name in ("top_drawer", "middle_drawer", "bottom_drawer")
                       if name.replace("_", " ") in instruction), None)
        if drawer is None:
            raise RuntimeError(f"cannot resolve drawer target for {instruction!r}")
        target_ids = [int(link.id) for link in u.art_obj.get_links() if link.name == drawer]
        if len(target_ids) != 1:
            raise RuntimeError(f"expected one drawer-link ID for {drawer}, got {target_ids}")
    elif task == "google_robot_pick_coke_can":
        target_ids = [int(u.obj.id)]
    elif task == "google_robot_move_near":
        target_ids = [int(u.episode_source_obj.id), int(u.episode_target_obj.id)]
    else:
        raise ValueError(f"GT-region mechanism experiment does not support {task}")
    seg = np.asarray(obs["image"]["overhead_camera"]["Segmentation"])
    if seg.ndim != 3 or seg.shape[-1] < 2:
        raise RuntimeError(f"unexpected segmentation shape {seg.shape}")
    mask = np.isin(seg[..., 1], target_ids).astype(np.float32)
    resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_AREA)
    overlap = resized.reshape(16, 14, 16, 14).mean(axis=(1, 3)).reshape(256).astype(np.float32)
    if not np.isfinite(overlap).all() or not np.any(overlap > 0):
        raise RuntimeError(f"empty/non-finite GT target support for {task}: {instruction}")
    return overlap


def run_episode_gt_region(env, policy, instruction, obs):
    """Canonical rollout loop with per-observation GT overlap supplied to policy."""
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize

    predicted_terminated = truncated = False
    timestep = 0
    step_infos = []
    executed_actions = []
    image = get_image_from_maniskill2_obs_dict(env, obs)
    while not (predicted_terminated or truncated):
        policy._region_patch_overlap = gt_target_patch_overlap(env, obs, policy._task_name, instruction)
        _raw_action, actions, _aux = policy.step(
            image, None, instruction, proprio=obs["agent"]["eef_pos"]
        )
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            executed = flatten_action(action)
            if executed.shape != (7,) or not np.isfinite(executed).all():
                raise FloatingPointError(f"Invalid executed action: {executed}")
            executed_actions.append(executed.copy())
            obs, _reward, _success, truncated, info = env.step(executed)
            image = get_image_from_maniskill2_obs_dict(env, obs)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", choices=tuple(CARD_ARMS), required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, default=Path("artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument(
        "--partial-l11-attention", action="store_true",
        help="opt in to cached-projector L11-only attention extraction",
    )
    args = ap.parse_args()
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not arms or any(x not in CARD_ARMS[args.card] for x in arms):
        raise ValueError(f"invalid card {args.card} arms: {arms}")
    seeds = shared.parse_seeds(args.seeds)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    is_dose_run = any(a in DOSE_SCALES for a in arms)
    lock = {
        "protocol_id": (
            "OPENVLA_L11_FIG4_DOSE_RESPONSE_V1" if is_dose_run
            else "OPENVLA_L11_TABLE2_MAIN_ABLATION_V1"
        ) + ("_PARTIAL_L11_ATTENTION" if args.partial_l11_attention else ""),
        "model": "OpenVLA Fractal SIMPLER checkpoint",
        "tasks": list(FIG4_TASKS if is_dose_run else TASKS),
        "seeds": list(range(100, 200)),
        "arms": list(arms),
        "rendering": {
            t: {"shader_dir": "rt", "rt_max_path_depth": RT_DEPTH[t], "rt_samples_per_pixel": 32, "rt_use_denoiser": True}
            for t in (FIG4_TASKS if is_dose_run else TASKS)
        },
        "snapshot_artifact": str(source), "paired_reference": "task-matched RT baseline; RT0 reference for Pick/Move",
        "attention_layer_zero_based": 11, "lambda": {a: CARD_ARMS[args.card][a][2] for a in arms},
        "partial_l11_attention": bool(args.partial_l11_attention),
        "constructor": {a: CARD_ARMS[args.card][a][0] for a in arms},
        "execution": {a: CARD_ARMS[args.card][a][1] for a in arms},
        "intervention_location": {a: CARD_LOCATION.get((args.card, a), "projector") for a in arms},
    }
    if args.card == "7":
        lock.update({
            "gt_region_control": {
                "target_enriched": "same L11 ranking and dynamic Matched m; rank by L11 within visible GT target patches first, then fill from non-target patches",
                "target_sparing": "same L11 ranking and dynamic Matched m; choose highest-L11 tokens outside visible GT target patches",
                "target_definition": "drawer instruction-selected link; coke can actor; Move Near source and target union",
                "model_view_mapping": "instance mask INTER_AREA-resized to 224x224 then patch fractional overlap on 16x16x14x14 grid",
                "target_patch_membership": "any positive fractional overlap",
                "inference_time_gt": False,
            }
        })
    if any(a in {"full", "kmeans_ranking", "fixed32", "shuffle_m"} for a in arms):
        lock.update({
            "selection_count": {a: (32 if a == "fixed32" else None) for a in arms},
            "selection_budget_schedule": {
                a: ("within_task_shuffle" if a == "shuffle_m" else None) for a in arms
            },
            "mask_selector": {
                a: {"random_ranking": "random_matched", "geometry_ranking": "geometry_mask",
                    "kmeans_ranking": "kmeans_matched"}.get(a, "l11")
                for a in arms
            },
        })
    if is_dose_run:
        lock["matched_budget_scale"] = {a: DOSE_SCALES[a] for a in arms}
    lock_path = artifact / "CONFIG_LOCK.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise RuntimeError(f"CONFIG_LOCK mismatch: {lock_path}")
    atomic_json(lock_path, lock)

    env, environment_id = shared.make_environment(args.task)
    schedules = None
    if "shuffle_m" in arms:
        schedules = json.loads(SCHEDULE_FILE.read_text())["schedules"]["within_task_shuffle"]
    model_config = get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), args.task, {}, False)
    policies = build_policies(
        OpenVLAInference(**model_config), args.task, args.card, arms,
        partial_l11_attention=args.partial_l11_attention,
    )
    for seed in seeds:
        snapshot = pickle.loads((source / "snapshots" / args.task / f"seed_{seed:03d}.pkl").read_bytes())
        reference = load_scan_reference(source, args.task, seed)
        expected = (reference["canonical_snapshot_sha256"], reference["initial_state_sha256"], reference["initial_rgb_sha256"])
        if snapshot_sha(snapshot) != expected[0]:
            raise RuntimeError(f"snapshot hash mismatch: {args.task}:{seed}")
        pair_hashes = []
        for arm, policy in policies.items():
            out = artifact / "episodes" / args.task / arm
            summary_path, arrays_path = out / f"episode_{seed:03d}_summary.json", out / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                old = json.loads(summary_path.read_text())
                if old.get("protocol_id") not in {lock["protocol_id"], "SEVEN_CARD_C6_OPENVLA_V1"}:
                    raise RuntimeError(f"incompatible existing output: {summary_path}")
                pair_hashes.append((old["initial_state_sha256"], old["initial_rgb_sha256"]))
                continue
            obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
            if (state_hash, rgb_hash) != expected[1:]:
                raise RuntimeError(f"paired snapshot mismatch {args.task}:{seed}: state={state_hash == expected[1]} rgb={rgb_hash == expected[2]}")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError(f"instruction mismatch: {args.task}:{seed}")
            policy.reset(instruction, seed=seed)
            if arm == "shuffle_m":
                try:
                    policy.selection_budget_schedule = tuple(
                        int(value) for value in schedules[args.task][str(seed)]
                    )
                except KeyError as exc:
                    raise RuntimeError(
                        f"missing frozen within-task shuffle budget schedule for {args.task}:{seed}"
                    ) from exc
                policy.selection_budget_label = "shuffle_m"
            policy._episode_trace, policy._episode_logits = [], []
            started = time.monotonic()
            if args.card == "7":
                result, steps, reason, actions, jerk = run_episode_gt_region(env, policy, instruction, obs)
            else:
                result, steps, reason, actions, jerk = run_episode(env, policy, instruction, obs)
            trace = jsonable(policy._episode_trace)
            checks = audit(trace, arm, args.card)
            out.mkdir(parents=True, exist_ok=True)
            write_arrays(arrays_path, policy._episode_logits, actions)
            payload = {
                "protocol_id": lock["protocol_id"], "task": args.task, "seed": seed, "episode_id": seed,
                "arm": arm, "success": bool(result["success"]), "result": jsonable(result),
                "failure_reason": reason, "steps": steps, "runtime_seconds": time.monotonic() - started,
                "worker_id": args.worker_id, "gpu_id": args.gpu, "environment_id": environment_id,
                "canonical_snapshot_sha256": expected[0], "initial_state_sha256": state_hash,
                "initial_rgb_sha256": rgb_hash, "instruction": instruction,
                "lambda": CARD_ARMS[args.card][arm][2], "attention_layers": [11],
                "partial_l11_attention": bool(args.partial_l11_attention),
                "rendering": {"shader_dir": "rt", "rt_max_path_depth": RT_DEPTH[args.task], "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
                "selector": (
                    f"L11 prompt attention + dynamic Matched budget + {arm} GT-region diagnostic"
                    if args.card == "7" else "L11 prompt attention + canonical dynamic Matched budget"
                ),
                "counterfactual_constructor": CARD_ARMS[args.card][arm][0],
                "guidance_execution": CARD_ARMS[args.card][arm][1],
                "arrays_file": arrays_path.name, "action_jitter_index": jerk,
                "mean_selected_token_count": finite_mean(x.get("actual_selected_count") for x in trace),
                "selector_trace": trace, **checks,
            }
            atomic_json(summary_path, payload)
            pair_hashes.append((state_hash, rgb_hash))
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm, "success": payload["success"], "steps": steps, "runtime_seconds": round(payload["runtime_seconds"], 2)}, sort_keys=True), flush=True)
        if len(set(pair_hashes)) != 1:
            raise RuntimeError(f"arm pair identity mismatch: {args.task}:{seed}: {pair_hashes}")


if __name__ == "__main__":
    main()
