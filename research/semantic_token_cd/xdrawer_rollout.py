"""Closed-loop simulator-annotated drawer cross-instruction episodes.

Arms per (seed, instruction drawer):
  - vanilla: base OpenVLA greedy policy.
  - shr_target: SHR contrast decoding on the annotated current-target drawer region.
  - shr_other:  SHR contrast decoding on the annotated *other* drawer region.

Regions come from simulator 3D geometry each step (never from the instruction),
so a fixed region's mask and harmonic reconstruction are identical under both
instructions.  Vanilla episodes under instruction top/middle additionally emit
evenly spaced state snapshots for the offline four-branch analysis.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.xdrawer_protocol import (
    CONTACT_DIST_M, DRAWERS, ENV_BY_DRAWER, INSTRUCTION_BY_DRAWER, OPEN_QPOS_M,
    PROTOCOL, REPO_ROOT, SEEDS, TASK_BY_DRAWER, array_sha, atomic_json,
    drawer_regions, load_snapshot, make_drawer_env, region_aux,
    restore_snapshot, rgb_sha, snapshot_sha,
)
from research.semantic_token_cd.semantic_recon_rollout import LAMBDA, _init_common
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json as _aj  # noqa: F401 (style)

ARMS = ("vanilla", "shr_target", "shr_other")
STATE_STEPS = 3


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(s < 0 or s > 999 for s in result):
        raise ValueError(f"invalid seeds: {spec}")
    return result


def other_drawer(drawer_id: str) -> str:
    return "middle" if drawer_id == "top" else "top"


def build_policy(base, arm: str, drawer_id: str):
    if arm == "vanilla":
        from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
        policy = copy.copy(base)
        policy.__class__ = AuditedVanillaInference
        policy._episode_trace = []
        policy._episode_logits = []
        return policy
    policy = copy.copy(base)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0
    policy.selector_mode = "sim_region"
    policy.attention_layers = (11,)
    policy.save_prompt_attention = False
    policy.region_token_ids = None
    policy.region_label = None
    policy.region_aux = None
    policy.task_index = 0
    return policy


def sample_step_indices(total_steps: int) -> list[int]:
    if total_steps <= STATE_STEPS:
        return list(range(total_steps))
    return sorted({round(i * (total_steps - 1) / (STATE_STEPS - 1)) for i in range(STATE_STEPS)})


def emit_state(root: Path, drawer_id: str, seed: int, control_step: int, image,
               regs: dict, extra: dict) -> None:
    out = root / drawer_id / f"seed_{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    np_path = out / f"step_{control_step:04d}.npz"
    np.savez_compressed(np_path, image=np.asarray(image, dtype=np.uint8))
    meta = {
        "protocol_id": PROTOCOL, "drawer": drawer_id, "seed": seed,
        "instruction": INSTRUCTION_BY_DRAWER[drawer_id], "control_step": control_step,
        "regions": {d: regs[d] for d in DRAWERS},
        "sizes": regs["__sizes__"], "overlap": regs["__overlap_top_middle__"],
        **extra,
    }
    atomic_json(out / f"step_{control_step:04d}.json", meta)


def audit_shr_trace(trace, arm: str, region_label: str) -> dict:
    """Fail-closed technical audit for the simulator-region SHR arms."""
    if not trace:
        raise RuntimeError(f"{arm} produced no trace")
    checks = {
        "all_feature_equal": all(s.get("feature_equal") is True for s in trace),
        "all_guided_prefix": all(s.get("guided_prefix") is True for s in trace),
        "all_reconstruction_finite": all(s.get("reconstruction_finite") is True for s in trace),
        "all_lambda_locked": all(abs(float(s.get("lambda", -1.0)) - LAMBDA) < 1e-12 for s in trace),
        "all_beta_zero": all(abs(float(s.get("beta", 1.0))) < 1e-12 for s in trace),
        "all_coverage_exact": all(s.get("coverage_exact") is True for s in trace),
        "all_sim_region_mode": all(
            str(s.get("coverage_mode", "")).startswith("sim_region_") for s in trace
        ),
        "all_region_label_locked": all(
            s.get("region_label") == region_label
            and str(s.get("coverage_mode", "")) == f"sim_region_{region_label}"
            for s in trace
        ),
    }
    size_ok = True
    for s in trace:
        aux = s.get("region_aux") or {}
        own_size = int(aux.get(f"region_size_{region_label}", -1))
        if own_size >= 1 and len(s.get("selected_token_ids", [])) != own_size:
            size_ok = False
    checks["all_mask_size_matches_region"] = size_ok
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"sim-region SHR audit failed for {arm}: {checks}")
    return checks


def run_one(env, policy, arm, drawer_id, seed, snapshot, emit: Path | None) -> dict:
    from utils import convert_numpy_or_torch_to_python, summarize
    from parallel_inference import get_image_from_maniskill2_obs_dict

    obs, state_sha, rgb_sha_init = _restore_and_hash(env, seed, snapshot)
    inner = env.unwrapped
    instruction = inner.get_language_instruction()
    if instruction != INSTRUCTION_BY_DRAWER[drawer_id]:
        raise RuntimeError(f"instruction mismatch: {instruction!r} != expected for {drawer_id}")
    if inner.drawer_id != drawer_id:
        raise RuntimeError(f"env target drawer {inner.drawer_id} != {drawer_id}")
    region_label = drawer_id if arm == "shr_target" else (other_drawer(drawer_id) if arm == "shr_other" else None)
    if policy is not None:
        policy._episode_trace = []
        policy._episode_logits = []
        policy._episode_seed = seed
        policy._selector_step = 0
    if hasattr(policy, "reset") and arm != "vanilla":
        policy.reset(instruction, seed=seed)

    image = get_image_from_maniskill2_obs_dict(env, obs)
    predicted_terminated = truncated = False
    step_infos = []
    executed = []
    geometry_log = []
    contact = {"top": None, "middle": None}
    state_buffer = []
    control = 0
    while not (predicted_terminated or truncated) and control < 130:
        regs = drawer_regions(inner, obs)
        if emit is not None and arm == "vanilla":
            state_buffer.append({
                "control": control,
                "image": np.asarray(image, dtype=np.uint8).copy(),
                "regs": regs,
                "state_sha": state_sha,
            })
        eef_p = np.asarray(inner.tcp.pose.p, dtype=np.float64)
        row = {"control": control, "eef_p": eef_p.tolist(), "geometry": {}}
        for d in DRAWERS:
            geo = regs["__geometry__"][d]
            hp = geo["handle_p"]
            dist = None if hp is None else float(np.linalg.norm(eef_p - np.asarray(hp)))
            row["geometry"][d] = {
                "joint_qpos": geo["joint_qpos"], "handle_p": hp,
                "front_p": geo["front_p"], "eef_handle_dist": dist,
            }
            if dist is not None and dist < CONTACT_DIST_M and contact[d] is None:
                contact[d] = control
        row["region_sizes"] = regs["__sizes__"]
        row["region_overlap"] = regs["__overlap_top_middle__"]
        if region_label:
            q = regs[region_label]
            row["region_quality"] = {
                "label": region_label,
                "own_px": q.get("own_px", 0), "neighbor_px": q.get("neighbor_px", 0),
                "region_px": q.get("region_px", 0), "size": len(q["token_ids"]),
            }
        geometry_log.append(row)

        aux = region_aux(regs, region_label) if region_label else {}
        if arm != "vanilla":
            policy.region_token_ids = list(regs[region_label]["token_ids"])
            policy.region_label = region_label
            policy.region_aux = aux
        # All audited policies share the (image, contrast_image=None,
        # task_description, ...) signature and return a 3-tuple.
        raw_action, actions, _aux = policy.step(
            image, None, instruction, proprio=obs["agent"]["eef_pos"]
        )
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            a7 = _flatten(action)
            if a7.shape != (7,) or not np.isfinite(a7).all():
                raise FloatingPointError(f"invalid executed action at ctrl {control}: {a7}")
            executed.append(a7.copy())
            obs, _reward, _success, truncated, info = env.step(a7)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            control += 1
            step_infos.append(convert_numpy_or_torch_to_python(info))
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not env.unwrapped.is_final_subtask():
                predicted_terminated = False
                env.advance_to_next_subtask()
    if emit is not None and arm == "vanilla" and state_buffer:
        chosen = sorted(set(sample_step_indices(len(state_buffer))))
        for i in chosen:
            rec = state_buffer[i]
            rgb = array_sha(rec["image"])
            emit_state(emit, drawer_id, seed, rec["control"], rec["image"], rec["regs"],
                       {"state_sha": rec["state_sha"], "rgb_sha": rgb,
                        "buffer_index": i, "total_buffer": len(state_buffer)})
    result = summarize(step_infos)
    reason = None if result.get("success") else ("time_limit" if truncated else "policy_terminated")
    final_qpos = {}
    for d in DRAWERS:
        q = float(np.asarray(inner.art_obj.get_qpos())[np.asarray([inner.joint_names.index(f"{d}_drawer_joint")])[0]])
        final_qpos[d] = q
    region_stats = {"top": 0, "middle": 0, "overlap": 0}
    trace = getattr(policy, "_episode_trace", None)
    if trace:
        sizes = [t.get("region_aux") or {} for t in trace]
        if sizes:
            region_stats = {
                "top": float(np.mean([s.get("region_size_top", 0) for s in sizes])),
                "middle": float(np.mean([s.get("region_size_middle", 0) for s in sizes])),
                "overlap": float(np.mean([s.get("region_overlap_top_middle", 0) for s in sizes])),
            }
    shr_audit = None
    if arm in ("shr_target", "shr_other"):
        shr_audit = audit_shr_trace(trace, arm, region_label)
    return {
        "success": bool(result.get("success", False)),
        "result": jsonable(result),
        "failure_reason": reason,
        "control_steps": control,
        "initial_state_sha256": state_sha,
        "initial_rgb_sha256": rgb_sha_init,
        "snapshot_sha256": snapshot_sha(snapshot),
        "contact_step": {d: contact[d] for d in DRAWERS},
        "contact_order": [d for d in DRAWERS if contact[d] is not None],
        "final_drawer_qpos": final_qpos,
        "region_mean_sizes": region_stats,
        "region_steps": geometry_log,
        "shr_audit": shr_audit,
        "trace_len": len(trace) if trace else 0,
        "emitted_state_count": len(state_buffer) if (emit is not None and arm == "vanilla") else 0,
    }


def _restore_and_hash(env, seed, snapshot):
    obs = restore_snapshot(env, seed, snapshot)
    inner = env.unwrapped
    state_sha = array_sha(np.asarray(inner.get_state()))
    rgb = rgb_sha(env, obs)
    return obs, state_sha, rgb


def _flatten(action):
    from research.semantic_token_cd.rollout_pilot import flatten_action
    return flatten_action(action)


def jsonable(value):
    from research.semantic_token_cd.distractor_rollout import jsonable as _jsonable
    return _jsonable(value)


def ensure_config(artifact: Path, snapshots: Path, emit: Path | None) -> dict:
    artifact.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_id": PROTOCOL,
        "purpose": "cross-instruction simulator-region mechanism experiment (open top vs open middle)",
        "arms": list(ARMS),
        "envs": ENV_BY_DRAWER,
        "instructions": INSTRUCTION_BY_DRAWER,
        "seeds": list(SEEDS),
        "snapshot_root": str(snapshots),
        "region_rule": "current-frame overhead camera + drawer front panel (URDF 0.615x0.14 box) corners "
                       "-> 16x16 token rectangle, neighbour-pixel trimmed by current-frame segmentation",
        "region_rule_independent_of_instruction": True,
        "emit_states": "vanilla episodes only; worker-local output directory (not part of config identity)",
        "locked_downstream": {
            "lambda": 0.5, "beta": 0.0, "guided_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper": "clean positive dimension 6", "prefix": "shared clean greedy prefix",
            "sampling": False, "spatial_postprocessing": False,
            "harmonic": "16x16 four-neighbor Dirichlet beta=0/gamma=1",
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("xdrawer config lock differs")
    atomic_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--drawer", choices=DRAWERS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--emit-states", type=Path, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    snapshots = args.snapshots.resolve()
    emit = None if args.emit_states is None else args.emit_states.resolve()
    ensure_config(artifact, snapshots, emit)
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not arms or any(a not in ARMS for a in arms):
        raise ValueError(f"invalid arms: {arms}")
    seeds = parse_seeds(args.seeds)
    env, environment_id = make_drawer_env(args.drawer, args.gpu)
    checkpoint = str(REPO_ROOT.parent / "pcd_openvla_simpler_box_31b027e/source/pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, TASK_BY_DRAWER[args.drawer], {}, False)
    base = OpenVLAInference(**config)
    policies = {a: build_policy(base, a, args.drawer) for a in arms}

    for seed in seeds:
        snapshot = load_snapshot(snapshots, args.drawer, seed)
        for arm in arms:
            out = artifact / "episodes" / TASK_BY_DRAWER[args.drawer] / arm
            summary_path = out / f"episode_{seed:03d}_summary.json"
            if summary_path.exists():
                print(json.dumps({"skip": True, "drawer": args.drawer, "seed": seed, "arm": arm}), flush=True)
                continue
            started = time.monotonic()
            info = run_one(env, policies[arm], arm, args.drawer, seed, snapshot, emit)
            runtime = time.monotonic() - started
            info.update({
                "protocol_id": PROTOCOL, "task": TASK_BY_DRAWER[args.drawer],
                "environment_id": environment_id, "drawer": args.drawer,
                "instruction": INSTRUCTION_BY_DRAWER[args.drawer], "seed": seed,
                "evaluation_seed": seed, "episode_id": seed, "arm": arm,
                "runtime_seconds": runtime, "gpu_id": args.gpu, "worker_id": args.worker_id,
            })
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(summary_path, info)
            print(json.dumps({"drawer": args.drawer, "seed": seed, "arm": arm,
                              "success": info["success"], "steps": info["control_steps"],
                              "runtime_seconds": round(runtime, 1)}), flush=True)


if __name__ == "__main__":
    main()
