#!/usr/bin/env python3
"""Replay canonical OpenVLA trajectories and extract independent GT object area.

This reads actions from the completed VTCG Full runs, restores the exact
canonical snapshot, and records simulator instance-segmentation area at two
deterministic intervention states per episode. It does not load a policy or
generate new actions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

TASK_TARGETS = {
    "google_robot_open_drawer": ("drawer_from_instruction", None),
    "google_robot_close_drawer": ("drawer_from_instruction", None),
    "google_robot_pick_coke_can": ("actor", "opened_coke_can"),
    "google_robot_move_near": ("objects_from_instruction", None),
}


def sha_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.tobytes()).hexdigest()


OBJECT_ALIASES = {
    "blue plastic bottle": ("blue_plastic_bottle",),
    "pepsi can": ("baked_opened_pepsi_can_v2", "baked_opened_pepsi_can"),
    "coke can": ("baked_opened_coke_can_v2", "baked_opened_coke_can"),
    "redbull can": ("baked_opened_redbull_can_v2", "baked_opened_redbull_can"),
    "7up can": ("baked_opened_7up_can_v2", "baked_opened_7up_can"),
    "apple": ("baked_apple_v2", "baked_apple"),
    "sponge": ("baked_sponge_v2", "baked_sponge"),
    "orange": ("orange",),
}


def instance_ids(env, task, instruction):
    mode, names = TASK_TARGETS[task]
    world = env.unwrapped
    actors = {actor.get_name(): int(actor.get_id()) for actor in getattr(world, "_actors", [])}
    links = {}
    for articulation in getattr(world, "_articulations", []):
        for link in articulation.get_links():
            links[link.get_name()] = int(link.get_id())
    if mode == "actor":
        matches = [name for name in actors if name == names or name.endswith(names)]
        if not matches:
            raise KeyError(f"target actor {names!r} absent; available={sorted(actors)}")
        name = matches[0]
        return {name: actors[name]}
    if mode == "link":
        if names not in links:
            raise KeyError(f"target link {names!r} absent; available={sorted(links)}")
        return {names: links[names]}
    if mode == "actors_union":
        missing = [name for name in names if name not in actors]
        if missing:
            raise KeyError(f"target actors missing={missing}; available={sorted(actors)}")
        return {name: actors[name] for name in names}
    if mode == "drawer_from_instruction":
        drawer = next((name for name in ("top_drawer", "middle_drawer", "bottom_drawer") if name.replace("_", " ") in instruction), None)
        if drawer is None or drawer not in links:
            raise KeyError(f"cannot map drawer from instruction={instruction!r}; links={sorted(links)}")
        return {drawer: links[drawer]}
    if mode == "objects_from_instruction":
        import re
        match = re.fullmatch(r"move (.+?) near (.+)", instruction.strip())
        if not match:
            raise ValueError(f"unexpected move-near instruction {instruction!r}")
        names = []
        for phrase in match.groups():
            aliases = OBJECT_ALIASES.get(phrase)
            if not aliases:
                raise KeyError(f"unmapped object phrase {phrase!r}")
            resolved = next((candidate for candidate in aliases if candidate in actors), None)
            if resolved is None:
                raise KeyError(f"no actor for {phrase!r}; aliases={aliases}; actors={sorted(actors)}")
            names.append(resolved)
        return {name: actors[name] for name in names}
    raise ValueError(mode)


def get_seg(obs):
    seg = np.asarray(obs["image"]["overhead_camera"]["Segmentation"])
    if seg.ndim != 3 or seg.shape[-1] < 2:
        raise ValueError(f"unexpected segmentation shape {seg.shape}")
    # ManiSkill2/SAPIEN channel 1 is the actor/articulation-link instance id.
    return seg[..., 1].astype(np.int64, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASK_TARGETS), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--seed-start", type=int, default=100)
    ap.add_argument("--seed-end", type=int, default=199)
    ap.add_argument("--workspace", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    # The local launcher sets CUDA_VISIBLE_DEVICES before Python starts. The
    # SIMPLER renderer is therefore addressed as cuda:0 inside this worker.
    from research.semantic_token_cd import l11_table2_openvla_rollout as _locked_runner  # noqa: F401
    from research.semantic_token_cd.prompt_attn_l11_budget_causal_rollout import make_environment
    from research.semantic_token_cd.distractor_rollout import restore_snapshot

    workspace = args.workspace.resolve()
    snapshot_root = workspace / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
    full_root = workspace / "artifacts/l11_table2_main_ablation_v1/formal/full/episodes" / args.task / "full"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env, env_id = make_environment(args.task)
    out_rows = []
    replay_checks = []
    try:
        for seed in range(args.seed_start, args.seed_end + 1):
            summary_path = full_root / f"episode_{seed:03d}_summary.json"
            arrays_path = full_root / f"episode_{seed:03d}_arrays.npz"
            snapshot_path = snapshot_root / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
            summary = json.loads(summary_path.read_text())
            with np.load(arrays_path, allow_pickle=True) as arrays:
                actions = np.asarray(arrays["executed_actions"], dtype=np.float32)
            trace = summary["selector_trace"]
            if len(trace) != len(actions) or len(actions) != int(summary["steps"]):
                raise RuntimeError(f"trace/action/summary length mismatch {args.task}:{seed}")
            active = [i for i, row in enumerate(trace) if int(row.get("actual_selected_count", 0)) > 0]
            if len(active) < 2:
                raise RuntimeError(f"fewer than two effective intervention steps {args.task}:{seed}")
            split = (len(active) + 1) // 2
            sample_indices = [active[(split - 1) // 2], active[split + (len(active) - split - 1) // 2]]
            sample_indices = sorted(set(sample_indices))
            snapshot = pickle.loads(snapshot_path.read_bytes())
            obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
            if state_hash != summary["initial_state_sha256"] or rgb_hash != summary["initial_rgb_sha256"]:
                raise RuntimeError(f"initial state/RGB mismatch {args.task}:{seed}")
            ids = instance_ids(env, args.task, summary["instruction"])
            H, W = get_seg(obs).shape
            captured = set()
            final_info = None
            for action_index, action in enumerate(actions):
                if action_index in sample_indices:
                    seg = get_seg(obs)
                    per_name = {name: int(np.count_nonzero(seg == iid)) for name, iid in ids.items()}
                    union = np.isin(seg, list(ids.values()))
                    trace_row = trace[action_index]
                    out_rows.append({
                        "task": args.task,
                        "seed": seed,
                        "episode_success": bool(summary["success"]),
                        "sample_segment": "early" if action_index == sample_indices[0] else "late",
                        "step_index_zero_based": action_index,
                        "normalized_step": action_index / max(1, len(actions) - 1),
                        "m_t": int(trace_row.get("m_t", 0)),
                        "gt_instance_ids": json.dumps(ids, sort_keys=True),
                        "gt_target_pixel_count": int(union.sum()),
                        "gt_target_area_fraction": float(union.mean()),
                        "gt_target_bbox_area_fraction": float((
                            ((np.where(union)[0].max() - np.where(union)[0].min() + 1) *
                             (np.where(union)[1].max() - np.where(union)[1].min() + 1)) / (H * W)
                        )) if union.any() else 0.0,
                        "gt_per_instance_pixels": json.dumps(per_name, sort_keys=True),
                        "gt_per_instance_area_fraction": json.dumps({name: count / (H * W) for name, count in per_name.items()}, sort_keys=True),
                        "segmentation_sha256": sha_array(seg),
                        "rgb_sha256_replay": sha_array(np.asarray(obs["image"]["overhead_camera"]["rgb"])),
                        "initial_state_sha256": state_hash,
                        "initial_rgb_sha256": rgb_hash,
                    })
                    captured.add(action_index)
                obs, _reward, _terminated, _truncated, final_info = env.step(action)
            if captured != set(sample_indices):
                raise RuntimeError(f"sample capture incomplete {args.task}:{seed}")
            if seed < args.seed_start + 5:
                replay_success = bool(final_info.get("success", False)) if final_info else False
                replay_checks.append({"task": args.task, "seed": seed,
                                      "stored_success": bool(summary["success"]),
                                      "replay_success": replay_success,
                                      "match": replay_success == bool(summary["success"])})
            if seed % 10 == 0:
                print(json.dumps({"task": args.task, "seed": seed, "states": len(out_rows)}), flush=True)
    finally:
        env.close()

    fields = list(out_rows[0]) if out_rows else []
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    check_path = args.output.with_suffix(".replay_checks.json")
    check_path.write_text(json.dumps({"env_id": env_id, "task": args.task, "target_spec": TASK_TARGETS[args.task],
                                     "instance_ids": ids, "replay_checks": replay_checks,
                                     "all_replay_checks_match": all(x["match"] for x in replay_checks),
                                     "states": len(out_rows)}, indent=2) + "\n")
    print(json.dumps({"task": args.task, "states": len(out_rows), "csv": str(args.output),
                      "replay_check": str(check_path), "all_replay_checks_match": all(x["match"] for x in replay_checks)}), flush=True)


if __name__ == "__main__":
    main()
