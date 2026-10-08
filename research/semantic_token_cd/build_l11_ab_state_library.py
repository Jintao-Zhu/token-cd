#!/usr/bin/env python3
"""Replay canonical OpenVLA Full trajectories and save four RGB/seg states.

This is a shared offline state library for the L11 Matched budget, token-region,
and reconstruction mechanism studies. It never generates actions or changes a
policy rollout. Each episode is restored from the canonical RT snapshot and
replayed with its already-recorded executed actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
FULL = ROOT / "artifacts/l11_table2_main_ablation_v1/formal/full"


def sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def get_seg(obs: dict) -> np.ndarray:
    seg = np.asarray(obs["image"]["overhead_camera"]["Segmentation"])
    if seg.ndim != 3 or seg.shape[-1] < 2:
        raise ValueError(f"unexpected segmentation shape {seg.shape}")
    return seg[..., 1].astype(np.int64, copy=False)


def values_int(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from values_int(value)
    elif isinstance(obj, int) and not isinstance(obj, bool):
        yield obj


def entity_ids(env, task: str, instruction: str) -> dict[str, dict[str, int]]:
    u = env.unwrapped
    actors = {a.get_name(): int(a.get_id()) for a in getattr(u, "_actors", [])}
    articulations = list(getattr(u, "_articulations", []))
    links = {link.get_name(): int(link.get_id())
             for art in articulations for link in art.get_links()}
    robot = {}
    for art in articulations:
        name = art.get_name().lower()
        if "google_robot" in name or "widowx" in name or "robot" in name:
            robot.update({link.get_name(): int(link.get_id()) for link in art.get_links()})
    result: dict[str, dict[str, int]] = {"robot_links": robot}
    if task in {"google_robot_open_drawer", "google_robot_close_drawer"}:
        drawer = next((n for n in ("top_drawer", "middle_drawer", "bottom_drawer")
                       if n.replace("_", " ") in instruction), None)
        if drawer is None or drawer not in links:
            raise KeyError(f"cannot resolve instructed drawer {instruction!r}; links={sorted(links)}")
        result["target_drawer_link"] = {drawer: links[drawer]}
        # SAPIEN instance segmentation is link-level here: handle geometry is
        # part of the drawer link and has no independently addressable ID.
        result["drawer_handle"] = {"segmentation_note": "shares target drawer link; no separate instance id"}  # type: ignore[assignment]
    elif task == "google_robot_pick_coke_can":
        actor = getattr(u, "obj", None)
        if actor is None:
            raise RuntimeError("pick-coke target actor handle unavailable")
        result["target_object"] = {actor.get_name(): int(actor.get_id())}
    elif task == "google_robot_move_near":
        src, dst = getattr(u, "episode_source_obj", None), getattr(u, "episode_target_obj", None)
        if src is None or dst is None:
            raise RuntimeError("move-near source/target handles unavailable")
        result["source_object"] = {src.get_name(): int(src.id)}
        result["target_object"] = {dst.get_name(): int(dst.id)}
    return result


def entity_poses(env, entity_id_map: dict) -> dict[str, dict]:
    wanted = set(int(x) for x in values_int(entity_id_map))
    result = {}
    u = env.unwrapped
    entities = list(getattr(u, "_actors", []))
    entities.extend(link for art in getattr(u, "_articulations", []) for link in art.get_links())
    for entity in entities:
        entity_id = int(entity.get_id() if hasattr(entity, "get_id") else entity.id)
        if entity_id not in wanted:
            continue
        pose = entity.get_pose()
        result[str(entity_id)] = {
            "name": entity.get_name() if hasattr(entity, "get_name") else entity.name,
            "p": np.asarray(pose.p, dtype=np.float64).tolist(),
            "q": np.asarray(pose.q, dtype=np.float64).tolist(),
        }
    return result


def sample_quartile_indices(trace: list[dict]) -> list[int]:
    active = [i for i, row in enumerate(trace) if int(row.get("actual_selected_count", 0)) > 0]
    if not active:
        raise RuntimeError("trajectory has no effective intervention step")
    # One deterministic midpoint from each quarter of the valid intervention
    # trajectory; short episodes retain unique available indices only.
    n = len(active)
    picks = []
    for q in range(4):
        lo = (q * n) // 4
        hi = ((q + 1) * n) // 4
        if hi <= lo:
            continue
        picks.append(active[(lo + hi - 1) // 2])
    return sorted(set(picks))


def process_one(task: str, seed: int, gpu: int, worker: str, outroot: Path,
                env, env_id: str, restore_snapshot) -> dict:
    summary_path = FULL / "episodes" / task / "full" / f"episode_{seed:03d}_summary.json"
    arrays_path = summary_path.with_name(f"episode_{seed:03d}_arrays.npz")
    snapshot_path = SOURCE / "snapshots" / task / f"seed_{seed:03d}.pkl"
    if not summary_path.is_file() or not arrays_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError(f"missing canonical Full trajectory files for {task}:{seed}")
    summary = json.loads(summary_path.read_text())
    with np.load(arrays_path, allow_pickle=True) as arrays:
        actions = np.asarray(arrays["executed_actions"], dtype=np.float32)
    trace = summary["selector_trace"]
    if len(trace) != len(actions) or len(actions) != int(summary["steps"]):
        raise RuntimeError(f"trace/action length mismatch {task}:{seed}")
    sample_indices = sample_quartile_indices(trace)
    snapshot = pickle.loads(snapshot_path.read_bytes())
    states = []
    obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
    expected = (summary["initial_state_sha256"], summary["initial_rgb_sha256"])
    if (state_hash, rgb_hash) != expected:
        raise RuntimeError(f"initial state/RGB mismatch {task}:{seed}")
    instruction = str(summary["instruction"])
    ids = entity_ids(env, task, instruction)
    wanted = set(sample_indices)
    final_info = None
    for step, action in enumerate(actions):
        if step in wanted:
            rgb = np.asarray(obs["image"]["overhead_camera"]["rgb"])
            seg = get_seg(obs)
            sim_state = np.asarray(env.unwrapped.get_state()).copy()
            eef_pos = np.asarray(obs["agent"]["eef_pos"], dtype=np.float32).copy()
            agent_state_blob = np.frombuffer(
                pickle.dumps(env.unwrapped.agent.get_state(), protocol=pickle.HIGHEST_PROTOCOL),
                dtype=np.uint8,
            ).copy()
            rng_state_blob = np.frombuffer(
                pickle.dumps(env.unwrapped._episode_rng.get_state(), protocol=pickle.HIGHEST_PROTOCOL),
                dtype=np.uint8,
            ).copy()
            row = trace[step]
            state_id = f"{task}__seed{seed:03d}__step{step:04d}"
            state_dir = outroot / "states" / task
            state_dir.mkdir(parents=True, exist_ok=True)
            arr_path = state_dir / f"{state_id}.npz"
            meta_path = state_dir / f"{state_id}.json"
            tmp_path = arr_path.with_name(f".{arr_path.name}.{worker}.tmp.npz")
            np.savez_compressed(
                tmp_path, rgb=rgb, segmentation_instance=seg, sim_state=sim_state,
                eef_pos=eef_pos, agent_state_blob=agent_state_blob,
                episode_rng_blob=rng_state_blob,
                elapsed_steps=np.asarray(int(env.unwrapped._elapsed_steps), dtype=np.int64),
            )
            tmp_path.replace(arr_path)
            meta = {
                "protocol_id": "OPENVLA_L11_ABC_STATE_LIBRARY_V1",
                "task": task, "seed": seed, "step": step,
                "phase_quartile": len(states), "episode_success": bool(summary["success"]),
                "instruction": instruction, "environment_id": env_id,
                "renderer": summary.get("rendering"), "gpu_id": gpu,
                "canonical_snapshot_sha256": summary["canonical_snapshot_sha256"],
                "initial_state_sha256": state_hash, "initial_rgb_sha256": rgb_hash,
                "state_rgb_sha256": sha_array(rgb), "segmentation_sha256": sha_array(seg),
                "rgb_shape": list(rgb.shape), "segmentation_shape": list(seg.shape),
                "m_matched": int(row["m_t"]),
                "selected_token_ids": list(row.get("selected_token_ids", [])),
                "l11_score_sha256": row.get("l11_score_sha256", row.get("attention_sha256")),
                "target_entities": ids,
                "target_entity_poses": entity_poses(env, ids),
                "arrays_file": arr_path.name,
                "handle_segmentation_limitation": (
                    "Drawer handle belongs to the drawer link mesh and is not a separate SAPIEN instance ID."
                    if task in {"google_robot_open_drawer", "google_robot_close_drawer"} else None
                ),
            }
            meta_tmp = meta_path.with_name(f".{meta_path.name}.{worker}.tmp")
            meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            meta_tmp.replace(meta_path)
            states.append(step)
        obs, _reward, _terminated, _truncated, final_info = env.step(action)
    if wanted != set(sample_indices) or len(states) != len(sample_indices):
        raise RuntimeError(f"sample capture incomplete {task}:{seed}")
    replay_success = bool(final_info.get("success", False)) if final_info else False
    return {"task": task, "seed": seed, "steps": len(actions),
            "sampled_steps": states, "stored_success": bool(summary["success"]),
            "replay_success": replay_success,
            "replay_success_match": replay_success == bool(summary["success"]),
            "initial_state_match": True, "initial_rgb_match": True}


def parse_job(spec: str) -> tuple[str, int, int]:
    task, lo, hi = spec.split(":")
    if task not in TASKS or int(lo) > int(hi):
        raise ValueError(f"invalid job {spec!r}")
    return task, int(lo), int(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", nargs="+", required=True, help="task:seed_start:seed_end blocks")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a seed only when all four state RGB/segmentation and metadata files exist")
    args = ap.parse_args()
    # Import the locked runner before environment creation; it sets RT depth 8
    # for drawer tasks and RT depth 0 for Pick/Move.
    from research.semantic_token_cd import l11_table2_openvla_rollout as _locked_runner  # noqa: F401
    from research.semantic_token_cd.prompt_attn_l11_budget_causal_rollout import make_environment
    from research.semantic_token_cd.distractor_rollout import restore_snapshot

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    env = None
    current_task = None
    current_env_id = None
    try:
      for spec in args.jobs:
        task, lo, hi = parse_job(spec)
        if task != current_task:
            if env is not None:
                env.close()
            env, current_env_id = make_environment(task)
            current_task = task
        for seed in range(lo, hi + 1):
            if args.skip_existing:
                state_dir = args.out / "states" / task
                complete = True
                for path in state_dir.glob(f"{task}__seed{seed:03d}__step*.npz"):
                    meta = path.with_suffix(".json")
                    if not meta.is_file():
                        complete = False
                        break
                    try:
                        with np.load(path, allow_pickle=False) as saved:
                            if not {"sim_state", "eef_pos", "agent_state_blob", "episode_rng_blob"}.issubset(saved.files):
                                complete = False
                                break
                    except Exception:
                        complete = False
                        break
                matches = list(state_dir.glob(f"{task}__seed{seed:03d}__step*.npz"))
                if complete and len(matches) == 4:
                    print(json.dumps({"event": "skip_complete", "task": task,
                                      "seed": seed, "worker": args.worker}), flush=True)
                    continue
            print(json.dumps({"event": "start", "task": task, "seed": seed,
                              "gpu": args.gpu, "worker": args.worker}), flush=True)
            record = process_one(task, seed, args.gpu, args.worker, args.out,
                                 env, current_env_id, restore_snapshot)
            records.append(record)
            if seed == lo:
                print(json.dumps({"event": "replay_audit", **record}), flush=True)
            if len(records) % 5 == 0:
                print(json.dumps({"event": "progress", "worker": args.worker,
                                  "completed_episodes": len(records)}), flush=True)
    finally:
        if env is not None:
            env.close()
    audit_path = args.out / "worker_audits" / f"{args.worker}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({"worker": args.worker, "gpu": args.gpu,
                                      "completed_episode_count": len(records),
                                      "episodes": records,
                                      "all_sample_replays_match": all(r["replay_success_match"] for r in records)},
                                     indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "worker_complete", "worker": args.worker,
                      "gpu": args.gpu, "episodes": len(records)}), flush=True)


if __name__ == "__main__":
    main()
