"""Targeted close_drawer replay — gripper trajectory + handle/panel geometry.

Replays the RECORDED executed_actions (no model, no torch) from the exact per-seed
snapshot to answer the mechanism question behind the Phase 1B reversal:

    why does semantic_attn_k8_l8_15 (block Action Query -> selected handle tokens)
    RESCUE close_drawer, while semantic_merge_k8_eta100 (collapse handle tokens to a
    prototype) HARM it?

Two competing causal paths (user's hypothesis):
  (a) attn-blocking removes *misleading* handle evidence  -> the gripper AIMs the
      handle/panel more accurately (better approach angle, still a grasp/pull);
  (b) attn-blocking forces a *push-the-panel* strategy     -> the gripper stops
      reaching for the handle and instead presses the front face shut;
  (c) merge-collapse misaligns the fine approach          -> grasp/push angle off by
      a few cm, leaving the drawer ajar.

Recorded per step (t): tcp pose (p,q), target-drawer qpos, handle world pos, front
panel world pos, gripper action dim, full 7-dim action, robot qpos. The grasp-vs-push
discriminator is computed offline from gripper action + tcp-vs-handle/panel + qpos
timing.

Crash-resilient + resumable: writes one seed_XXX.json per seed atomically; a seed
already written is skipped, so the driver can be re-run until every seed is done.
(State_dict obs_mode avoids camera rendering; the SAPIEN physics loop occasionally
hits an intermittent heap-corruption crash, so resume-on-restart is essential.)

Usage:
  <venv>/bin/python research/semantic_token_cd/replay_close_drawer.py \
    --gpu 1 --out artifacts/attn_semantic_merge_k8_v1/replay_close_drawer
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
for _p in (str(REPO_ROOT), str(PCD_SOURCE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK = "google_robot_close_drawer"
REF_ARTIFACT = REPO_ROOT / "artifacts/attn_global_merge_v1"
ARTIFACT = REPO_ROOT / "artifacts/attn_semantic_merge_k8_v1"
VANILLA_ARM = "vanilla"
ATTN_ARM = "semantic_attn_k8_l8_15"
MERGE_ARM = "semantic_merge_k8_eta100"

ATTN_RESCUE_SEEDS = [200, 210, 211, 226, 236, 248, 253, 258, 260, 270, 277, 286, 287, 294]
MERGE_HARM_SEEDS = [205, 214, 215, 221, 223, 233, 241, 271, 281, 297]
BOTH_SUCC_CONTROLS = [201, 213, 220, 222, 224, 227]
BOTH_FAIL_CONTROLS = [202, 206, 207, 209, 216, 217]


def array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    d = hashlib.sha256()
    d.update(str(a.dtype).encode("ascii"))
    d.update(str(tuple(a.shape)).encode("ascii"))
    d.update(a.tobytes())
    return d.hexdigest()


def clone(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return type(value)((k, clone(v)) for k, v in value.items())
    if isinstance(value, list):
        return [clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone(v) for v in value)
    if hasattr(value, "p") and hasattr(value, "q"):  # RandomState
        return type(value)(np.asarray(value.p).copy(), np.asarray(value.q).copy())
    return copy.deepcopy(value)


def capture_snapshot(env, seed):
    env.reset(seed=seed)
    inner = env.unwrapped
    return {
        "sim_state": np.asarray(inner.get_state()).copy(),
        "agent_state": clone(inner.agent.get_state()),
        "rng_state": clone(inner._episode_rng.get_state()),
        "instruction": inner.get_language_instruction(),
    }


def snapshot_sha(snapshot):
    d = hashlib.sha256()
    d.update(array_sha256(snapshot["sim_state"]).encode())
    d.update(repr(snapshot["agent_state"]).encode())
    d.update(repr(snapshot["rng_state"]).encode())
    d.update(snapshot["instruction"].encode())
    return d.hexdigest()


def restore_snapshot(env, seed, snapshot):
    env.reset(seed=seed)
    inner = env.unwrapped
    inner.set_state(snapshot["sim_state"].copy())
    inner.agent.set_state(clone(snapshot["agent_state"]))
    inner._episode_rng.set_state(clone(snapshot["rng_state"]))
    inner._elapsed_steps = 0
    return inner


def q_rotate(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[0]
    u = q[1:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def handle_front_local(link):
    """Constant link-frame positions of handle0 and the drawer front panel."""
    h = f = None
    for vb in link.get_visual_bodies():
        nm = getattr(vb, "name", "")
        if nm == "handle0":
            h = np.asarray(vb.local_pose.p, dtype=np.float64)
        elif nm == "drawer_front":
            f = np.asarray(vb.local_pose.p, dtype=np.float64)
    return h, f


def load_arrays(arm, seed):
    art = REF_ARTIFACT if arm == VANILLA_ARM else ARTIFACT
    p = art / "episodes" / TASK / arm / f"episode_{seed:03d}_arrays.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)


def load_canonical(arm, seed):
    art = REF_ARTIFACT if arm == VANILLA_ARM else ARTIFACT
    p = art / "episodes" / TASK / arm / f"episode_{seed:03d}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("canonical_snapshot_sha256")


def replay_arm(env, inner, seed, snapshot, act, h_local, f_local):
    restore_snapshot(env, seed, snapshot)
    target_joint = inner.joint_idx
    link = inner.drawer_obj
    T = act.shape[0]
    recs = []
    for t in range(T):
        link_pose = link.pose
        hp = None if h_local is None else (q_rotate(link_pose.q, h_local) + np.asarray(link_pose.p))
        fp = None if f_local is None else (q_rotate(link_pose.q, f_local) + np.asarray(link_pose.p))
        recs.append({
            "t": t,
            "tcp_p": np.asarray(inner.tcp.pose.p, dtype=np.float64).tolist(),
            "tcp_q": np.asarray(inner.tcp.pose.q, dtype=np.float64).tolist(),
            "drawer_qpos": float(np.asarray(inner.art_obj.get_qpos())[target_joint]),
            "handle_p": None if hp is None else hp.tolist(),
            "front_p": None if fp is None else fp.tolist(),
            "gripper_action": float(act[t][6]),
            "act": act[t].astype(np.float64).tolist(),
            "robot_qpos": np.asarray(inner.agent.robot.get_qpos(), dtype=np.float64).tolist(),
        })
        a = act[t].astype(np.float64)
        if not np.isfinite(a).all():
            a = np.nan_to_num(a)
        _obs, _r, _terminated, truncated, _info = env.step(a)
        if truncated:
            break
    return recs


def build_env():
    import gymnasium as gym
    import simpler_env  # registers the -v0 envs  # noqa: F401
    for attempt in range(12):
        try:
            return gym.make("CloseDrawerCustomInScene-v0",
                            obs_mode="state_dict", prepackaged_config=True)
        except Exception as e:
            print(f"[warn] env make attempt {attempt} failed: {e!r}", flush=True)
    raise RuntimeError("could not build env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ARTIFACT / "replay_close_drawer")
    ap.add_argument("--seeds", default="")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    out_dir = a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.seeds.strip():
        seeds = [int(x) for x in a.seeds.split(",") if x]
    else:
        seeds = sorted(set(ATTN_RESCUE_SEEDS + MERGE_HARM_SEEDS
                           + BOTH_SUCC_CONTROLS + BOTH_FAIL_CONTROLS))

    arms = [VANILLA_ARM, ATTN_ARM, MERGE_ARM]

    for seed in seeds:
        seed_file = out_dir / f"seed_{seed:03d}.json"
        if seed_file.exists():
            continue  # already done (resume)

        env = build_env()
        inner = env.unwrapped
        try:
            snapshot = capture_snapshot(env, seed)
            canonical = snapshot_sha(snapshot)
            stored = load_canonical(VANILLA_ARM, seed)
            warnings = []
            if stored is not None and stored != canonical:
                warnings.append(f"canonical mismatch vs stored vanilla {stored[:12]}...")

            # cache constant link-frame handle/front once
            h_local, f_local = handle_front_local(inner.drawer_obj)
            out = {"seed": seed, "canonical": canonical,
                   "warnings": warnings, "arms": {}}
            for arm in arms:
                arr = load_arrays(arm, seed)
                if arr is None:
                    out["arms"][arm] = {"error": "missing arrays"}
                    continue
                act = arr["executed_actions"]
                recs = replay_arm(env, inner, seed, snapshot, act, h_local, f_local)
                h = recs[0]["handle_p"]
                front = recs[0]["front_p"]

                def _d(p, q):
                    return None if (p is None or q is None) else float(
                        np.linalg.norm(np.array(p) - np.array(q)))

                qpos = [r["drawer_qpos"] for r in recs]
                gripper = [r["gripper_action"] for r in recs]
                q0 = qpos[0]
                contact_t = None
                for r in recs:
                    if q0 - r["drawer_qpos"] > 0.01:
                        contact_t = r["t"]
                        break
                out["arms"][arm] = {
                    "n_steps": len(recs),
                    "qpos_initial": q0, "qpos_final": qpos[-1], "qpos_min": min(qpos),
                    "contact_t": contact_t,
                    "tcp_at_contact": recs[contact_t]["tcp_p"] if contact_t is not None else None,
                    "gripper_at_contact": gripper[contact_t] if contact_t is not None else None,
                    "handle_p": h, "front_p": front,
                    "n_gripper_close": sum(1 for g in gripper if g < -0.5),
                    "n_gripper_open": sum(1 for g in gripper if g > 0.5),
                    "traj": recs,
                }
                print(json.dumps({"seed": seed, "arm": arm, "n": len(recs),
                                  "q0": round(q0, 3), "qT": round(qpos[-1], 3),
                                  "contact_t": contact_t,
                                  "n_close": out["arms"][arm]["n_gripper_close"]}), flush=True)

            tmp = seed_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
            tmp.rename(seed_file)
        finally:
            try:
                env.close()
            except Exception:
                pass

    done = sorted(p.stem for p in out_dir.glob("seed_*.json"))
    print(json.dumps({"DONE": True, "n_done": len(done), "n_target": len(seeds),
                      "out": str(out_dir)}), flush=True)


if __name__ == "__main__":
    main()
