"""Shared protocol for the simulator-annotated drawer cross-instruction study.

Primary question: does the effect of a fixed simulator-annotated drawer region
change when that drawer is (or is not) the current instruction target?

This protocol deliberately replaces the attention/KMeans selector with a
simulator 3D -> camera -> 16x16 token annotation.  It is a mechanism research
tool, NOT an SHR performance evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
N_VISUAL = 256
PROTOCOL = "PROMPT_SIMREGION_OPEN_TOP_MID_V1"

DRAWERS = ("top", "middle")
ENV_BY_DRAWER = {
    "top": "OpenTopDrawerCustomInScene-v0",
    "middle": "OpenMiddleDrawerCustomInScene-v0",
}
TASK_BY_DRAWER = {"top": "google_robot_open_top_drawer", "middle": "google_robot_open_middle_drawer"}
INSTRUCTION_BY_DRAWER = {"top": "open top drawer", "middle": "open middle drawer"}
SEEDS = list(range(100, 120))          # 20 scenes
SMOKE_SEEDS = [100, 101, 102]
CONTACT_DIST_M = 0.06                  # eef-center -> handle-center proxy threshold
OPEN_QPOS_M = 0.15

# Token grid: 16x16 over the 512x640 overhead frame (each cell 32x40 px).
# The camera is attached to a robot link that can move between episodes, so the
# extrinsic matrix is NEVER hard-coded: it is read from the current observation
# camera_param (see overhead_camera()) so that regions always line up with the
# exact frame the policy sees.
ROW_PX = 32.0
COL_PX = 40.0
CELL_AREA = int(ROW_PX * COL_PX)  # 1280

# mk_station_recolor URDF drawer front panel box (same for top/middle drawers):
# center local x=0.305 (0.3425 for handle0), size 0.015 x 0.615 x 0.140.
DRAWER_FRONT_CENTER_LOCAL = np.array([0.3050, 0.0, 0.0], dtype=np.float64)
DRAWER_FRONT_HALF_LOCAL = np.array([0.0075, 0.3075, 0.07], dtype=np.float64)
# A candidate cell must contain at least this fraction of its area as own-drawer
# pixels; a cell where the *neighbor* drawer owns more than this fraction (and
# more than its own-drawer pixels) is trimmed so the region does not visibly
# bleed into the adjacent drawer.
OWN_CELL_MIN_FRAC = 0.04
NEIGHBOR_CELL_MAX_FRAC = 0.25


def array_sha(value: np.ndarray) -> str:
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
    if hasattr(value, "p") and hasattr(value, "q"):
        return type(value)(np.asarray(value.p).copy(), np.asarray(value.q).copy())
    import copy as _copy
    return _copy.deepcopy(value)


def q_rotate(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[0]
    u = q[1:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def overhead_camera(obs) -> dict:
    """Read the overhead camera intrinsic/extrinsic from the current obs."""
    cam = obs["camera_param"]["overhead_camera"]
    return {
        "K": np.asarray(cam["intrinsic_cv"], dtype=np.float64),
        "W2C": np.asarray(cam["extrinsic_cv"], dtype=np.float64),
    }


def project(p_world, cam):
    K = cam["K"]
    W2C = cam["W2C"]
    p = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
    c = W2C @ p
    u = K[0, 0] * c[0] / c[2] + K[0, 2]
    v = K[1, 1] * c[1] / c[2] + K[1, 2]
    return float(u), float(v)


def tok_cell(u, v):
    return int(v // ROW_PX), int(u // COL_PX)


def link_for(inner, drawer_id: str):
    from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
    link = get_entity_by_name(inner.art_obj.get_links(), f"{drawer_id}_drawer")
    if link is None:
        raise RuntimeError(f"missing drawer link {drawer_id}_drawer")
    return link


def drawer_joint_index(inner, drawer_id: str) -> int:
    name = f"{drawer_id}_drawer_joint"
    try:
        return inner.joint_names.index(name)
    except ValueError as exc:
        raise RuntimeError(f"missing drawer joint {name}") from exc


def visual_front_handle_local(link):
    centers = {}
    for vb in link.get_visual_bodies():
        nm = getattr(vb, "name", "")
        if nm in ("handle0", "handle1", "handle2", "drawer_front"):
            centers[nm] = np.asarray(vb.local_pose.p, dtype=np.float64)
    return centers


def drawer_geometry(inner, drawer_id: str) -> dict:
    """Current drawer link poses, handle/front world points, and joint qpos."""
    link = link_for(inner, drawer_id)
    pose = link.pose
    q = np.asarray(pose.q, dtype=np.float64)
    p = np.asarray(pose.p, dtype=np.float64)
    local = visual_front_handle_local(link)
    hp = None if local.get("handle0") is None else q_rotate(q, local["handle0"]) + p
    fp = None if local.get("drawer_front") is None else q_rotate(q, local["drawer_front"]) + p
    handles = [
        q_rotate(q, local[nm]) + p
        for nm in ("handle0", "handle1", "handle2") if local.get(nm) is not None
    ]
    joint = drawer_joint_index(inner, drawer_id)
    qpos = float(np.asarray(inner.art_obj.get_qpos())[joint])
    return {
        "drawer_id": drawer_id,
        "link_p": p.tolist(),
        "link_q": q.tolist(),
        "handle_p": None if hp is None else hp.tolist(),
        "handle_points": [h.tolist() for h in handles],
        "front_p": None if fp is None else fp.tolist(),
        "joint_qpos": qpos,
        "joint_index": int(joint),
    }


def _candidate_cells(geo: dict, cam: dict) -> list[tuple[int, int]]:
    """Cells hit by the drawer front panel box and its handle bars.

    The front panel is a box centred at the world front point with half extents
    (x,y,z) = DRAWER_FRONT_HALF_LOCAL in the drawer link frame.  Project its
    eight corners plus the handle-bar centres through the *current frame*
    camera; the region is the bounding token-cell rectangle.  Pure geometry:
    never conditioned on the instruction.
    """
    q = np.asarray(geo["link_q"], dtype=np.float64)
    p = np.asarray(geo["link_p"], dtype=np.float64)
    fp = np.asarray(geo["front_p"], dtype=np.float64)
    ax = [q_rotate(q, np.eye(3)[i]) for i in range(3)]
    cells = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = DRAWER_FRONT_CENTER_LOCAL + DRAWER_FRONT_HALF_LOCAL * np.array([sx, sy, sz])
                corner = q_rotate(q, local) + p
                r, c = tok_cell(*project(corner, cam))
                cells.append((r, c))
    for hp in geo.get("handle_points", []):
        r, c = tok_cell(*project(hp, cam))
        cells.append((r, c))
    rows = [max(0, min(15, r)) for r, _ in cells]
    cols = [max(0, min(15, c)) for _, c in cells]
    out = []
    for r in range(min(rows), max(rows) + 1):
        for c in range(min(cols), max(cols) + 1):
            out.append((r, c))
    return out


def _link_actor_id(inner, drawer_id: str) -> int:
    return link_for(inner, drawer_id).id


def _trim_cells_by_segmentation(inner, drawer_id: str, obs, cells, cam) -> tuple[list, dict]:
    """Drop candidate cells whose visible pixels mostly belong to the neighbour.

    Keeps a cell only when it contains a meaningful share of the target drawer
    and the *other* drawer does not dominate it.  Segmentation comes from the
    current frame, so masks are identical for both instructions (the physical
    frame is shared).  Returns (kept_cells, quality) where quality records the
    own/neighbour pixel content used for later offline audits.
    """
    seg = np.asarray(obs["image"]["overhead_camera"]["Segmentation"][..., 1], dtype=np.int32)
    own_id = _link_actor_id(inner, drawer_id)
    other_id = _link_actor_id(inner, other_drawer(drawer_id))
    kept = []
    counts = {"own_px": 0, "neighbor_px": 0, "region_px": 0}
    for r, c in cells:
        cell = seg[r * int(ROW_PX):(r + 1) * int(ROW_PX), c * int(COL_PX):(c + 1) * int(COL_PX)]
        own = int((cell == own_id).sum())
        neighbor = int((cell == other_id).sum())
        counts["own_px"] += own
        counts["neighbor_px"] += neighbor
        counts["region_px"] += int(CELL_AREA)
        if own >= OWN_CELL_MIN_FRAC * CELL_AREA and neighbor <= max(own, NEIGHBOR_CELL_MAX_FRAC * CELL_AREA):
            kept.append((r, c))
    if not kept:
        # Safety: never return an empty region on a drawer that is on screen.
        kept = [(r, c) for r, c in cells
                if (seg[r * int(ROW_PX):(r + 1) * int(ROW_PX), c * int(COL_PX):(c + 1) * int(COL_PX)] == own_id).sum() > 0]
    if not kept:
        kept = list(cells)
    return kept, counts


def other_drawer(drawer_id: str) -> str:
    return "middle" if drawer_id == "top" else "top"


def region_from_cells(geo: dict, cells: list, counts: dict | None = None) -> dict:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = max(0, min(rows)), min(15, max(rows))
    c0, c1 = max(0, min(cols)), min(15, max(cols))
    tokens = sorted({r * 16 + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)})
    quality = counts or {"own_px": 0, "neighbor_px": 0, "region_px": 0}
    return {
        "drawer_id": geo["drawer_id"],
        "row_span": [r0, r1],
        "col_span": [c0, c1],
        "token_ids": tokens,
        "joint_qpos": geo["joint_qpos"],
        "front_p": geo["front_p"],
        "handle_p": geo["handle_p"],
        **quality,
    }


def drawer_regions(inner, obs) -> dict:
    """Per-frame simulator-annotated regions for both drawers.

    Requires the current observation (camera parameters + segmentation) so that
    the projected cells align exactly with the frame that the policy sees.
    """
    cam = overhead_camera(obs)
    geos = {d: drawer_geometry(inner, d) for d in DRAWERS}
    regs = {}
    for d in DRAWERS:
        candidates = _candidate_cells(geos[d], cam)
        kept, counts = _trim_cells_by_segmentation(inner, d, obs, candidates, cam)
        regs[d] = region_from_cells(geos[d], kept, counts)
    top = set(regs["top"]["token_ids"])
    mid = set(regs["middle"]["token_ids"])
    regs["__sizes__"] = {"top": len(top), "middle": len(mid)}
    regs["__overlap_top_middle__"] = len(top & mid)
    regs["__overlap_ratio_top__"] = len(top & mid) / max(1, len(top))
    regs["__overlap_ratio_middle__"] = len(top & mid) / max(1, len(mid))
    regs["__geometry__"] = geos
    return regs


def region_aux(regs: dict, label: str) -> dict:
    return {
        "region_size_top": regs["__sizes__"]["top"],
        "region_size_middle": regs["__sizes__"]["middle"],
        "region_overlap_top_middle": regs["__overlap_top_middle__"],
        "region_overlap_ratio_top": regs["__overlap_ratio_top__"],
        "region_overlap_ratio_middle": regs["__overlap_ratio_middle__"],
        "region_label": label,
        "row_span": regs[label]["row_span"],
        "col_span": regs[label]["col_span"],
    }


def make_drawer_env(drawer_id: str, gpu: int):
    import gymnasium as gym
    import mani_skill2_real2sim.envs  # noqa: F401
    import simpler_env  # noqa: F401
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    env_id = ENV_BY_DRAWER[drawer_id]
    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    return gym.make(env_id, obs_mode="rgbd", prepackaged_config=True,
                    renderer_kwargs=renderer_kwargs), env_id


def get_image(env):
    from parallel_inference import get_image_from_maniskill2_obs_dict
    obs = env.unwrapped.get_obs() if hasattr(env, "unwrapped") else None
    return None  # replaced below when wrapper is used


def capture_snapshot(env, seed):
    env.reset(seed=seed)
    inner = env.unwrapped
    return {
        "sim_state": np.asarray(inner.get_state()).copy(),
        "agent_state": clone(inner.agent.get_state()),
        "rng_state": clone(inner._episode_rng.get_state()),
        "instruction": inner.get_language_instruction(),
    }


def restore_snapshot(env, seed, snapshot):
    env.reset(seed=seed)
    inner = env.unwrapped
    inner.set_state(snapshot["sim_state"].copy())
    inner.agent.set_state(clone(snapshot["agent_state"]))
    inner._episode_rng.set_state(clone(snapshot["rng_state"]))
    inner._elapsed_steps = 0
    obs = _wrapped_observation(env)
    return obs


def _wrapped_observation(env):
    from research.semantic_token_cd.rollout_pilot import wrapped_observation
    return wrapped_observation(env)


def snapshot_sha(snapshot):
    d = hashlib.sha256()
    d.update(array_sha(snapshot["sim_state"]).encode())
    d.update(repr(snapshot["agent_state"]).encode())
    d.update(repr(snapshot["rng_state"]).encode())
    d.update(snapshot["instruction"].encode())
    return d.hexdigest()


def rgb_sha(env, obs):
    from parallel_inference import get_image_from_maniskill2_obs_dict
    img = get_image_from_maniskill2_obs_dict(env, obs)
    return array_sha(np.asarray(img, dtype=np.uint8))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_snapshot(root: Path, drawer_id: str, seed: int) -> dict:
    path = root / "snapshots" / TASK_BY_DRAWER[drawer_id] / f"seed_{seed:03d}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)
