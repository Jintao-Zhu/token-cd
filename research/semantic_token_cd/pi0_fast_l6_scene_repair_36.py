"""Canonical-scene repair for the 36 unmatched pi0-FAST lambda-scan seeds.

For each task/seed, capture one complete ManiSkill simulator state after reset,
then restore that state before every arm. This avoids treating repeated seeded
resets as identical when dynamic-object settling changes their final poses.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from research.semantic_token_cd import pi0_fast_l6_lambda125_150_9task as impl
from research.semantic_token_cd import pi0_fast_simpler_l6_matched_rollout as base

ROOT = Path("/home/leju-suzhou/zjt_ws")
OUT = ROOT / "token-cd/runs/pi0fast_l6_scene_repair_36seeds_rtfull_v1"
SOURCE_RUN = ROOT / "token-cd/runs/pi0fast_9task_l6jit_lambda125_150_rtfull_seed0_99_v1"
ARMS = ("vanilla", "l6_matched_050", "l6_matched_100", "l6_matched_125", "l6_matched_150")
LAMBDAS = {"l6_matched_050": 0.5, "l6_matched_100": 1.0,
           "l6_matched_125": 1.25, "l6_matched_150": 1.5}
TASKS = ("google_robot_move_near", "widowx_put_eggplant_in_basket")
RGB_MAE_MAX = 0.05
RGB_MAX_ABS_MAX = 8
RT_DEPTH = 8


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def info_identity(info: dict[str, Any]) -> dict[str, Any]:
    """Only compare discrete task identity; dynamic post-settle poses are snapshotted."""
    keys = ("episode_id", "episode_model_ids", "episode_model_scales",
            "episode_source_obj_name", "episode_target_obj_name")
    result = {key: info.get(key) for key in keys}
    for key, value in result.items():
        if isinstance(value, tuple):
            result[key] = list(value)
    return result


def fixture_paths(task: str, seed: int) -> tuple[Path, Path]:
    stem = OUT / "scene_fixtures" / task / f"seed{seed:03d}"
    return stem.with_suffix(".npy"), stem.with_suffix(".json")


def load_or_create_fixture(env, task: str, seed: int) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    state_path, meta_path = fixture_paths(task, seed)
    if state_path.is_file() and meta_path.is_file():
        state = np.load(state_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text())
        if meta.get("task") != task or int(meta.get("seed", -1)) != seed:
            raise RuntimeError(f"fixture identity mismatch: {task}/{seed}")
        return state, meta["reset_info"], meta["identity"]

    obs, reset_info = env.reset(seed=seed)
    del obs
    state = np.asarray(env.get_state()).copy()
    if state.ndim != 1 or not np.isfinite(state).all():
        raise RuntimeError(f"invalid simulator state for {task}/{seed}: {state.shape}")
    identity = info_identity(reset_info)
    if identity.get("episode_id") is None:
        raise RuntimeError(f"reset_info has no episode_id for {task}/{seed}: {identity}")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_state = state_path.with_suffix(f".npy.tmp.{os.getpid()}")
    with tmp_state.open("wb") as f:
        np.save(f, state, allow_pickle=False)
    os.replace(tmp_state, state_path)
    impl.atomic_json(meta_path, {
        "task": task, "seed": seed, "state_sha256": digest(state),
        "state_shape": list(state.shape), "identity": identity,
        "reset_info": reset_info, "renderer": impl.expected_renderer(),
    })
    return state, reset_info, identity


class SnapshotResetEnv:
    """Reset the task, then restore the same complete simulator state before rollout."""
    def __init__(self, env, state: np.ndarray, reset_info: dict[str, Any], identity: dict[str, Any]):
        self._env = env
        self._state = np.asarray(state).copy()
        self._reset_info = reset_info
        self._identity = identity
        self.last_reset_state_sha256 = digest(self._state)

    def reset(self, seed=None, options=None):
        _, fresh_info = self._env.reset(seed=seed, options=options)
        fresh_identity = info_identity(fresh_info)
        if fresh_identity != self._identity:
            raise RuntimeError(
                f"scene identity changed for seed={seed}: {fresh_identity} != {self._identity}"
            )
        fresh_state = np.asarray(self._env.get_state())
        if fresh_state.shape != self._state.shape:
            raise RuntimeError(
                f"sim-state shape changed for seed={seed}: {fresh_state.shape} != {self._state.shape}"
            )
        self._env.set_state(self._state.copy())
        restored = np.asarray(self._env.get_state())
        # SAPIEN's state setters round through float32 transforms even though the
        # packed state is float64.  The observed round-trip error is <1.2e-7.
        if not np.allclose(restored, self._state, rtol=0.0, atol=2e-7):
            max_diff = float(np.max(np.abs(restored - self._state)))
            raise RuntimeError(
                f"simulator state restore exceeded tolerance for seed={seed}: max_diff={max_diff}"
            )
        # The unwrapped ManiSkill get_obs() returns raw Color/Position textures;
        # apply the same RGBD wrapper used by reset() to recover uint8 rgb/depth.
        raw_obs = self._env.unwrapped.get_obs()
        wrapped_obs = self._env.observation(raw_obs)
        return wrapped_obs, self._reset_info

    def __getattr__(self, name):
        return getattr(self._env, name)


class RepairPolicy(base.RolloutPolicy):
    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.guided_fallbacks = 0

    def chunk(self, arm: str, raw: dict, seed: int):
        if arm == "vanilla":
            result = self.client.infer({**raw, "cd_arm": "vanilla", "cd_seed": int(seed)})
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid Vanilla action chunk: {actions.shape}")
            return actions, {"arm": "vanilla"}
        if arm not in LAMBDAS:
            raise ValueError(f"unsupported repair arm: {arm}")
        lam = LAMBDAS[arm]
        result = self.client.infer({
            **raw, "cd_arm": "l6_matched", "cd_seed": int(seed),
            "cd_lambda": lam, "cd_jit_header": 1,
        })
        actions = np.asarray(result["actions"], dtype=np.float32)
        trace = dict(result.get("pi0_fast_matched_cd_trace", {}))
        meta = dict(result.get("pi0_fast_server_meta", {}))
        if meta.get("lambda") != lam or trace.get("lambda") != lam:
            raise RuntimeError(f"lambda mismatch: requested={lam}, meta={meta}, trace={trace}")
        if meta.get("jit_header") is not True or trace.get("jit_header") is not True:
            raise RuntimeError(f"fixed-header JIT was not active: meta={meta}, trace={trace}")
        if trace.get("selector_layer_label") != "L6" or not trace.get("selected_count_equals_matched"):
            raise RuntimeError(f"L6-Matched invariant failed: {trace}")
        if not trace.get("fast_action_stream_valid"):
            raise RuntimeError(f"invalid FAST action stream: {trace}")
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid guided action chunk: {actions.shape}")
        if not trace.get("openpi_official_decoder_agrees"):
            fallback = self.client.infer({**raw, "cd_arm": "vanilla", "cd_seed": int(seed)})
            actions = np.asarray(fallback["actions"], dtype=np.float32)
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid Vanilla fallback action chunk: {actions.shape}")
            self.guided_fallbacks += 1
            trace["guided_fallback"] = True
        else:
            trace["guided_fallback"] = False
        return actions, trace


def rgb_diff(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, int, bool]:
    x = np.frombuffer(base64.b64decode(a["initial_rgb_lowres_b64"]), dtype=np.uint8)
    y = np.frombuffer(base64.b64decode(b["initial_rgb_lowres_b64"]), dtype=np.uint8)
    if x.shape != (56 * 56 * 3,) or y.shape != x.shape:
        raise RuntimeError("unexpected stored 56x56 RGB payload")
    d = np.abs(x.astype(np.int16) - y.astype(np.int16))
    mae, max_abs = float(d.mean()), int(d.max())
    return mae, max_abs, mae <= RGB_MAE_MAX and max_abs <= RGB_MAX_ABS_MAX


def preflight_one(policy, env, task: str, seed: int) -> dict[str, Any]:
    state, reset_info, identity = load_or_create_fixture(env, task, seed)
    proxy = SnapshotResetEnv(env, state, reset_info, identity)
    camera = "overhead_camera" if task.startswith("google_robot_") else "3rd_view_camera"
    baseline = None
    arm_results = {}
    image_hashes = []
    state_hashes = []
    for arm in ARMS:
        obs, _ = proxy.reset(seed=seed)
        image = np.asarray(obs["image"][camera]["rgb"])
        instruction = str(env.unwrapped.get_language_instruction())
        raw = base._raw_policy_input(image, instruction, obs["agent"], env)
        _, trace = policy.chunk(arm, raw, seed * 1009)
        low = base._lowres_rgb(image)
        current = {"rgb": base64.b64encode(low.tobytes()).decode("ascii"),
                   "rgb_sha256": digest(low),
                   "eef_sha256": digest(base._eef_pos(obs["agent"], env)),
                   "sim_sha256": digest(state)}
        if baseline is None:
            baseline = current
        else:
            x = np.frombuffer(base64.b64decode(baseline["rgb"]), dtype=np.uint8).astype(np.int16)
            y = np.frombuffer(base64.b64decode(current["rgb"]), dtype=np.uint8).astype(np.int16)
            d = np.abs(x - y)
            mae, max_abs = float(d.mean()), int(d.max())
            arm_results[arm] = {
                "rgb_mae_vs_vanilla": mae, "rgb_max_abs_diff_vs_vanilla": max_abs,
                "rgb_match": bool(mae <= RGB_MAE_MAX and max_abs <= RGB_MAX_ABS_MAX),
                "eef_match": current["eef_sha256"] == baseline["eef_sha256"],
                "sim_state_match": current["sim_sha256"] == baseline["sim_sha256"],
                "guidance_trace": {k: trace.get(k) for k in (
                    "lambda", "selector_layer_label", "selected_count_equals_matched",
                    "jit_header", "fast_action_stream_valid", "openpi_official_decoder_agrees",
                    "guided_fallback")},
            }
            if arm_results[arm]["rgb_match"] is False or not arm_results[arm]["eef_match"] or not arm_results[arm]["sim_state_match"]:
                raise RuntimeError(f"canonical-scene preflight failed for {task}/{seed}/{arm}: {arm_results[arm]}")
        image_hashes.append(current["rgb_sha256"])
        state_hashes.append(current["sim_sha256"])
    report = {
        "status": "passed", "task": task, "seed": seed, "identity": identity,
        "fixture_sha256": digest(state), "arms": arm_results,
        "all_arm_rgb_hashes_equal": len(set(image_hashes)) == 1,
        "all_arm_sim_state_hashes_equal": len(set(state_hashes)) == 1,
    }
    impl.atomic_json(OUT / "preflight" / task / f"seed{seed:03d}.json", report)
    print("PREFLIGHT " + json.dumps(report), flush=True)
    return report


def pair_seed(task: str, seed: int) -> dict[str, Any]:
    records = {}
    for arm in ARMS:
        path = OUT / "episodes" / task / f"{arm}_seed{seed:03d}.json"
        records[arm] = json.loads(path.read_text())
    v = records["vanilla"]
    arm_diagnostics = {}
    for arm in ARMS[1:]:
        ep = records[arm]
        mae, max_abs, rgb_ok = rgb_diff(v, ep)
        state_ok = (ep.get("initial_sim_state_sha256") == v.get("initial_sim_state_sha256"))
        arm_diagnostics[arm] = {
            "success": bool(ep["success"]), "valid_vs_vanilla": bool(rgb_ok and state_ok),
            "rgb_mae": mae, "rgb_max_abs_diff": max_abs,
            "sim_state_equal": state_ok,
        }
    pair = {
        "status": "paired" if all(x["valid_vs_vanilla"] for x in arm_diagnostics.values()) else "scene_mismatch",
        "task": task, "seed": seed, "fixture_sha256": v.get("initial_sim_state_sha256"),
        "success": {a: bool(records[a]["success"]) for a in ARMS},
        "arms": arm_diagnostics,
        "all_five_scene_matched": all(x["valid_vs_vanilla"] for x in arm_diagnostics.values()),
    }
    impl.atomic_json(OUT / "pairs" / task / f"seed{seed:03d}.json", pair)
    print("PAIR " + json.dumps(pair), flush=True)
    return pair


def run_full_seed(policy, env, task: str, seed: int, worker_gpu: int) -> dict[str, Any]:
    state, reset_info, identity = load_or_create_fixture(env, task, seed)
    proxy = SnapshotResetEnv(env, state, reset_info, identity)
    family = "google" if task.startswith("google_robot_") else "widowx"
    base._CHECKPOINT_FOR_LOG = str(impl.checkpoint_for(family))
    state_hash = digest(state)
    for arm in ARMS:
        output = OUT / "episodes" / task / f"{arm}_seed{seed:03d}.json"
        if output.is_file():
            candidate = json.loads(output.read_text())
            if candidate.get("status") == "complete" and candidate.get("initial_sim_state_sha256") == state_hash:
                continue
        result = base._episode(policy, proxy, task, seed, arm, OUT, RT_DEPTH)
        result_path = OUT / "episodes" / task / f"{arm}_seed{seed:03d}.json"
        saved = json.loads(result_path.read_text())
        saved.update({
            "repair_fixture_sha256": state_hash,
            "initial_sim_state_sha256": state_hash,
            "repair_worker_gpu": worker_gpu,
            "pairing_protocol": "reset then restore the same complete env.get_state() snapshot for every arm",
        })
        impl.atomic_json(result_path, saved)
    return pair_seed(task, seed)


def parse_seeds(raw: str) -> list[int]:
    values = sorted({int(x) for x in raw.split(",") if x.strip()})
    if not values or any(x < 0 or x > 99 for x in values):
        raise ValueError(f"invalid seed list {raw!r}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="render worker GPU (6 or 7)")
    parser.add_argument("--port", type=int, required=True, help="inference server port")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True, help="comma-separated seed IDs")
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    args = parser.parse_args()
    if args.gpu not in (6, 7):
        raise SystemExit(f"render worker GPU must be 6 or 7, got {args.gpu}")
    seeds = parse_seeds(args.seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    policy = RepairPolicy("127.0.0.1", args.port)
    env = base._environment(args.task, RT_DEPTH)
    try:
        if args.mode == "preflight":
            for seed in seeds:
                preflight_one(policy, env, args.task, seed)
        else:
            for seed in seeds:
                try:
                    run_full_seed(policy, env, args.task, seed, args.gpu)
                except BaseException as exc:
                    error = {
                        "status": "error", "task": args.task, "seed": seed,
                        "gpu": args.gpu, "error_type": type(exc).__name__,
                        "error": repr(exc), "traceback": traceback.format_exc(),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    impl.atomic_json(OUT / "errors" / args.task / f"seed{seed:03d}.json", error)
                    raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
