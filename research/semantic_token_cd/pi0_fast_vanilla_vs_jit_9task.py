"""Paired Vanilla vs fixed-header JIT L6-Matched rollouts on SIMPLER's nine tasks."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

TOKEN_CD = Path(__file__).resolve().parents[2]
if str(TOKEN_CD) not in sys.path:
    sys.path.insert(0, str(TOKEN_CD))

from research.semantic_token_cd import pi0_fast_simpler_l6_matched_rollout as base

ROOT = base.ROOT
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
VANILLA = "vanilla"
GUIDED = "l6_jit_lambda050"
LAMBDA = 0.5


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(tmp, path)


class PairedPolicy(base.RolloutPolicy):
    """Use the same WebSocket service for both arms; guided calls explicitly request JIT."""

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.guided_fallbacks = 0

    def chunk(self, arm: str, raw: dict, seed: int):
        if arm == VANILLA:
            request = {**raw, "cd_arm": "vanilla", "cd_seed": int(seed)}
            started = time.perf_counter()
            result = self.client.infer(request)
            elapsed = time.perf_counter() - started
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid Vanilla action chunk: {actions.shape}")
            return actions, {
                "policy_call_seconds": elapsed,
                "action_chunk_sha256": sha256_array(actions),
                "arm": VANILLA,
            }

        if arm != GUIDED:
            raise ValueError(f"unknown arm: {arm}")
        request = {
            **raw,
            "cd_arm": "l6_matched",
            "cd_seed": int(seed),
            "cd_lambda": LAMBDA,
            "cd_jit_header": 1,
        }
        started = time.perf_counter()
        result = self.client.infer(request)
        guided_elapsed = time.perf_counter() - started
        actions = np.asarray(result["actions"], dtype=np.float32)
        trace = dict(result.get("pi0_fast_matched_cd_trace", {}))
        meta = result.get("pi0_fast_server_meta", {})
        if meta.get("jit_header") is not True or trace.get("jit_header") is not True:
            raise RuntimeError(f"JIT fixed-header mode was not active: {meta}")
        if meta.get("lambda") != LAMBDA or trace.get("lambda") != LAMBDA:
            raise RuntimeError(f"guidance lambda mismatch: {meta}")
        if trace.get("selector_layer_label") != "L6" or not trace.get("selected_count_equals_matched"):
            raise RuntimeError(f"L6-Matched invariant failed: {trace}")
        if not trace.get("fast_action_stream_valid"):
            raise RuntimeError(f"invalid FAST action stream: {trace}")
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid guided action chunk: {actions.shape}")

        # Preserve rollout continuity if OpenPI's generic decoder rejects a rare malformed
        # FAST payload; record the fallback explicitly in the episode trace.
        if not trace.get("openpi_official_decoder_agrees"):
            fallback_started = time.perf_counter()
            fallback = self.client.infer({**raw, "cd_arm": "vanilla", "cd_seed": int(seed)})
            fallback_elapsed = time.perf_counter() - fallback_started
            fallback_actions = np.asarray(fallback["actions"], dtype=np.float32)
            if fallback_actions.shape != (10, 7) or not np.isfinite(fallback_actions).all():
                raise RuntimeError(f"invalid Vanilla fallback chunk: {fallback_actions.shape}")
            self.guided_fallbacks += 1
            trace.update({
                "guided_fallback": True,
                "fallback_reason": "official_decoder_disagreement",
                "guided_call_seconds": guided_elapsed,
                "fallback_call_seconds": fallback_elapsed,
                "policy_call_seconds": guided_elapsed + fallback_elapsed,
                "action_chunk_sha256": sha256_array(fallback_actions),
            })
            return fallback_actions, trace

        trace.update({
            "guided_fallback": False,
            "policy_call_seconds": guided_elapsed,
            "action_chunk_sha256": sha256_array(actions),
        })
        return actions, trace


def parse_job(spec: str) -> tuple[str, int, int]:
    try:
        task, interval = spec.split(":", 1)
        start, end = (int(x) for x in interval.split("-", 1))
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError("job must be TASK:SEED_START-SEED_END") from exc
    if task not in TASKS or start < 0 or end > 99 or start > end:
        raise argparse.ArgumentTypeError(f"invalid job {spec!r}; seeds are 0..99")
    return task, start, end


def run_pair(policy: PairedPolicy, env, task: str, seed: int, out: Path, rt_depth: int,
             checkpoint: Path) -> None:
    order = (VANILLA, GUIDED) if seed % 2 == 0 else (GUIDED, VANILLA)
    paths = {
        arm: out / "episodes" / task / f"{arm}_seed{seed:03d}.json"
        for arm in (VANILLA, GUIDED)
    }
    records: dict[str, dict[str, Any]] = {}
    for arm in order:
        if paths[arm].is_file():
            try:
                prior = json.loads(paths[arm].read_text())
                if prior.get("status") == "complete" and prior.get("initial_rgb_lowres_b64"):
                    records[arm] = prior
                    continue
            except Exception:  # noqa: BLE001
                pass
        try:
            base._CHECKPOINT_FOR_LOG = str(checkpoint)
            records[arm] = base._episode(policy, env, task, seed, arm, out, rt_depth)
        except Exception as exc:  # noqa: BLE001
            error = {
                "status": "error", "task": task, "seed": seed, "arm": arm,
                "checkpoint": str(checkpoint), "renderer_depth": rt_depth,
                "error_type": type(exc).__name__, "error": repr(exc),
                "traceback": traceback.format_exc(),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            atomic_json(paths[arm], error)
            print(json.dumps({"event": "episode_error", **{k: error[k] for k in ("task", "seed", "arm", "error_type", "error")}}), flush=True)

    if all(arm in records and records[arm].get("status") == "complete" for arm in (VANILLA, GUIDED)):
        v, g = records[VANILLA], records[GUIDED]
        rgb_equal = v.get("initial_rgb_sha256") == g.get("initial_rgb_sha256")
        rgb_a = np.frombuffer(base64.b64decode(v["initial_rgb_lowres_b64"]), dtype=np.uint8)
        rgb_b = np.frombuffer(base64.b64decode(g["initial_rgb_lowres_b64"]), dtype=np.uint8)
        if rgb_a.shape != (56 * 56 * 3,) or rgb_b.shape != rgb_a.shape:
            raise RuntimeError(f"unexpected low-resolution RGB shape for {task}/{seed}")
        rgb_diff = np.abs(rgb_a.astype(np.int16) - rgb_b.astype(np.int16))
        rgb_mae = float(rgb_diff.mean())
        rgb_max = int(rgb_diff.max())
        rgb_coarse_equal = rgb_mae <= 0.05 and rgb_max <= 8
        state_equal = v.get("initial_agent_state_sha256") == g.get("initial_agent_state_sha256")
        if not rgb_coarse_equal or not state_equal:
            mismatch = {
                "task": task, "seed": seed,
                "checkpoint": str(checkpoint),
                "renderer_depth": rt_depth,
                "initial_rgb_equal": rgb_equal,
                "initial_rgb_coarse_equal": rgb_coarse_equal,
                "initial_rgb_lowres_mae": rgb_mae,
                "initial_rgb_lowres_max_abs_diff": rgb_max,
                "initial_state_equal": state_equal,
                "pair_excluded": True,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            atomic_json(out / "pair_mismatches" / task / f"seed{seed:03d}.json", mismatch)
            print("PAIR_MISMATCH " + json.dumps(mismatch), flush=True)
            return
        pair = {
            "task": task, "seed": seed,
            "renderer_depth": rt_depth,
            "checkpoint": str(checkpoint),
            "vanilla_success": bool(v["success"]),
            "guided_success": bool(g["success"]),
            "rescue": bool(not v["success"] and g["success"]),
            "harm": bool(v["success"] and not g["success"]),
            "initial_rgb_equal": rgb_equal,
            "initial_rgb_coarse_equal": rgb_coarse_equal,
            "initial_rgb_lowres_mae": rgb_mae,
            "initial_rgb_lowres_max_abs_diff": rgb_max,
            "initial_state_equal": state_equal,
            "vanilla_steps": v["steps"], "guided_steps": g["steps"],
            "vanilla_runtime_seconds": v["runtime_seconds"],
            "guided_runtime_seconds": g["runtime_seconds"],
            "guided_fallbacks": g.get("guided_fallbacks", 0),
            "guidance_lambda": LAMBDA,
            "guided_mode": "L6-Matched harmonic CD, lambda=0.5, fixed-header JIT",
        }
        atomic_json(out / "pairs" / task / f"seed{seed:03d}.json", pair)
        print("PAIR " + json.dumps(pair), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="renderer GPU (4..7)")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--family", choices=("google", "widowx"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rt-depth", type=int, default=8)
    parser.add_argument("--job", action="append", type=parse_job, required=True)
    args = parser.parse_args()
    if args.gpu not in (1, 2, 3, 4, 5, 6, 7):
        raise SystemExit(f"this run is restricted to GPUs 1..7, got {args.gpu}")
    if args.rt_depth != 8:
        raise SystemExit("protocol locked to RT-full depth 8")
    if not (args.out / "protocol.json").exists():
        raise SystemExit(f"protocol.json must be written before workers start: {args.out}")
    expected_prefix = "google_robot_" if args.family == "google" else "widowx_"
    if any(not task.startswith(expected_prefix) for task, _, _ in args.job):
        raise SystemExit(f"job task does not match checkpoint family {args.family}")
    checkpoint = ROOT / "checkpoints/openpi_fast_finetuned" / ("fractal" if args.family == "google" else "bridge")
    base._CHECKPOINT_FOR_LOG = str(checkpoint)
    policy = PairedPolicy("127.0.0.1", args.port)
    current_task = None
    env = None
    try:
        for task, start, end in args.job:
            if task != current_task:
                if env is not None:
                    env.close()
                env = base._environment(task, args.rt_depth)
                current_task = task
                print(json.dumps({"event": "environment_ready", "gpu": args.gpu, "task": task,
                                  "checkpoint": str(checkpoint), "rt_depth": args.rt_depth}), flush=True)
            for seed in range(start, end + 1):
                run_pair(policy, env, task, seed, args.out, args.rt_depth, checkpoint)
    finally:
        if env is not None:
            env.close()
    print(json.dumps({"event": "worker_complete", "gpu": args.gpu}), flush=True)


if __name__ == "__main__":
    main()
