"""L6-Matched pi0-FAST lambda=1.0 rollout paired to frozen Vanilla references."""
from __future__ import annotations

import argparse
import base64
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

TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
)
GUIDED_ARM = "l6_matched_100"
LAMBDA = 1.0
RT_DEPTH = 8
RGB_MAE_MAX = 0.05
RGB_MAX_ABS_MAX = 8
STAGED_PROMPT_TASKS = {"google_robot_place_apple_in_closed_top_drawer"}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def parse_job(spec: str) -> tuple[str, int, int]:
    try:
        task, interval = spec.split(":", 1)
        start, end = (int(x) for x in interval.split("-", 1))
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError("job must be TASK:SEED_START-SEED_END") from exc
    if task not in TASKS or start < 0 or end > 99 or start > end:
        raise argparse.ArgumentTypeError(f"invalid job {spec!r}; seeds are 0..99")
    return task, start, end


def checkpoint_for(family: str) -> Path:
    name = "fractal" if family == "google" else "bridge"
    return (base.ROOT / "checkpoints/openpi_fast_finetuned" / name).resolve()


def ref_episode(reference: Path, task: str, seed: int, arm: str) -> dict[str, Any]:
    path = reference / "episodes" / task / f"{arm}_seed{seed:03d}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


class LambdaOneJITPolicy(base.RolloutPolicy):
    """Explicitly request lambda=1 and fixed-header JIT on every guided call."""

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.guided_fallbacks = 0

    def chunk(self, arm: str, raw: dict, seed: int):
        if arm != GUIDED_ARM:
            raise ValueError(f"this runner only supports {GUIDED_ARM}, got {arm}")
        result = self.client.infer({
            **raw, "cd_arm": "l6_matched", "cd_seed": int(seed),
            "cd_lambda": LAMBDA, "cd_jit_header": 1,
        })
        actions = np.asarray(result["actions"], dtype=np.float32)
        trace = dict(result.get("pi0_fast_matched_cd_trace", {}))
        meta = dict(result.get("pi0_fast_server_meta", {}))
        if meta.get("lambda") != LAMBDA or trace.get("lambda") != LAMBDA:
            raise RuntimeError(f"lambda mismatch: meta={meta}, trace={trace}")
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
            fb_actions = np.asarray(fallback["actions"], dtype=np.float32)
            if fb_actions.shape != (10, 7) or not np.isfinite(fb_actions).all():
                raise RuntimeError(f"invalid Vanilla fallback chunk: {fb_actions.shape}")
            self.guided_fallbacks += 1
            trace.update({
                "guided_fallback": True,
                "fallback_reason": "official_decoder_disagreement",
                "fallback_server_meta": fallback.get("pi0_fast_server_meta", {}),
            })
            return fb_actions, trace
        trace["guided_fallback"] = False
        return actions, trace


def expected_renderer() -> dict[str, Any]:
    return {
        "shader_dir": "rt", "rt_max_path_depth": RT_DEPTH,
        "rt_samples_per_pixel": 32, "rt_use_denoiser": True,
    }


def check_vanilla_reference(reference: Path, task: str, seed: int,
                            family: str) -> dict[str, Any]:
    vanilla = ref_episode(reference, task, seed, "vanilla")
    if vanilla.get("status") != "complete":
        raise RuntimeError(f"frozen Vanilla is incomplete: {task}/{seed}")
    if str(Path(vanilla.get("checkpoint", "")).resolve()) != str(checkpoint_for(family)):
        raise RuntimeError(f"Vanilla checkpoint mismatch: {task}/{seed}: {vanilla.get('checkpoint')}")
    if vanilla.get("renderer") != expected_renderer():
        raise RuntimeError(f"Vanilla renderer mismatch: {task}/{seed}: {vanilla.get('renderer')}")
    return vanilla


def preflight(policy: LambdaOneJITPolicy, env, reference: Path, out: Path,
              task: str, seed: int, family: str) -> None:
    vanilla = check_vanilla_reference(reference, task, seed, family)
    obs, reset_info = env.reset(seed=seed)
    instruction = str(env.unwrapped.get_language_instruction())
    camera = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"
    image = np.asarray(obs["image"][camera]["rgb"])
    state_sha = base._digest(base._eef_pos(obs["agent"], env))
    if state_sha != vanilla.get("initial_agent_state_sha256"):
        raise RuntimeError(f"initial agent state differs from frozen Vanilla: {task}/{seed}")
    reset_info_equal = (
        json.dumps(reset_info, sort_keys=True, default=str)
        == json.dumps(vanilla.get("reset_info"), sort_keys=True, default=str)
    )
    if not reset_info_equal:
        raise RuntimeError(f"reset scene metadata differs from frozen Vanilla: {task}/{seed}")
    prompt_equal = None if task in STAGED_PROMPT_TASKS else instruction == vanilla.get("instruction")
    if prompt_equal is False:
        raise RuntimeError(f"initial instruction differs from frozen Vanilla: {task}/{seed}")
    env_id = getattr(getattr(env, "spec", None), "id", None)
    if env_id != vanilla.get("environment_id"):
        raise RuntimeError(f"environment id differs from frozen Vanilla: {task}/{seed}: {env_id!r}")
    raw = base._raw_policy_input(image, instruction, obs["agent"], env)
    _, trace = policy.chunk(GUIDED_ARM, raw, seed * 1009)
    a = base._lowres_rgb(image).reshape(-1).astype(np.int16)
    b = np.frombuffer(base64.b64decode(vanilla["initial_rgb_lowres_b64"]), dtype=np.uint8).astype(np.int16)
    diff = np.abs(a - b)
    report = {
        "status": "passed", "task": task, "seed": seed, "family": family,
        "checkpoint": str(checkpoint_for(family)), "renderer": expected_renderer(),
        "initial_agent_state_equal": True,
        "initial_rgb_coarse_equal": bool(diff.mean() <= RGB_MAE_MAX and diff.max() <= RGB_MAX_ABS_MAX),
        "initial_rgb_lowres_mae": float(diff.mean()),
        "initial_rgb_lowres_max_abs_diff": int(diff.max()),
        "prompt_equal": prompt_equal,
        "initial_instruction": instruction,
        "prompt_check_note": (
            "Task changes its prompt by stage; frozen episode stores the terminal-stage prompt."
            if task in STAGED_PROMPT_TASKS else "Initial prompt equals frozen Vanilla."
        ),
        "reset_scene_metadata_equal": reset_info_equal,
        "environment_id_equal": True,
        "guidance_lambda": trace.get("lambda"),
        "selector_layer": trace.get("selector_layer_label"),
        "selected_count_equals_matched": trace.get("selected_count_equals_matched"),
        "jit_header": trace.get("jit_header"),
        "fast_action_stream_valid": trace.get("fast_action_stream_valid"),
        "official_decoder_agrees": trace.get("openpi_official_decoder_agrees"),
        "guided_fallback": trace.get("guided_fallback", False),
    }
    atomic_json(out / "preflight" / task / f"seed{seed:03d}.json", report)
    print("PREFLIGHT " + json.dumps(report), flush=True)


def pair_episode(out: Path, reference: Path, task: str, seed: int,
                 guided: dict[str, Any]) -> dict[str, Any]:
    family = "google" if task.startswith("google_robot_") else "widowx"
    vanilla = check_vanilla_reference(reference, task, seed, family)
    if guided.get("status") != "complete":
        raise RuntimeError(f"guided result incomplete: {task}/{seed}")
    a = np.frombuffer(base64.b64decode(vanilla["initial_rgb_lowres_b64"]), dtype=np.uint8)
    b = np.frombuffer(base64.b64decode(guided["initial_rgb_lowres_b64"]), dtype=np.uint8)
    if a.shape != (56 * 56 * 3,) or b.shape != a.shape:
        raise RuntimeError(f"unexpected initial RGB payload shape: {task}/{seed}")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    state_equal = vanilla.get("initial_agent_state_sha256") == guided.get("initial_agent_state_sha256")
    rgb_equal = vanilla.get("initial_rgb_sha256") == guided.get("initial_rgb_sha256")
    rgb_mae = float(diff.mean())
    rgb_max = int(diff.max())
    matched = state_equal and rgb_mae <= RGB_MAE_MAX and rgb_max <= RGB_MAX_ABS_MAX
    common = {
        "task": task, "seed": seed,
        "initial_state_equal": state_equal,
        "initial_rgb_equal": rgb_equal,
        "initial_rgb_coarse_equal": bool(rgb_mae <= RGB_MAE_MAX and rgb_max <= RGB_MAX_ABS_MAX),
        "initial_rgb_lowres_mae": rgb_mae,
        "initial_rgb_lowres_max_abs_diff": rgb_max,
        "pair_excluded": not matched,
    }
    if not matched:
        mismatch = {**common, "reason": "initial state or RGB failed the frozen pairing threshold"}
        atomic_json(out / "pair_mismatches" / task / f"seed{seed:03d}.json", mismatch)
        return mismatch
    old_pair_path = reference / "pairs" / task / f"seed{seed:03d}.json"
    old_pair = json.loads(old_pair_path.read_text()) if old_pair_path.is_file() else None
    pair = {
        **common, "status": "paired", "checkpoint": guided["checkpoint"],
        "renderer": guided["renderer"], "guidance_lambda": LAMBDA,
        "guided_mode": "L6-Matched harmonic CD, lambda=1.0, fixed-header JIT",
        "vanilla_success": bool(vanilla["success"]),
        "lambda100_success": bool(guided["success"]),
        "rescue": bool(not vanilla["success"] and guided["success"]),
        "harm": bool(vanilla["success"] and not guided["success"]),
        "vanilla_steps": vanilla["steps"], "lambda100_steps": guided["steps"],
        "vanilla_runtime_seconds": vanilla["runtime_seconds"],
        "lambda100_runtime_seconds": guided["runtime_seconds"],
        "guided_fallbacks": guided.get("guided_fallbacks", 0),
        "lambda050_success": old_pair.get("guided_success") if old_pair else None,
        "lambda050_pair_available": bool(old_pair),
    }
    atomic_json(out / "pairs" / task / f"seed{seed:03d}.json", pair)
    return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--family", choices=("google", "widowx"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--job", nargs="+", required=True)
    args = parser.parse_args()
    jobs = [parse_job(spec) for spec in args.job]
    if args.gpu not in (1, 2, 3):
        raise SystemExit(f"GPU {args.gpu} is not authorized for this run")
    if not (args.out / "protocol.json").is_file():
        raise SystemExit(f"missing protocol lock: {args.out / 'protocol.json'}")
    prefix = "google_robot_" if args.family == "google" else "widowx_"
    if any(not task.startswith(prefix) for task, _, _ in jobs):
        raise SystemExit(f"job does not match checkpoint family {args.family}")
    if args.mode == "preflight" and any(start != 0 or end != 0 for _, start, end in jobs):
        raise SystemExit("preflight jobs must use seed 0 only")
    checkpoint = checkpoint_for(args.family)
    base._CHECKPOINT_FOR_LOG = str(checkpoint)
    policy = LambdaOneJITPolicy("127.0.0.1", args.port)
    env = None
    current_task = None
    try:
        for task, start, end in jobs:
            if task != current_task:
                if env is not None:
                    env.close()
                env = base._environment(task, RT_DEPTH)
                current_task = task
            if args.mode == "preflight":
                preflight(policy, env, args.reference, args.out, task, start, args.family)
                continue
            for seed in range(start, end + 1):
                ep_path = args.out / "episodes" / task / f"{GUIDED_ARM}_seed{seed:03d}.json"
                try:
                    if ep_path.is_file():
                        existing = json.loads(ep_path.read_text())
                        if existing.get("status") == "complete" and existing.get("arm") == GUIDED_ARM:
                            guided = existing
                        else:
                            guided = base._episode(policy, env, task, seed, GUIDED_ARM, args.out, RT_DEPTH)
                    else:
                        guided = base._episode(policy, env, task, seed, GUIDED_ARM, args.out, RT_DEPTH)
                    pair = pair_episode(args.out, args.reference, task, seed, guided)
                    print("PAIR " + json.dumps(pair), flush=True)
                except Exception as exc:  # noqa: BLE001
                    error = {
                        "status": "error", "task": task, "seed": seed, "arm": GUIDED_ARM,
                        "checkpoint": str(checkpoint), "error_type": type(exc).__name__,
                        "error": repr(exc), "traceback": traceback.format_exc(),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    atomic_json(args.out / "errors" / task / f"seed{seed:03d}.json", error)
                    print("EPISODE_ERROR " + json.dumps(
                        {k: error[k] for k in ("task", "seed", "error_type", "error")}), flush=True)
    finally:
        if env is not None:
            env.close()
    print(json.dumps({"event": "worker_complete", "mode": args.mode, "gpu": args.gpu}), flush=True)


if __name__ == "__main__":
    main()
