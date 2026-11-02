"""Paired RT-full SIMPLER rollouts for pi0-FAST L6-Matched at lambda 1.25/1.50."""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from research.semantic_token_cd import pi0_fast_simpler_l6_matched_rollout as base

TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
)
ARMS = ("l6_matched_125", "l6_matched_150")
LAMBDA_BY_ARM = {"l6_matched_125": 1.25, "l6_matched_150": 1.50}
RT_DEPTH = 8
RGB_MAE_MAX = 0.05
RGB_MAX_ABS_MAX = 8
STAGED_PROMPT_TASKS = {"google_robot_place_apple_in_closed_top_drawer"}
VANILLA_RUN = Path("/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_9task_vanilla_vs_l6jit_lambda050_rtfull_100seed_v1")
LAMBDA100_RUN = Path("/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_9task_l6jit_lambda100_rtfull_seed0_99_v1")


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


def expected_renderer() -> dict[str, Any]:
    return {"shader_dir": "rt", "rt_max_path_depth": RT_DEPTH,
            "rt_samples_per_pixel": 32, "rt_use_denoiser": True}


def checkpoint_for(family: str) -> Path:
    name = "fractal" if family == "google" else "bridge"
    return (base.ROOT / "checkpoints/openpi_fast_finetuned" / name).resolve()


def read_episode(root: Path, task: str, seed: int, arm: str) -> dict[str, Any]:
    path = root / "episodes" / task / f"{arm}_seed{seed:03d}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def lowres_match(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, float, int]:
    x = np.frombuffer(base64.b64decode(a["initial_rgb_lowres_b64"]), dtype=np.uint8)
    y = np.frombuffer(base64.b64decode(b["initial_rgb_lowres_b64"]), dtype=np.uint8)
    if x.shape != (56 * 56 * 3,) or y.shape != x.shape:
        raise RuntimeError("unexpected stored 56x56 RGB payload shape")
    diff = np.abs(x.astype(np.int16) - y.astype(np.int16))
    mae, max_abs = float(diff.mean()), int(diff.max())
    return mae <= RGB_MAE_MAX and max_abs <= RGB_MAX_ABS_MAX, mae, max_abs


def episode_matches(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    state_equal = a.get("initial_agent_state_sha256") == b.get("initial_agent_state_sha256")
    rgb_ok, mae, max_abs = lowres_match(a, b)
    return state_equal and rgb_ok, {
        "initial_state_equal": state_equal,
        "initial_rgb_equal": a.get("initial_rgb_sha256") == b.get("initial_rgb_sha256"),
        "initial_rgb_lowres_mae": mae,
        "initial_rgb_lowres_max_abs_diff": max_abs,
        "matched": state_equal and rgb_ok,
    }


def check_reference(task: str, seed: int, family: str) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any] | None, bool, dict[str, Any]
]:
    vanilla = read_episode(VANILLA_RUN, task, seed, "vanilla")
    l100 = read_episode(LAMBDA100_RUN, task, seed, "l6_matched_100")
    for label, episode in (("Vanilla", vanilla), ("lambda=1.0", l100)):
        if episode.get("status") != "complete":
            raise RuntimeError(f"incomplete {label} reference: {task}/{seed}")
        if str(Path(episode.get("checkpoint", "")).resolve()) != str(checkpoint_for(family)):
            raise RuntimeError(f"{label} checkpoint mismatch: {task}/{seed}")
        if episode.get("renderer") != expected_renderer():
            raise RuntimeError(f"{label} renderer mismatch: {task}/{seed}")
    same, diag = episode_matches(vanilla, l100)
    old_pair_path = VANILLA_RUN / "pairs" / task / f"seed{seed:03d}.json"
    old_pair = json.loads(old_pair_path.read_text()) if old_pair_path.is_file() else None
    if old_pair and (old_pair.get("guidance_lambda") != 0.5 or not old_pair.get("initial_state_equal")
                     or not old_pair.get("initial_rgb_coarse_equal") or old_pair.get("pair_excluded", False)):
        old_pair = None
    return vanilla, l100, old_pair, same, diag


class L6JITPolicy(base.RolloutPolicy):
    """Keep the previous verified L6/Matched/JIT decode path; vary lambda only."""

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.guided_fallbacks = 0

    def chunk(self, arm: str, raw: dict, seed: int):
        if arm not in LAMBDA_BY_ARM:
            raise ValueError(f"unsupported arm: {arm}")
        lam = LAMBDA_BY_ARM[arm]
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
            fb_actions = np.asarray(fallback["actions"], dtype=np.float32)
            if fb_actions.shape != (10, 7) or not np.isfinite(fb_actions).all():
                raise RuntimeError(f"invalid Vanilla fallback chunk: {fb_actions.shape}")
            self.guided_fallbacks += 1
            trace.update({"guided_fallback": True,
                          "fallback_reason": "official_decoder_disagreement",
                          "fallback_server_meta": fallback.get("pi0_fast_server_meta", {})})
            return fb_actions, trace
        trace["guided_fallback"] = False
        return actions, trace


def run_preflight(policy: L6JITPolicy, env, out: Path, task: str, seed: int, family: str) -> None:
    vanilla, l100, _, same_l1, l1_diag = check_reference(task, seed, family)
    if not same_l1:
        raise RuntimeError(f"preflight Vanilla/lambda=1.0 pairing failed: {task}/{seed}: {l1_diag}")
    obs, reset_info = env.reset(seed=seed)
    instruction = str(env.unwrapped.get_language_instruction())
    camera = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"
    image = np.asarray(obs["image"][camera]["rgb"])
    state_sha = base._digest(base._eef_pos(obs["agent"], env))
    reset_equal = json.dumps(reset_info, sort_keys=True, default=str) == json.dumps(
        vanilla.get("reset_info"), sort_keys=True, default=str)
    if state_sha != vanilla.get("initial_agent_state_sha256") or not reset_equal:
        raise RuntimeError(f"reset state/scene metadata mismatch: {task}/{seed}")
    if task not in STAGED_PROMPT_TASKS and instruction != vanilla.get("instruction"):
        raise RuntimeError(f"initial prompt mismatch: {task}/{seed}")
    env_id = getattr(getattr(env, "spec", None), "id", None)
    if env_id != vanilla.get("environment_id"):
        raise RuntimeError(f"environment id mismatch: {task}/{seed}: {env_id!r}")
    raw = base._raw_policy_input(image, instruction, obs["agent"], env)
    lowres = base._lowres_rgb(image).reshape(-1).astype(np.int16)
    ref_lowres = np.frombuffer(base64.b64decode(vanilla["initial_rgb_lowres_b64"]), dtype=np.uint8).astype(np.int16)
    d = np.abs(lowres - ref_lowres)
    if float(d.mean()) > RGB_MAE_MAX or int(d.max()) > RGB_MAX_ABS_MAX:
        raise RuntimeError(f"reset RGB mismatch: {task}/{seed}: mae={d.mean()}, max={d.max()}")
    arm_reports = {}
    for arm in ARMS:
        _, trace = policy.chunk(arm, raw, seed * 1009)
        arm_reports[arm] = {
            "lambda": LAMBDA_BY_ARM[arm], "selector_layer": trace.get("selector_layer_label"),
            "matched_budget_ok": trace.get("selected_count_equals_matched"),
            "jit_header": trace.get("jit_header"),
            "fast_action_stream_valid": trace.get("fast_action_stream_valid"),
            "decoder_agrees": trace.get("openpi_official_decoder_agrees"),
            "guided_fallback": trace.get("guided_fallback", False),
        }
    report = {
        "status": "passed", "task": task, "seed": seed, "family": family,
        "checkpoint": str(checkpoint_for(family)), "renderer": expected_renderer(),
        "initial_agent_state_equal_to_vanilla": True, "reset_scene_metadata_equal": True,
        "initial_rgb_lowres_mae": float(d.mean()), "initial_rgb_lowres_max_abs_diff": int(d.max()),
        "lambda100_reference_matches_vanilla": same_l1,
        "arms": arm_reports,
    }
    atomic_json(out / "preflight" / task / f"seed{seed:03d}.json", report)
    print("PREFLIGHT " + json.dumps(report), flush=True)


def pair_episodes(out: Path, task: str, seed: int, episodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    family = "google" if task.startswith("google_robot_") else "widowx"
    vanilla, l100, old_pair, ref_l1_matches, ref_l1_diag = check_reference(task, seed, family)
    per_arm = {}
    for arm in ARMS:
        ep = episodes[arm]
        if ep.get("status") != "complete":
            continue
        match_v, diag_v = episode_matches(vanilla, ep)
        match_l1, diag_l1 = episode_matches(l100, ep)
        match_l1 = match_l1 and ref_l1_matches and match_v
        per_arm[arm] = {
            "success": bool(ep["success"]), "valid_vs_vanilla": match_v,
            "valid_vs_lambda100": match_l1, "vanilla_match": diag_v,
            "lambda100_match": diag_l1,
        }
    pair = {
        "status": "paired" if per_arm else "incomplete", "task": task, "seed": seed,
        "checkpoint": str(checkpoint_for(family)), "renderer": expected_renderer(),
        "vanilla_success": bool(vanilla["success"]),
        "lambda050_success": old_pair.get("guided_success") if old_pair else None,
        "lambda050_pair_available": bool(old_pair),
        "lambda100_success": bool(l100["success"]),
        "lambda100_pair_available": bool(ref_l1_matches),
        "vanilla_lambda100_reference_match": ref_l1_matches,
        "vanilla_lambda100_reference_diag": ref_l1_diag,
        "arms": per_arm,
        "valid_all_lambdas_125_150_vs_vanilla": all(
            per_arm.get(a, {}).get("valid_vs_vanilla", False) for a in ARMS),
        "valid_all_lambdas_125_150_vs_lambda100": all(
            per_arm.get(a, {}).get("valid_vs_lambda100", False) for a in ARMS),
    }
    for arm in ARMS:
        suffix = arm.rsplit("_", 1)[1]
        pair[f"lambda{suffix}_success"] = per_arm.get(arm, {}).get("success")
        pair[f"lambda{suffix}_pair_available"] = bool(
            per_arm.get(arm, {}).get("valid_vs_vanilla", False))
    atomic_json(out / "pairs" / task / f"seed{seed:03d}.json", pair)
    return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--family", choices=("google", "widowx"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--job", nargs="+", required=True)
    args = parser.parse_args()
    jobs = [parse_job(s) for s in args.job]
    if args.gpu not in (1, 2, 3):
        raise SystemExit(f"GPU {args.gpu} is not authorized")
    if not (args.out / "protocol.json").is_file():
        raise SystemExit(f"missing protocol lock: {args.out / 'protocol.json'}")
    prefix = "google_robot_" if args.family == "google" else "widowx_"
    if any(not t.startswith(prefix) for t, _, _ in jobs):
        raise SystemExit(f"job does not match family {args.family}")
    if args.mode == "preflight" and any(a != 0 or b != 0 for _, a, b in jobs):
        raise SystemExit("preflight jobs must use seed 0")
    policy = L6JITPolicy("127.0.0.1", args.port)
    env = None
    current_task = None
    try:
        for task, start, end in jobs:
            family = "google" if task.startswith("google_robot_") else "widowx"
            base._CHECKPOINT_FOR_LOG = str(checkpoint_for(family))
            if task != current_task:
                if env is not None:
                    env.close()
                env = base._environment(task, RT_DEPTH)
                current_task = task
            if args.mode == "preflight":
                run_preflight(policy, env, args.out, task, start, family)
                continue
            for seed in range(start, end + 1):
                completed = {}
                for arm in ARMS:
                    ep_path = args.out / "episodes" / task / f"{arm}_seed{seed:03d}.json"
                    try:
                        if ep_path.is_file():
                            candidate = json.loads(ep_path.read_text())
                            if candidate.get("status") == "complete" and candidate.get("arm") == arm:
                                completed[arm] = candidate
                                continue
                        completed[arm] = base._episode(policy, env, task, seed, arm, args.out, RT_DEPTH)
                    except Exception as exc:  # noqa: BLE001
                        error = {
                            "status": "error", "task": task, "seed": seed, "arm": arm,
                            "gpu": args.gpu, "error_type": type(exc).__name__, "error": repr(exc),
                            "traceback": traceback.format_exc(),
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        atomic_json(args.out / "errors" / task / f"{arm}_seed{seed:03d}.json", error)
                        print("EPISODE_ERROR " + json.dumps({k: error[k] for k in ("task", "seed", "arm", "error_type", "error")}), flush=True)
                if len(completed) == len(ARMS):
                    pair = pair_episodes(args.out, task, seed, completed)
                    print("PAIR " + json.dumps(pair), flush=True)
    finally:
        if env is not None:
            env.close()
    print(json.dumps({"event": "worker_complete", "mode": args.mode, "gpu": args.gpu}), flush=True)


if __name__ == "__main__":
    main()
