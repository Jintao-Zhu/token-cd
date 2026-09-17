"""Prepare fixed Prompt-L11 drawer snapshots for phase/strength/token causality."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import clone, get_image_from_maniskill2_obs_dict, restore_snapshot
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.xdrawer_protocol import drawer_geometry

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer")
PROMPT_SUBDIR = {"google_robot_open_drawer": "closed_loop", "google_robot_close_drawer": "closed_loop_remaining6"}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); temporary.replace(path)


def snapshot(env) -> dict:
    inner = env.unwrapped
    return {"sim_state": np.asarray(inner.get_state()).copy(), "agent_state": clone(inner.agent.get_state()),
            "rng_state": clone(inner._episode_rng.get_state()), "elapsed_steps": int(inner._elapsed_steps),
            "instruction": inner.get_language_instruction()}


def prompt_files(prompt_root: Path, task: str,diensten: int):
    directory = prompt_root / PROMPT_SUBDIR[task] / "episodes" / task / "prompt_single"
    stem = f"episode_{diensten:03d}"
    return directory / f"{stem}_summary.json", directory / f"{stem}_arrays.npz"


def fixed_cases(canonical: Path, prompt_root: Path) -> list[dict]:
    cases = []
    for task in TASKS:
        buckets = {"harm": [], "rescue": []}
        for seed in range(100):
            vanilla_path = canonical / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json"
            prompt_path, arrays_path = prompt_files(prompt_root, task, seed)
            if not all(path.exists() for path in (vanilla_path, prompt_path, arrays_path)): continue
            vanilla, prompt = json.loads(vanilla_path.read_text()), json.loads(prompt_path.read_text())
            if any(vanilla.get(key) != prompt.get(key) for key in
                   ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")):
                raise RuntimeError(f"identity mismatch {task}/{seed}")
            if vanilla["success"] and not prompt["success"]: category = "harm"
            elif not vanilla["success"] and prompt["success"]: category = "rescue"
            else: continue
            buckets[category].append({"task": task, "seed": seed, "category": category,
                                      "prompt_summary": str(prompt_path), "prompt_arrays": str(arrays_path)})
        harm = buckets["harm"]
        rescue = buckets["rescue"][:len(harm)]
        cases.extend(harm + rescue)
    return cases


def status(inner, task: str) -> dict:
    geometry = drawer_geometry(inner, "top")
    tcp = np.asarray(inner.tcp.pose.p, dtype=np.float64)
    points = geometry.get("handle_points") or [geometry["handle_p"]]
    return {"qpos": float(np.asarray(inner.art_obj.get_qpos())[inner.joint_idx]),
            "tcp_handle_distance": min(float(np.linalg.norm(tcp - np.asarray(p))) for p in points)}


def choose_steps(trace: list[dict], physical: list[dict], task: str) -> dict[str, dict]:
    n = len(trace); sign = 1.0 if "open" in task else -1.0; q0 = physical[0]["qpos"]
    first_change = next((i for i, row in enumerate(trace) if int(row["guided_changed_dims"]) > 0), 0)
    contact = next((i for i, row in enumerate(physical) if row["tcp_handle_distance"] <= .06), None)
    motion = next((i for i, row in enumerate(physical) if sign * (row["qpos"] - q0) >= .005), None)
    middle_candidates = [value for value in (contact, motion) if value is not None]
    middle = min(middle_candidates) if middle_candidates else int(round(.5 * (n - 1)))
    late = int(round(.8 * (n - 1)))
    steps = [min(first_change, n - 1), min(middle, n - 1), min(late, n - 1)]
    fallback = [int(round(f * (n - 1))) for f in (.1, .5, .9)]
    for i in range(3):
        if steps[i] in steps[:i]: steps[i] = fallback[i]
    if len(set(steps)) < 3:
        steps = sorted(set(steps + fallback))[:3]
    return {
        "approach": {"step": steps[0], "rule": "first guided action change"},
        "contact_or_motion": {"step": steps[1], "rule": "first <=6cm handle or >=5mm drawer motion; midpoint fallback"},
        "late": {"step": steps[2], "rule": "80% of Prompt trajectory"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True); parser.add_argument("--prompt-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True); parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ.setdefault("DISPLAY", "")
    root, canonical, prompt_root = args.artifact.resolve(), args.canonical.resolve(), args.prompt_artifact.resolve()
    cases = [x for x in fixed_cases(canonical, prompt_root) if x["task"] == args.task]
    env, _ = make_environment(args.task); records = []
    try:
        for case in cases:
            seed = case["seed"]; summary = json.loads(Path(case["prompt_summary"]).read_text())
            arrays = np.load(case["prompt_arrays"], allow_pickle=False); actions = arrays["executed_actions"]
            with (canonical / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle: initial = pickle.load(handle)
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, initial)
            if (state_sha, rgb_sha) != (summary["initial_state_sha256"], summary["initial_rgb_sha256"]):
                raise RuntimeError(f"initial restore mismatch {args.task}/{seed}")
            states, physical = [], []
            for step in range(len(actions)):
                states.append((snapshot(env), np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)))
                physical.append(status(env.unwrapped, args.task))
                obs, _reward, _terminated, truncated, _info = env.step(actions[step].astype(np.float64))
                if truncated and step + 1 < len(actions): raise RuntimeError(f"early replay truncation {args.task}/{seed}")
            phase_steps = choose_steps(summary["selector_trace"], physical, args.task)
            for phase, spec in phase_steps.items():
                step = int(spec["step"]); snap, image = states[step]
                directory = root / "snapshots" / args.task / f"seed_{seed:03d}"; directory.mkdir(parents=True, exist_ok=True)
                pkl = directory / f"{phase}_step_{step:03d}.pkl"; npz = pkl.with_suffix(".npz")
                with pkl.open("wb") as handle: pickle.dump(snap, handle)
                np.savez_compressed(npz, image=image, selected_mask=arrays["selected_mask"][step],
                                    prompt_attention=arrays["prompt_attention"][step])
                records.append({**case, "phase": phase, "phase_rule": spec["rule"], "control_step": step,
                                "snapshot_file": str(pkl), "arrays_file": str(npz),
                                "rgb_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                                "physical_before": physical[step],
                                "selected_token_ids": np.flatnonzero(arrays["selected_mask"][step]).astype(int).tolist()})
                print(json.dumps({"prepared": f"{args.task}/{seed}/{phase}/{step}", "category": case["category"]}), flush=True)
    finally: env.close()
    manifest_path = root / "MANIFEST.json"; old = json.loads(manifest_path.read_text())["states"] if manifest_path.exists() else []
    kept = [x for x in old if x["task"] != args.task] + records
    atomic_json(manifest_path, {"protocol_id": "DRAWER_PHASE_STRENGTH_TOKEN_CAUSAL_V1",
        "case_rule": "all historical Prompt-L11 Harm; equal number earliest-seed Rescue within each task",
        "tasks": list(TASKS), "treatment_horizon": 10,
        "strength_arms": ["dynamic_l11_lambda_0p5", "dynamic_l11_lambda_0p25", "clean"],
        "token_arms": "remove each nonempty fixed 3x2 sector from the locked branch-point L11 mask for first action, then L11 lambda 0.5",
        "states": sorted(kept, key=lambda x: (x["task"], x["category"], x["seed"], x["phase"]))})
    print(json.dumps({"task": args.task, "states": len(records), "total_manifest": len(kept)}))


if __name__ == "__main__": main()
