"""Prepare exact mid-trajectory snapshots for the L11 token causal audit.

Cases are selected deterministically from the paired Vanilla vs Prompt-L11
results.  The branch point is the first Prompt trajectory control step whose
guided action differs from its clean-positive action.  No policy inference is
performed here: the saved Prompt actions are replayed from the canonical
snapshot and the simulator state immediately before that step is serialized.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    clone, get_image_from_maniskill2_obs_dict, restore_snapshot,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment


TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
PROMPT_SUBDIR = {
    "google_robot_open_drawer": "closed_loop",
    "google_robot_close_drawer": "closed_loop_remaining6",
    "google_robot_pick_coke_can": "closed_loop",
    "google_robot_move_near": "closed_loop",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def current_snapshot(env) -> dict:
    inner = env.unwrapped
    return {
        "sim_state": np.asarray(inner.get_state()).copy(),
        "agent_state": clone(inner.agent.get_state()),
        "rng_state": clone(inner._episode_rng.get_state()),
        "elapsed_steps": int(inner._elapsed_steps),
        "instruction": inner.get_language_instruction(),
    }


def prompt_paths(root: Path, task: str, seed: int) -> tuple[Path, Path]:
    directory = root / PROMPT_SUBDIR[task] / "episodes" / task / "prompt_single"
    stem = f"episode_{seed:03d}"
    return directory / f"{stem}_summary.json", directory / f"{stem}_arrays.npz"


def select_cases(vanilla_root: Path, prompt_root: Path) -> list[dict]:
    selected = []
    for task in TASKS:
        buckets = {"rescue": [], "harm": []}
        for seed in range(100):
            vanilla_path = vanilla_root / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json"
            prompt_path, arrays_path = prompt_paths(prompt_root, task, seed)
            if not (vanilla_path.exists() and prompt_path.exists() and arrays_path.exists()):
                continue
            vanilla = json.loads(vanilla_path.read_text())
            prompt = json.loads(prompt_path.read_text())
            if not prompt.get("technical_pass"):
                continue
            hash_keys = ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")
            if any(vanilla.get(key) != prompt.get(key) for key in hash_keys):
                raise RuntimeError(f"paired identity mismatch: {task}/{seed}")
            if not vanilla["success"] and prompt["success"]:
                category = "rescue"
            elif vanilla["success"] and not prompt["success"]:
                category = "harm"
            else:
                continue
            if len(buckets[category]) >= 10:
                continue
            trace = prompt["selector_trace"]
            step = next((i for i, row in enumerate(trace) if int(row["guided_changed_dims"]) > 0), None)
            if step is None:
                continue
            buckets[category].append({
                "task": task, "seed": seed, "category": category,
                "control_step": step,
                "vanilla_success": bool(vanilla["success"]),
                "prompt_success": bool(prompt["success"]),
                "paired_hashes": {
                    key: prompt[key] for key in hash_keys
                },
                "prompt_summary": str(prompt_path), "prompt_arrays": str(arrays_path),
            })
        selected.extend(buckets["rescue"] + buckets["harm"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--prompt-artifact", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DISPLAY", "")

    artifact = args.artifact.resolve()
    canonical = args.canonical.resolve()
    prompt_root = args.prompt_artifact.resolve()
    cases = select_cases(canonical, prompt_root)
    if args.task:
        cases = [row for row in cases if row["task"] == args.task]
    if not cases:
        raise RuntimeError("no paired Rescue/Harm cases found")

    manifest_path = artifact / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()).get("cases", []) if manifest_path.exists() else []
    completed = {(row["task"], int(row["seed"])) for row in manifest}
    for task in TASKS:
        task_cases = [row for row in cases if row["task"] == task]
        if not task_cases:
            continue
        env, _ = make_environment(task)
        try:
            for case in task_cases:
                seed, target = case["seed"], case["control_step"]
                if (task, seed) in completed:
                    continue
                snapshot_path = canonical / "snapshots" / task / f"seed_{seed:03d}.pkl"
                with snapshot_path.open("rb") as handle:
                    initial_snapshot = pickle.load(handle)
                obs, state_sha, rgb_sha = restore_snapshot(env, seed, initial_snapshot)
                expected = case["paired_hashes"]
                if (state_sha, rgb_sha) != (expected["initial_state_sha256"], expected["initial_rgb_sha256"]):
                    raise RuntimeError(f"initial restore mismatch: {task}/{seed}")
                arrays = np.load(case["prompt_arrays"], allow_pickle=False)
                actions = arrays["executed_actions"]
                if target >= len(actions):
                    raise RuntimeError(f"branch step outside action trace: {task}/{seed}/{target}")
                info = {}
                for step in range(target):
                    obs, _reward, _terminated, truncated, info = env.step(actions[step].astype(np.float64))
                    if truncated:
                        raise RuntimeError(f"trajectory truncated before branch: {task}/{seed}/{step}")
                image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
                branch = current_snapshot(env)
                destination = artifact / "snapshots" / task / f"seed_{seed:03d}_step_{target:03d}.pkl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    pickle.dump(branch, handle)
                source_mask = arrays["selected_mask"][target].astype(bool)
                positive = arrays["positive"][target]
                negative = arrays["negative"][target]
                source = json.loads(Path(case["prompt_summary"]).read_text())["selector_trace"][target]
                row = {
                    **case,
                    "snapshot_file": str(destination),
                    "branch_rgb_sha256": __import__("hashlib").sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                    "source_selected_token_ids": np.flatnonzero(source_mask).astype(int).tolist(),
                    "source_positive_token_ids": positive.argmax(axis=-1).astype(int).tolist(),
                    "source_negative_token_ids": negative.argmax(axis=-1).astype(int).tolist(),
                    "source_final_token_ids": [int(x) for x in source["final_token_ids"]],
                    "source_guided_changed_dims": int(source["guided_changed_dims"]),
                }
                sidecar = destination.with_suffix(".npz")
                np.savez_compressed(sidecar, image=image, selected_mask=source_mask.astype(np.uint8),
                                    positive=positive, negative=negative)
                row["source_arrays_file"] = str(sidecar)
                manifest.append(row)
                completed.add((task, seed))
                atomic_json(manifest_path, {
                    "protocol_id": "L11_TOKEN_GROUP_CAUSAL_AUDIT_V1",
                    "selection": "seed ascending; 10 Rescue and up to 10 available Harm per task",
                    "branch_point": "first Prompt-L11 control step with guided_changed_dims > 0",
                    "cases": sorted(manifest, key=lambda x: (x["task"], x["category"], x["seed"])),
                })
                print(json.dumps({"prepared": f"{task}/{seed}/{target}", "category": case["category"]}), flush=True)
        finally:
            env.close()
    atomic_json(manifest_path, {
        "protocol_id": "L11_TOKEN_GROUP_CAUSAL_AUDIT_V1",
        "selection": "seed ascending; 10 Rescue and up to 10 available Harm per task",
        "branch_point": "first Prompt-L11 control step with guided_changed_dims > 0",
        "cases": sorted(manifest, key=lambda x: (x["task"], x["category"], x["seed"])),
    })
    print(json.dumps({"artifact": str(artifact), "cases": len(manifest)}))


if __name__ == "__main__":
    main()
