"""SEMANTIC_ENTITY_CD_ROLLOUT_PILOT_V1 — closed-loop rollout runner (one task).

Runs three arms — vanilla, entity_cd_λ0.25, entity_cd_λ0.5 — on a single task with
exact snapshot pairing (all arms start from the identical initial state/rng per seed)
so that Rescue/Harm can be computed per-pair. Shares one frozen OpenVLA-7B model
across arms via copy.copy + __class__ reassignment.

This is a PILOT (not a formal confirmation): it answers whether the Phase-2B
language-selected semantic entity negative branch (PCD alignment 0.586) is already
above the minimum closed-loop effectiveness threshold.

Run (per task):
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3 \
  PYTHONPATH="task1/shim_site:$PWD:official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD" \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/rollout_pilot.py \
      --artifact artifacts/semantic_entity_cd_rollout_pilot_v1 \
      --task google_robot_close_drawer --seeds 0,1,2,3,4,5,6,7,8,9
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
import torch

# The SIMPLER rollout runs several processes in parallel on a shared 32-core box.
# PyTorch's default nproc-sized thread pools spin and thrash (130 threads/process,
# load avg 106, ~5x per-episode slowdown). Pin every CPU-side pool to 1 thread;
# GPU compute is unaffected.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

REPO_ROOT = Path("/data/docker/dev_zjt/data/code")
PCD_ROOT = REPO_ROOT / "official-reproductions/pcd_openvla_simpler_box_31b027e"
PCD_SOURCE = PCD_ROOT / "source/PCD"
MEAN_PATH = (REPO_ROOT / "artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/"
             "position_conditioned_visual_mean.pt")

for _p in (str(REPO_ROOT / "task1/shim_site"), str(REPO_ROOT), str(PCD_SOURCE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parallel_inference import get_image_from_maniskill2_obs_dict  # noqa: E402
from properties import get_policy_config  # noqa: E402
from simpler_env.policies.openvla.openvla_model import OpenVLAInference  # noqa: E402
from utils import (  # noqa: E402
    convert_numpy_or_torch_to_python, stat_final, stat_first, summarize, write_video,
)

from research.token_pcd_stage_a.core import array_sha256  # noqa: E402
from research.semantic_token_cd.rollout_policy import SemanticEntityCDInference  # noqa: E402

ARMS = ("vanilla", "entity_cd_025", "entity_cd_050")


def jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def flatten_action(action):
    return np.concatenate([
        np.asarray(action["world_vector"], dtype=np.float64),
        np.asarray(action["rot_axangle"], dtype=np.float64),
        np.asarray(action["gripper"], dtype=np.float64),
    ])


def clone(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return type(value)((key, clone(item)) for key, item in value.items())
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if hasattr(value, "p") and hasattr(value, "q"):  # RandomState
        return type(value)(np.asarray(value.p).copy(), np.asarray(value.q).copy())
    return copy.deepcopy(value)


def wrapped_observation(env):
    obs = env.unwrapped.get_obs()
    wrappers, current = [], env
    while hasattr(current, "env"):
        if hasattr(current, "observation"):
            wrappers.append(current)
        current = current.env
    for wrapper in reversed(wrappers):
        obs = wrapper.observation(obs)
    return obs


def capture_snapshot(env, seed):
    env.reset(seed=seed)
    inner = env.unwrapped
    return {"sim_state": np.asarray(inner.get_state()).copy(),
            "agent_state": clone(inner.agent.get_state()),
            "rng_state": clone(inner._episode_rng.get_state()),
            "instruction": inner.get_language_instruction()}


def restore_snapshot(env, seed, snapshot):
    env.reset(seed=seed)
    inner = env.unwrapped
    inner.set_state(snapshot["sim_state"].copy())
    inner.agent.set_state(clone(snapshot["agent_state"]))
    inner._episode_rng.set_state(clone(snapshot["rng_state"]))
    inner._elapsed_steps = 0
    obs = wrapped_observation(env)
    return obs, array_sha256(np.asarray(inner.get_state())), array_sha256(get_image_from_maniskill2_obs_dict(env, obs))


def snapshot_sha(snapshot):
    digest = hashlib.sha256()
    digest.update(array_sha256(snapshot["sim_state"]).encode())
    digest.update(repr(snapshot["agent_state"]).encode())
    digest.update(repr(snapshot["rng_state"]).encode())
    digest.update(snapshot["instruction"].encode())
    return digest.hexdigest()


def build_entity_policy(vanilla, mean, lambd, kmeans_K=8, kmeans_seed=0):
    p = copy.copy(vanilla)
    p.__class__ = SemanticEntityCDInference
    p.alpha = lambd
    p.lambd = lambd
    p.replacement_mean = mean
    p.kmeans_K = kmeans_K
    p.kmeans_seed = kmeans_seed
    p._selector_instr = None
    p._entities = []
    p._entity_emb = []
    p._emb_cache = {}
    p._episode_trace = []
    return p


def run_episode(env, policy, arm, instruction, obs):
    predicted_terminated, truncated, timestep = False, False, 0
    step_infos = []
    image = get_image_from_maniskill2_obs_dict(env, obs)
    while not (predicted_terminated or truncated):
        if arm == "vanilla":
            raw_action, actions = policy.step(image, instruction, proprio=obs["agent"]["eef_pos"])
        else:
            raw_action, actions, aux = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            executed = flatten_action(action)
            if not np.isfinite(executed).all():
                raise FloatingPointError("Non-finite action")
            obs, _reward, _success, truncated, info = env.step(executed)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            timestep += 1
            step_infos.append(convert_numpy_or_torch_to_python(info))
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not env.unwrapped.is_final_subtask():
                predicted_terminated = False
                env.advance_to_next_subtask()
            new_instruction = env.unwrapped.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction
    result = summarize(step_infos)
    result.update(stat_first(step_infos))
    result.update(stat_final(step_infos))
    return result, timestep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pcd-root", type=Path, default=PCD_ROOT)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seeds", required=True, help="comma-separated ints")
    p.add_argument("--lambdas", default="0.25,0.50", help="comma-separated floats")
    p.add_argument("--mean-path", type=Path, default=MEAN_PATH)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seeds = [int(x) for x in a.seeds.split(",") if x]
    lambdas = [float(x) for x in a.lambdas.split(",") if x]
    art = a.artifact.resolve()
    pcd_source = (a.pcd_root / "source/PCD").resolve()
    for _p in (str(REPO_ROOT / "task1/shim_site"), str(REPO_ROOT), str(pcd_source)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    task_root = art / "episodes" / a.task
    task_root.mkdir(parents=True, exist_ok=True)

    import simpler_env
    env = simpler_env.make(a.task)

    checkpoint = str(pcd_source / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, a.task, {}, False)
    vanilla = OpenVLAInference(**config)
    mean = torch.load(a.mean_path.resolve(), map_location="cpu", weights_only=True)["mean"]

    policies = {"vanilla": vanilla}
    for lambd in lambdas:
        key = f"entity_cd_{int(round(lambd * 100)):03d}"
        policies[key] = build_entity_policy(vanilla, mean, lambd)

    arms = ["vanilla"] + [f"entity_cd_{int(round(l * 100)):03d}" for l in lambdas]

    manifests = []
    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        initial = {}
        pair_summary = {}
        for arm in arms:
            arm_dir = task_root / arm
            arm_dir.mkdir(exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            if summary_path.exists():
                print(json.dumps({"skip": a.task, "seed": seed, "arm": arm}), flush=True)
                pair_summary[arm] = json.loads(summary_path.read_text())
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            result, timestep = run_episode(env, policy, arm, instruction, obs)
            summary = {
                "protocol_id": "SEMANTIC_ENTITY_CD_ROLLOUT_PILOT_V1",
                "task": a.task, "seed": seed, "arm": arm,
                "success": bool(result["success"]),
                "control_steps": timestep,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                "result": jsonable(result),
            }
            if arm != "vanilla":
                summary["selector_trace"] = jsonable(policy._episode_trace)
                summary["mean_language_score"] = (
                    float(np.mean([t["language_score"] for t in policy._episode_trace]))
                    if policy._episode_trace else None)
                summary["n_degenerate_steps"] = sum(1 for t in policy._episode_trace if t.get("degenerate"))
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            pair_summary[arm] = summary
            print(json.dumps({"task": a.task, "seed": seed, "arm": arm,
                              "success": summary["success"], "steps": timestep}), flush=True)

        # Pairing verification — must hold across ALL arms, including ones already
        # run in a prior session (resume). A seed with no fresh arms (initial == {})
        # is validated from the stored hashes instead of raising.
        state_hashes, rgb_hashes, canonical_hashes = {}, {}, {}
        for arm in arms:
            ps = pair_summary[arm]
            state_hashes[arm] = initial[arm][0] if arm in initial else ps["initial_state_sha256"]
            rgb_hashes[arm] = initial[arm][1] if arm in initial else ps["initial_rgb_sha256"]
            canonical_hashes[arm] = canonical if arm in initial else ps.get("canonical_snapshot_sha256")
        if len(set(state_hashes.values())) != 1 or len(set(rgb_hashes.values())) != 1:
            raise RuntimeError(f"Cross-arm pairing mismatch seed {seed}: state={state_hashes}")
        if len(set(canonical_hashes.values())) != 1:
            raise RuntimeError(f"Snapshot non-determinism across sessions seed {seed}: {canonical_hashes}")
        manifests.append({"seed": seed, "canonical_snapshot_sha256": canonical,
                          "initial_state_sha256": state_hashes["vanilla"],
                          "initial_rgb_sha256": rgb_hashes["vanilla"],
                          "exact_pairing": True})

    (task_root / "pairing_manifest.json").write_text(json.dumps(
        {"task": a.task, "seeds": seeds, "arms": arms,
         "pairs": manifests, "all_arm_exact_pairing": True}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": a.task, "DONE": True, "n_seeds": len(seeds), "arms": arms}), flush=True)


if __name__ == "__main__":
    main()
