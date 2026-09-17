"""One-state equivalence gate before fixed-lambda discovery rollout."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, get_image_from_maniskill2_obs_dict, restore_snapshot, snapshot_sha
from research.semantic_token_cd.prompt_attn_l11_count_rollout import make_environment
from research.semantic_token_cd.prompt_attn_l11_lambda_heterogeneity_rollout import (
    ARM_LAMBDAS, PROTOCOL, TASKS, audit, build_policies, ensure_config,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json, load_reference


REFERENCE_ROOTS = {
    "google_robot_open_drawer": Path("artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes"),
    "google_robot_close_drawer": Path("artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes"),
    "google_robot_pick_coke_can": Path("artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes"),
    "google_robot_move_near": Path("artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    repo = Path(__file__).resolve().parents[2]
    artifact = args.artifact.resolve(); canonical = args.canonical.resolve()
    ensure_config(artifact, canonical)
    report = {"protocol_id": PROTOCOL, "seed": 0, "tasks": {}}
    for task in TASKS:
        env, _environment_id = make_environment(task)
        checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
        config = get_policy_config("openvla", checkpoint, task, {}, False)
        arms = ("l11_positive_only", "l11_fixed_025") if task == "google_robot_open_drawer" else ("l11_positive_only",)
        policies = build_policies(OpenVLAInference(**config), task, arms)
        with (canonical / "snapshots" / task / "seed_000.pkl").open("rb") as handle: snapshot = pickle.load(handle)
        reference = load_reference(canonical, task, 0)
        fixed_dir = repo / REFERENCE_ROOTS[task] / task / "prompt_single"
        fixed_summary = json.loads((fixed_dir / "episode_000_summary.json").read_text())
        with np.load(fixed_dir / "episode_000_arrays.npz") as fixed_arrays:
            fixed_first = {key: np.asarray(fixed_arrays[key][0]) for key in ("positive", "negative", "selected_mask")}
        task_report = {}
        for arm, policy in policies.items():
            obs, state_sha, rgb_sha = restore_snapshot(env, 0, snapshot)
            instruction = env.unwrapped.get_language_instruction()
            policy.reset(instruction, seed=0); policy._episode_trace = []; policy._episode_logits = []
            image = get_image_from_maniskill2_obs_dict(env, obs)
            _raw, _actions, trace = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
            checks = audit([trace], arm)
            record = policy._episode_logits[0]
            equivalence = {
                "snapshot_hash_equal": snapshot_sha(snapshot) == reference["canonical_snapshot_sha256"] == fixed_summary["canonical_snapshot_sha256"],
                "state_hash_equal": state_sha == reference["initial_state_sha256"] == fixed_summary["initial_state_sha256"],
                "rgb_hash_equal": rgb_sha == reference["initial_rgb_sha256"] == fixed_summary["initial_rgb_sha256"],
                "positive_logits_bit_equal_l11_050": np.array_equal(record["positive"], fixed_first["positive"]),
                "negative_logits_bit_equal_l11_050": np.array_equal(record["negative"], fixed_first["negative"]),
                "selected_mask_bit_equal_l11_050": np.array_equal(record["selected_mask"], fixed_first["selected_mask"]),
                "technical_audit_pass": checks["technical_pass"],
            }
            equivalence["passed"] = all(equivalence.values())
            if not equivalence["passed"]:
                raise RuntimeError({"task": task, "arm": arm, **equivalence})
            task_report[arm] = equivalence
        report["tasks"][task] = task_report
        del policies
        env.close()
    report["passed"] = all(
        arm["passed"] for task in report["tasks"].values() for arm in task.values()
    )
    atomic_json(artifact / "preflight/PREFLIGHT_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
