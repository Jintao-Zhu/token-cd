#!/usr/bin/env python3
"""Initial-state invariance preflight for the OpenVLA global lambda grid."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=5)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from properties import get_policy_config
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, restore_snapshot, snapshot_sha
from research.semantic_token_cd import openvla_global_lambda_grid_rollout as runner
from research.semantic_token_cd import prompt_attn_l11_budget_causal_rollout as shared
from research.semantic_token_cd.spatial_grid_rollout import get_image_from_maniskill2_obs_dict


ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
SNAPSHOT_ROOT = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
ARTIFACT = ROOT / "artifacts/openvla_global_lambda_grid_0_199_v1"
ARMS = tuple(runner.CARD_ARMS["6"])
SEEDS = range(5)


def main() -> None:
    import gymnasium as gym  # noqa: F401

    ARTIFACT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in runner.TASKS:
        print(f"[preflight] task={task}", flush=True)
        env, environment_id = shared.make_environment(task)
        cfg = get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), task, {}, False)
        base = OpenVLAInference(**cfg)
        policies = runner.build_policies(base, task, "6", ARMS)
        for seed in SEEDS:
            snapshot = pickle.loads((SNAPSHOT_ROOT / "snapshots" / task / f"seed_{seed:03d}.pkl").read_bytes())
            reference = runner.load_scan_reference(SNAPSHOT_ROOT, task, seed)
            obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
            instruction = str(env.unwrapped.get_language_instruction())
            expected = (
                str(reference["canonical_snapshot_sha256"]),
                str(reference["initial_state_sha256"]),
                str(reference["initial_rgb_sha256"]),
            )
            actual = (snapshot_sha(snapshot), state_hash, rgb_hash)
            if actual != expected:
                raise RuntimeError(f"initial hash mismatch {task}:{seed}: expected={expected}, actual={actual}")
            if instruction != str(reference["instruction"]):
                raise RuntimeError(f"instruction mismatch {task}:{seed}")
            image = get_image_from_maniskill2_obs_dict(env, obs)
            arm_rows = {}
            for arm, policy in policies.items():
                policy.reset(instruction, seed=seed)
                policy._episode_trace = []
                policy._episode_logits = []
                policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
                trace = policy._episode_trace[-1]
                logits = policy._episode_logits[-1]
                arm_rows[arm] = {
                    "m_t": int(trace["m_t"]),
                    "selected_token_ids": list(trace["selected_token_ids"]),
                    "positive_token_ids": list(trace["positive_token_ids"]),
                    "negative_token_ids": list(trace["negative_token_ids"]),
                    "final_token_ids": list(trace["final_token_ids"]),
                    "l11_score_sha256": trace.get("l11_score_sha256"),
                    "D_feat": float(trace["D_feat"]),
                    "positive_logits": np.asarray(logits["positive"]).tolist(),
                    "negative_logits": np.asarray(logits["negative"]).tolist(),
                    "clean_action": list(trace["clean_action"]),
                    "guided_action": list(trace["guided_action"]),
                    "lambda": float(trace["lambda"]),
                }
            ref = arm_rows["lambda_000"]
            for arm in ARMS:
                row = arm_rows[arm]
                for key in ("m_t", "selected_token_ids", "positive_token_ids", "negative_token_ids", "l11_score_sha256"):
                    if row[key] != ref[key]:
                        raise RuntimeError(f"lambda changed pre-guidance field {key}: {task}:{seed}:{arm}")
                if not np.array_equal(row["positive_logits"], ref["positive_logits"]):
                    raise RuntimeError(f"lambda changed clean logits: {task}:{seed}:{arm}")
                if not np.array_equal(row["negative_logits"], ref["negative_logits"]):
                    raise RuntimeError(f"lambda changed negative logits: {task}:{seed}:{arm}")
                if not np.isclose(row["D_feat"], ref["D_feat"], rtol=0, atol=0):
                    raise RuntimeError(f"lambda changed harmonic reconstruction dose: {task}:{seed}:{arm}")
            zero = arm_rows["lambda_000"]
            if zero["final_token_ids"] != zero["positive_token_ids"]:
                raise RuntimeError(f"lambda=0 token output differs from clean: {task}:{seed}")
            if not np.array_equal(np.asarray(zero["guided_action"]), np.asarray(zero["clean_action"])):
                raise RuntimeError(f"lambda=0 decoded action differs from clean: {task}:{seed}")
            rows.append({
                "task": task, "seed": seed, "environment_id": environment_id,
                "snapshot_sha256": actual[0], "initial_state_sha256": actual[1],
                "initial_rgb_sha256": actual[2], "instruction": instruction,
                "arms": {
                    arm: {k: v for k, v in result.items() if k not in {"positive_logits", "negative_logits"}}
                    for arm, result in arm_rows.items()
                },
            })
            print(f"[preflight] PASS {task}:{seed:03d}", flush=True)
        env.close()

    report = {
        "protocol_id": "OPENVLA_GLOBAL_LAMBDA_GRID_0_199_V1_PREFLIGHT",
        "tasks": list(runner.TASKS), "seeds": list(SEEDS), "arms": list(ARMS),
        "n_scenes": len(rows), "n_scene_arm_checks": len(rows) * len(ARMS),
        "all_checks_passed": len(rows) == 45,
        "checks": [
            "same canonical snapshot/state/RGB/instruction across lambda arms",
            "same m_t, selected token IDs, L11 attention hash, and harmonic feature dose across lambda arms",
            "positive and negative logits exactly equal across lambda arms",
            "lambda=0 final token IDs and decoded action equal clean output",
        ],
        "rows": rows,
    }
    path = ARTIFACT / "PREFLIGHT_45_SCENES.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md = [
        "# OpenVLA global lambda-grid preflight",
        "",
        f"STATUS: {'PASS' if report['all_checks_passed'] else 'FAIL'}",
        f"Scenes: {report['n_scenes']}; scene-arm checks: {report['n_scene_arm_checks']}",
        "",
        "The only changing field across arms is lambda. All renderer, snapshot, clean/negative branch, selector, budget, and harmonic checks passed.",
    ]
    (ARTIFACT / "PREFLIGHT_45_SCENES.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
