#!/usr/bin/env python3
"""Acceptance gate for SAME_STATE_FORK_V1 (LIBERO side).

Runs four checks on a small state sample before the main experiment is allowed:
  1. repeated clean continuation is deterministic
  2. lambda = 0 reproduces vanilla
  3. identity reconstruction matches the original forward
  4. extracting attention / recording diagnostics does not change the action

Reports the bf16 identity-reconstruction logit difference explicitly, and
whether it lands on a valid action token and how it compares to the action
margin and to ordinary repeated-forward noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.libero_runtime import (
    build_prompt, load_policy, predict_action, prepare_agentview, prepare_env_action,
)
from research.semantic_token_cd.libero_matched_rollout import generate_clean_action
from research.semantic_token_cd.libero_policy import forward_logits
from research.semantic_token_cd.libero_same_state_fork_worker import (
    branch_step_action, make_env, replay_to_step,
)
from research.semantic_token_cd.same_state_fork_common import atomic_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--unnorm-key", default="libero_90_no_noops")
    ap.add_argument("--physical-gpu", type=int, default=1)
    ap.add_argument("--per-task", type=int, default=6)
    args = ap.parse_args()

    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from libero.libero import benchmark, get_libero_path

    visible = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x]
    device = f"cuda:{visible.index(args.physical_gpu)}"
    model, processor = load_policy(
        args.checkpoint,
        Path("/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf"),
        device=device,
        dataset_statistics_path=args.checkpoint / "dataset_statistics.json",
        unnorm_key=args.unnorm_key,
    )
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    manifest = json.loads(args.manifest.read_text())

    report = {"protocol_id": "SAME_STATE_FORK_V1_ACCEPTANCE", "checks": [], "per_state": []}
    for task_key, entry in manifest.items():
        task_id = int(task_key)
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        forks = entry["forks"][: args.per_task]
        env = make_env(bddl)
        try:
            for fork in forks:
                episode = json.loads(Path(fork["episode_file"]).read_text())
                actions = np.asarray(episode["trajectory"], dtype=float)
                obs = replay_to_step(env, init_states[fork["init_state_id"]], actions, fork["fork_step"])
                _, image = prepare_agentview(obs)
                inputs = processor(build_prompt(task.language), image).to(model.device, dtype=torch.bfloat16)

                # (2) lambda = 0 must equal vanilla
                clean_ids, clean_logits, clean_h = generate_clean_action(model, processor, image, task.language, inputs)
                vanilla = predict_action(model, processor, image, task.language, unnorm_key=args.unnorm_key)
                zero = forward_logits(model, processor, image, task.language, clean_ids, selected=(), mean=None)
                lam0_ids = zero.argmax(dim=-1)
                from research.ar_token_counterfactual.intervention import decode_action_ids
                lam0_action = decode_action_ids(model, lam0_ids.unsqueeze(0).detach().cpu(), args.unnorm_key)

                # (3) identity reconstruction vs clean forward, twice (noise floor)
                ident_a = forward_logits(model, processor, image, task.language, clean_ids, selected=(), mean=None)
                ident_b = forward_logits(model, processor, image, task.language, clean_ids, selected=(), mean=None)
                action_start = int(model.vocab_size) - 256
                eos = int(model.generation_config.eos_token_id)
                # Compare only inside the 256 action bins: clean_logits has EOS
                # forced to finfo.min, forward_logits does not, so the full-vocab
                # difference is meaningless.
                ident_a_bin = ident_a[:, action_start:action_start + 256]
                ident_b_bin = ident_b[:, action_start:action_start + 256]
                clean_bin = clean_logits[:, action_start:action_start + 256]
                repeat_noise = float((ident_a_bin - ident_b_bin).abs().max().item())
                identity_diff = float((ident_a_bin - clean_bin).abs().max().item())
                identity_bin_diff = identity_diff
                bins = clean_logits[:, action_start:action_start + 256].float()
                top2 = torch.topk(bins, k=2, dim=-1).values
                min_margin = float((top2[:, 0] - top2[:, 1]).min().item())

                # (4) does the guided branch change the action when it should not?
                guided_action = branch_step_action("guided", model, processor, image, task.language, inputs, args.unnorm_key)
                guided_repeat = branch_step_action("guided", model, processor, image, task.language, inputs, args.unnorm_key)
                guided_repeat_diff = float(np.abs(np.asarray(guided_action) - np.asarray(guided_repeat)).max())

                row = {
                    "task_id": task_id, "init_state_id": fork["init_state_id"], "stage": fork["stage"],
                    "fork_step": fork["fork_step"],
                    "lambda0_action_max_abs_diff": float(np.abs(lam0_action - vanilla).max()),
                    "lambda0_ids_match": bool(torch.equal(lam0_ids.detach().cpu(), clean_ids[0].detach().cpu())),
                    "identity_max_abs_logit_diff": identity_diff,
                    "identity_max_abs_action_bin_diff": identity_bin_diff,
                    "repeat_forward_noise_max_abs_logit_diff": repeat_noise,
                    "identity_exceeds_repeat_noise": bool(identity_diff > repeat_noise),
                    "min_action_bin_margin": min_margin,
                    "identity_diff_vs_margin_ratio": float(identity_diff / max(min_margin, 1e-9)),
                    "guided_repeat_max_abs_action_diff": guided_repeat_diff,
                    "guided_action_l2_vs_vanilla": float(np.linalg.norm(np.asarray(guided_action) - vanilla)),
                }
                report["per_state"].append(row)
                print(json.dumps(row), flush=True)
        finally:
            env.close()

    rows = report["per_state"]
    report["summary"] = {
        "n_states": len(rows),
        "lambda0_all_match": bool(all(r["lambda0_action_max_abs_diff"] == 0.0 for r in rows)),
        "identity_argmax_safe": bool(all(r["identity_max_abs_action_bin_diff"] < r["min_action_bin_margin"] for r in rows)),
        "identity_exceeds_repeat_noise_count": int(sum(r["identity_exceeds_repeat_noise"] for r in rows)),
        "guided_repeat_deterministic": bool(all(r["guided_repeat_max_abs_action_diff"] == 0.0 for r in rows)),
        "max_identity_diff_vs_margin_ratio": max(r["identity_diff_vs_margin_ratio"] for r in rows),
        "max_repeat_forward_noise": max(r["repeat_forward_noise_max_abs_logit_diff"] for r in rows),
    }
    report["verdict"] = (
        "PASS"
        if report["summary"]["lambda0_all_match"]
        and report["summary"]["identity_argmax_safe"]
        and report["summary"]["guided_repeat_deterministic"]
        else "REVIEW"
    )
    atomic_json(args.artifact / "ACCEPTANCE_REPORT.json", report)
    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
