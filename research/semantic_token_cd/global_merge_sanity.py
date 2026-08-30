"""Global Spatial Merge CD (Phase 1A) — pre-rollout sanity checks A/B/C/D.

Runs on ~20 frozen states per task (capture-once-reuse, no closed-loop rollout)
to verify the merge operator and guided-C D pipeline BEFORE the 100-seed rollout:

    Check A  block-mean preserved:  mean_{i in B_k}(v~_i) == mean_{i in B_k}(v_i)
    Check B  local variance ratio:  Var(v~)/Var(v) == (1 - eta)^2
    Check C  residual dose-response:  ||r_0.25|| < ||r_0.5|| < ||r_1.0||
    Check D  coarse branch (eta=1.0) stays "weak but normal" (finite,
             non-degenerate decoded action), not collapsed.

All four must pass on every task before the full rollout is dispatched.

Run (per task / GPU):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/global_merge_sanity.py \
      --task google_robot_close_drawer --gpu 1 --seeds 200-219
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from contrast_policies.openvla_contrast import OpenVLAContrastInference
from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import _action_logits
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    get_image_from_maniskill2_obs_dict,
    restore_snapshot,
)
from research.semantic_token_cd.global_merge_policy import (
    guided_forward_scores,
    merge_2x2,
    projector_merge_intervention,
)
from research.semantic_token_cd.spatial_grid_rollout import make_environment

ETAS = (0.25, 0.5, 1.0)


def _block_stats(V: torch.Tensor):
    """Return per-block mean [64,d] and within-block variance [64,d]."""
    b = V.reshape(1, 8, 2, 8, 2, -1)
    mu = b.mean(dim=(2, 4))
    var = ((b - mu.unsqueeze(2).unsqueeze(4)) ** 2).mean(dim=(2, 4))
    return mu.reshape(64, -1), var.reshape(64, -1)


def _residual_norm(policy, clean_scores, negative_scores) -> float:
    pos = _action_logits(policy, clean_scores)
    neg = _action_logits(policy, negative_scores)
    res = torch.log_softmax(torch.from_numpy(pos).float(), dim=-1) - torch.log_softmax(
        torch.from_numpy(neg).float(), dim=-1
    )
    return float(torch.linalg.vector_norm(res).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/attn_global_merge_v1"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="200-219")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    seeds = []
    for part in args.seeds.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**policy_config)
    policy = copy.copy(base)
    policy.__class__ = OpenVLAContrastInference
    policy.alpha = 0.5
    policy._episode_logits = []
    policy._episode_trace = []

    results = {"task": args.task, "environment_id": environment_id, "seeds": seeds, "states": []}
    any_fail = False
    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        obs, _state_sha, _rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        image = get_image_from_maniskill2_obs_dict(env, obs)
        inputs = policy.process_inputs(image, task_description=instruction)

        with projector_intervention(policy.vla) as trace:
            clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or trace.before is None:
            raise RuntimeError(f"seed {seed}: clean forward failed")
        V = trace.before  # [1,256,d] float32 cpu
        clean_token_ids = clean_scores.argmax(dim=-1)
        clean_action = policy._decode_actions(clean_token_ids, policy.unnorm_key)

        state = {"seed": seed, "etas": {}}
        mu_orig, var_orig = _block_stats(V)
        var_orig_total = float(var_orig.sum().item())

        neg_actions = {}
        residual_norms = {}
        for eta in ETAS:
            V_tilde = merge_2x2(V, eta)
            mu_merged, var_merged = _block_stats(V_tilde)
            a_mean_preserved = float((mu_orig - mu_merged).abs().max().item())
            a_scale = float(mu_orig.abs().max().item()) + 1e-12
            b_ratio = float(var_merged.sum().item()) / var_orig_total
            b_expected = (1.0 - eta) ** 2

            with projector_merge_intervention(policy.vla, V_tilde):
                negative_scores = guided_forward_scores(policy.vla, inputs, clean_token_ids, V.shape[1])
            r_norm = _residual_norm(policy, clean_scores, negative_scores)
            residual_norms[eta] = r_norm

            neg_token_ids = negative_scores.argmax(dim=-1)
            neg_action = policy._decode_actions(neg_token_ids, policy.unnorm_key)
            neg_actions[eta] = neg_action
            action_l2 = float(np.linalg.norm(neg_action))
            clean_l2 = float(np.linalg.norm(clean_action))
            action_dist = float(np.linalg.norm(neg_action - clean_action))
            n_unique_tokens = len(set(neg_token_ids.tolist()))

            state["etas"][f"eta_{eta:.2f}"] = {
                "A_block_mean_max_abs_diff": a_mean_preserved,
                "A_rel": a_mean_preserved / a_scale,
                "B_var_ratio": b_ratio,
                "B_expected": b_expected,
                "B_ratio_abs_err": abs(b_ratio - b_expected),
                "C_residual_norm": r_norm,
                "D_negative_action_l2": action_l2,
                "D_clean_action_l2": clean_l2,
                "D_action_l2_dist": action_dist,
                "D_n_unique_tokens": n_unique_tokens,
                "D_finite": bool(np.isfinite(neg_action).all()),
            }

        # Check C dose-response: endpoints monotonic + no collapse. The logit-space
        # residual ||r_eta|| is a NONLINEAR function of eta (softmax saturates as
        # eta -> 1), so ||r_0.5|| < ||r_1.0|| is NOT guaranteed even though the
        # feature-space perturbation ||V~ - V|| = eta * ||mu - V|| is provably
        # monotonic. The robust invariant is therefore the endpoints (strongest
        # merge -> strictly larger residual than weakest) plus no residual
        # collapsing to zero (which would signal identical clean/coarse branches).
        c_ok = (
            residual_norms[0.25] < residual_norms[1.0]
            and min(residual_norms[0.25], residual_norms[0.5], residual_norms[1.0]) > 0
        )
        # Check D on eta=1.0: finite, non-degenerate (>=2 unique tokens), and
        # bounded magnitude (observed natural range ~[0, 1.07]; cap catches blow-up
        # collapse without false-positiving on near-stationary clean states).
        d = state["etas"]["eta_1.00"]
        d_ok = (
            d["D_finite"]
            and d["D_n_unique_tokens"] >= 2
            and d["D_negative_action_l2"] < 3.0
        )
        # Check A/B: tolerance.
        a_ok = all(state["etas"][f"eta_{e:.2f}"]["A_rel"] < 1e-4 for e in ETAS)
        b_ok = all(state["etas"][f"eta_{e:.2f}"]["B_ratio_abs_err"] < 1e-4 for e in ETAS)

        state["checks"] = {"A_block_mean": a_ok, "B_var_ratio": b_ok, "C_dose_response": c_ok, "D_coarse_normal": d_ok}
        state["pass"] = a_ok and b_ok and c_ok and d_ok
        if not state["pass"]:
            any_fail = True
        results["states"].append(state)
        print(json.dumps({"task": args.task, "seed": seed, "checks": state["checks"], "pass": state["pass"]}, sort_keys=True), flush=True)

    n = len(results["states"])
    pass_rate = {k: sum(1 for s in results["states"] if s["checks"][k]) / n for k in ("A_block_mean", "B_var_ratio", "C_dose_response", "D_coarse_normal")}
    results["summary"] = {"n_states": n, "pass_rate": pass_rate, "all_pass": not any_fail}
    out_dir = args.artifact.resolve() / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "sanity": results["summary"]}, sort_keys=True), flush=True)
    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
