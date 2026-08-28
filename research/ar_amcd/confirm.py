#!/usr/bin/env python3
"""AMCD Phase-0 confirmation: gates A-G + lambda headroom on the 200 held-out states.

At the locked layer l* (from AMCD_LAYER_LOCK.yaml), for every Confirmation state:
  - expert branch (teacher-forced EXPERT prefix): z_final, z_l*, logZ_final, logZ_l*
  - vanilla branch (model greedy prefix):        z_final_v, z_l*_v  (for gate F)

Gates A-G per CONFIG_LOCK.yaml; lambda headroom is a secondary offline diagnostic
(not used for the GO/NO-GO verdict).

Run:
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/ar_amcd/confirm.py \
      --workspace . --artifact artifacts/ar_amcd_action_maturity_phase0_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, set_determinism
from research.ar_token_counterfactual.intervention import ensure_empty_action_token
from research.ar_amcd.core import (
    ACTION_LO, build_multimodal, readout_specific_layers, encode_action_tokens, load_action_stats,
)
from research.ar_amcd.scan import (
    FINAL, OFFSETS, gather, softmax_256, state_permutations,
)

CHECKPOINT = "openvla-7b-finetuned-libero-object/287d6cfdf12d07b1449505f66d9bf3550257e9b3"
FROZEN_ARTIFACT = "artifacts/ar_sid_token_cd_phase0_v1_20260822T145651+0800"
LAMBDAS_PREREGISTERED = [0.05, 0.10, 0.25]
LAMBDAS_EXPLORATORY = [-0.25, -0.10, 0.0, 0.05, 0.10, 0.25, 0.50, 1.00]


def read_lock(art: Path) -> int:
    sel = json.loads((art / "selection_report.json").read_text())
    return int(sel["selected_layer"])


def guided_expert_logp(z_fin, z_l, logZ_fin, t_star, lam):
    """Full-vocab log-prob of expert token under z_guided = z_fin + lam*(z_fin - z_l)."""
    z_fin = z_fin.astype(np.float64)
    z_l = z_l.astype(np.float64)
    d = z_fin - z_l
    lp_fin = z_fin - float(logZ_fin)
    action_mass = float(np.exp(lp_fin).sum())
    guided_shift = float(np.exp(lp_fin + lam * d).sum()) + (1.0 - action_mass)
    return float((z_fin[t_star] + lam * d[t_star] - float(logZ_fin)) - np.log(guided_shift))


def bootstrap_median_ci(vals, n_boot=10000, seed=20260822):
    """vals: pooled 1-D array. Returns (median, lo, hi) of the median under state resampling
    is NOT done here — caller passes per-state groups; this takes a [N,7] array and
    bootstraps over the first axis."""
    rng = np.random.default_rng(seed)
    n = vals.shape[0]
    meds = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        meds[i] = np.median(vals[idx].astype(np.float64).ravel())
    return float(np.median(vals.astype(np.float64).ravel())), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--max-states", type=int, default=200)
    a = p.parse_args()
    w = a.workspace.resolve()
    art = a.artifact.resolve()

    locked = read_lock(art)
    layers = [locked, FINAL]

    ckpt_dir = w / f"checkpoints/{CHECKPOINT}"
    low, high, mask = load_action_stats(ckpt_dir)

    split = json.loads((w / FROZEN_ARTIFACT / "frozen_state_split.json").read_text())
    conf_ids = split["confirmation"][: a.max_states]
    by_id = {}
    with (w / FROZEN_ARTIFACT / "state_manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            by_id[row["state_id"]] = row
    rows = [by_id[sid] for sid in conf_ids]
    n_states = len(rows)

    set_determinism(20260822)
    model, processor = load_policy(ckpt_dir, w / "third_party/openvla/prismatic/extern/hf")

    # accumulators over (n, j) pairs
    z_fin_all = np.zeros((n_states, 7, 256), dtype=np.float32)
    z_l_all = np.zeros((n_states, 7, 256), dtype=np.float32)
    logZ_fin_all = np.zeros((n_states, 7), dtype=np.float32)
    logZ_l_all = np.zeros((n_states, 7), dtype=np.float32)
    expert_local_all = np.zeros((n_states, 7), dtype=np.int32)
    tasks = [r["task_id"] for r in rows]
    state_ids = [r["state_id"] for r in rows]

    # gate F: vanilla branch
    z_fin_v_all = np.zeros((n_states, 7, 256), dtype=np.float32)
    z_l_v_all = np.zeros((n_states, 7, 256), dtype=np.float32)

    for si, row in enumerate(rows):
        language = row["language"]
        expert_raw = np.asarray(json.loads(row["expert_action"]), dtype=np.float64)
        fine_image = Image.open(w / FROZEN_ARTIFACT / row["image"]).convert("RGB")
        inputs = processor(build_prompt(language), fine_image).to(model.device, dtype=torch.bfloat16)
        base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
        pv = inputs["pixel_values"]

        expert_tokens = torch.tensor(
            encode_action_tokens(expert_raw, low, high, mask), dtype=base_ids.dtype, device=base_ids.device
        ).unsqueeze(0)
        expert_local_all[si] = (expert_tokens[0].cpu().numpy() - ACTION_LO).astype(np.int32)

        # ---- expert branch ----
        teacher_ids = torch.cat([base_ids, expert_tokens[:, :-1]], dim=1)
        teacher_mask = torch.cat([base_mask, torch.ones((1, 6), device=base_mask.device, dtype=base_mask.dtype)], dim=1)
        mm_embeds, mm_mask, n_visual = build_multimodal(model, teacher_ids, teacher_mask, pv)
        assert n_visual == 256
        aq = list(range(mm_embeds.shape[1] - 7, mm_embeds.shape[1]))
        z_dict, logz_dict = readout_specific_layers(model, mm_embeds, mm_mask, aq, layers)
        z_fin_all[si] = z_dict[FINAL]
        z_l_all[si] = z_dict[locked]
        logZ_fin_all[si] = logz_dict[FINAL]
        logZ_l_all[si] = logz_dict[locked]

        # ---- vanilla branch ----
        gen = model.generate(input_ids=base_ids, attention_mask=base_mask, pixel_values=pv,
                             max_new_tokens=7, do_sample=False)
        vanilla_tokens = gen[:, -7:]  # 7 tokens
        v_teacher_ids = torch.cat([base_ids, vanilla_tokens[:, :-1]], dim=1)
        v_teacher_mask = torch.cat([base_mask, torch.ones((1, 6), device=base_mask.device, dtype=base_mask.dtype)], dim=1)
        v_mm, v_mm_mask, v_nvis = build_multimodal(model, v_teacher_ids, v_teacher_mask, pv)
        assert v_nvis == 256
        v_aq = list(range(v_mm.shape[1] - 7, v_mm.shape[1]))
        vz_dict, _ = readout_specific_layers(model, v_mm, v_mm_mask, v_aq, layers)
        z_fin_v_all[si] = vz_dict[FINAL]
        z_l_v_all[si] = vz_dict[locked]

        if (si + 1) % 20 == 0 or si + 1 == n_states:
            print(json.dumps({"done": si + 1, "of": n_states}), flush=True)

    # ---- aggregate gates ----
    expert = expert_local_all  # [N,7] local index
    k = np.arange(256)[None, None, :]
    is_expert = k == expert[:, :, None]

    def med(a):
        return float(np.median(a.astype(np.float64).ravel()))

    # maturity on expert branch
    expert_logp_l = gather(z_l_all, expert) - logZ_l_all
    expert_logp_f = gather(z_fin_all, expert) - logZ_fin_all
    logp_gain = expert_logp_f - expert_logp_l                       # [N,7]

    margin_l = gather(z_l_all, expert) - np.where(is_expert, -np.inf, z_l_all).max(axis=2)
    margin_f = gather(z_fin_all, expert) - np.where(is_expert, -np.inf, z_fin_all).max(axis=2)
    margin_gain = margin_f - margin_l                               # [N,7]

    d_l = z_fin_all - z_l_all                                       # [N,7,256]
    expert_d = gather(d_l, expert)
    nb_idx = np.clip(expert[:, :, None] + OFFSETS[None, None, :], 0, 255)
    specific = expert_d - np.take_along_axis(d_l, nb_idx, axis=2).mean(axis=2)

    perms = state_permutations(state_ids)
    perm_expert = np.take_along_axis(perms, expert, axis=1)
    perm_d = np.take_along_axis(d_l, perm_expert[:, :, None], axis=2)[:, :, 0]
    pnb_idx = np.clip(perm_expert[:, :, None] + OFFSETS[None, None, :], 0, 255)
    perm_specific = perm_d - np.take_along_axis(d_l, pnb_idx, axis=2).mean(axis=2)

    positive_rate = float((logp_gain > 0).mean())
    margin_gain_med = med(margin_gain)
    _, mg_lo, mg_hi = bootstrap_median_ci(margin_gain)
    specific_med = med(specific)
    perm_specific_med = med(perm_specific)

    # gate D: per-task median ExpertLogPGain
    tasks_arr = np.asarray(tasks)
    per_task_logp = {t: float(np.median(logp_gain[tasks_arr == t].astype(np.float64)))
                     for t in sorted(set(tasks))}
    n_task_pos = sum(1 for v in per_task_logp.values() if v > 0)

    # gate E: per-dim median ExpertLogPGain (pool over states for each position j)
    per_dim_logp = [float(np.median(logp_gain[:, j].astype(np.float64))) for j in range(7)]
    n_dim_pos = sum(1 for v in per_dim_logp if v > 0)

    # gate F: CorrectionPreference on vanilla branch
    d_l_v = z_fin_v_all - z_l_v_all
    z_guided_v = z_fin_v_all + 1.0 * d_l_v
    z_fin_vanilla_pred = z_fin_v_all.argmax(axis=2)                 # [N,7] local argmax
    vanilla_wrong = z_fin_vanilla_pred != expert                    # [N,7] bool
    guided_pred = z_guided_v.argmax(axis=2)
    n_wrong = int(vanilla_wrong.sum())
    n_corrected = int((vanilla_wrong & (guided_pred == expert)).sum())
    correction_pref = float(n_corrected / n_wrong) if n_wrong else float("nan")

    # extra diagnostic: does the residual correct expert-branch wrong positions?
    expert_wrong = z_fin_all.argmax(axis=2) != expert
    z_guided_e = z_fin_all + 1.0 * d_l
    n_expert_wrong = int(expert_wrong.sum())
    n_expert_corrected = int((expert_wrong & (z_guided_e.argmax(axis=2) == expert)).sum())
    expert_correction_pref = float(n_expert_corrected / n_expert_wrong) if n_expert_wrong else float("nan")

    # gate G: non-degenerate
    resid_norm = np.linalg.norm(d_l, axis=2)                        # [N,7]
    top1_l = float((z_l_all.argmax(axis=2) == expert).mean())
    non_degenerate = bool(locked < FINAL and med(resid_norm) > 0 and top1_l < 0.99)

    # lambda headroom (expert branch, full-vocab log-prob + Top1Acc), exploratory sweep
    headroom = {}
    for lam in LAMBDAS_EXPLORATORY:
        lp = [guided_expert_logp(z_fin_all[si, j], z_l_all[si, j], logZ_fin_all[si, j],
                                 int(expert[si, j]), lam)
              for si in range(n_states) for j in range(7)]
        z_g = z_fin_all + lam * d_l
        headroom[f"lambda_{lam}"] = {
            "expert_logp_mean": float(np.mean(lp)),
            "expert_logp_median": float(np.median(lp)),
            "top1acc": float((z_g.argmax(axis=2) == expert).mean()),
            "pre_registered": lam in LAMBDAS_PREREGISTERED or lam == 0.0,
        }

    gates = {
        "A": {"PositiveRate": positive_rate, "PASS": positive_rate >= 0.65},
        "B": {"median_ExpertMarginGain": margin_gain_med,
              "bootstrap_ci_low": mg_lo, "bootstrap_ci_high": mg_hi,
              "PASS": mg_lo > 0},
        "C": {"median_ExpertSpecificGain": specific_med,
              "median_PermutedExpertSpecificGain": perm_specific_med,
              "PASS": specific_med > 0 and specific_med > perm_specific_med},
        "D": {"per_task_median_logp_gain": per_task_logp, "n_tasks_positive": n_task_pos,
              "PASS": n_task_pos >= 8},
        "E": {"per_dim_median_logp_gain": per_dim_logp, "n_dims_positive": n_dim_pos,
              "PASS": n_dim_pos >= 5},
        "F": {"n_vanilla_wrong_positions": n_wrong, "n_corrected": n_corrected,
              "CorrectionPreference": correction_pref, "PASS": correction_pref >= 0.55,
              "expert_branch_correction_diagnostic": {
                  "n_wrong": n_expert_wrong, "n_corrected": n_expert_corrected,
                  "CorrectionPreference": expert_correction_pref,
              }},
        "G": {"locked_layer": locked, "final_layer": FINAL,
              "median_residual_norm": med(resid_norm), "top1acc_at_locked_layer": top1_l,
              "PASS": non_degenerate},
    }
    all_pass = all(g["PASS"] for g in gates.values())

    result = {
        "experiment": "AR_AMCD_ACTION_MATURITY_PHASE0_V1",
        "step": "confirmation gates A-G + lambda headroom",
        "locked_layer": locked,
        "n_states": n_states,
        "gates": gates,
        "lambda_headroom": headroom,
        "verdict": "AMCD_PHASE0_GO" if all_pass else "STOP_AMCD_NO_GO",
        "all_gates_pass": bool(all_pass),
    }
    (art / "confirmation_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        art / "confirmation_data.npz",
        z_fin=z_fin_all, z_l=z_l_all, logZ_fin=logZ_fin_all, logZ_l=logZ_l_all,
        expert_local=expert_local_all, tasks=np.asarray(tasks), state_ids=np.asarray(state_ids),
        z_fin_vanilla=z_fin_v_all, z_l_vanilla=z_l_v_all,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
