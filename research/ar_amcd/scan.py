#!/usr/bin/env python3
"""AMCD Phase-0 layer scan + maturity curve + deterministic selection.

For the 100 Selection states (teacher-forced EXPERT prefix), read out z_l at all
32 blocks via the model's own final_norm + lm_head (logit lens), and compute:

  per-layer  : ExpertLP, ExpertRank, ExpertMargin, Top1Acc, Entropy, ActionMass
  maturity   : ExpertLogPGain, ExpertMarginGain, PositiveRate (final=31 vs l)
  control 1  : ExpertSpecificGain (expert vs a*+-{1,2,3,4} local bins)
  control 2  : PermutedExpertSpecificGain (per-state label permutation)
  residual   : SignedExpertAdvantage

Then apply the pre-registered selection rule over candidates {7,11,15,19,23,27}
and write AMCD_LAYER_LOCK.yaml (or early-stop NO_GO).

Run:
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/ar_amcd/scan.py \
      --workspace . --artifact artifacts/ar_amcd_action_maturity_phase0_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, set_determinism
from research.ar_token_counterfactual.intervention import ensure_empty_action_token
from research.ar_amcd.core import (
    ACTION_LO, build_multimodal, scan_layer_readouts, encode_action_tokens, load_action_stats,
)

CHECKPOINT = "openvla-7b-finetuned-libero-object/287d6cfdf12d07b1449505f66d9bf3550257e9b3"
FROZEN_ARTIFACT = "artifacts/ar_sid_token_cd_phase0_v1_20260822T145651+0800"
N_LAYERS = 32
FINAL = N_LAYERS - 1
CANDIDATES = [7, 11, 15, 19, 23, 27]
OFFSETS = np.array([-4, -3, -2, -1, 1, 2, 3, 4])


def gather(z, idx):
    """z [N,7,256], idx [N,7] -> [N,7] = z[:,:,idx[:,:,0]]."""
    return np.take_along_axis(z, idx[:, :, None], axis=2)[:, :, 0]


def softmax_256(z):
    """z [N,7,256] -> row-normalized softmax over last axis."""
    z = z.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def state_permutations(state_ids):
    perms = []
    for sid in state_ids:
        seed = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
        perms.append(np.random.default_rng(seed).permutation(256))
    return np.stack(perms)  # [N, 256]


def compute_metrics(z_all, logZ_all, expert_all, tasks, state_ids):
    """All per-layer metrics. z_all/logZ_all/expert_all: [N,32,7,256]/[N,32,7]/[N,7]."""
    n = z_all.shape[0]
    z_final = z_all[:, FINAL]
    logZ_final = logZ_all[:, FINAL]
    perms = state_permutations(state_ids)

    # final-layer quantities (hoisted out of the per-layer loop)
    k = np.arange(256)[None, None, :]
    is_expert = k == expert_all[:, :, None]
    expert_logit_f = gather(z_final, expert_all)              # [N,7]
    expert_logp_f = expert_logit_f - logZ_final               # [N,7]
    max_excl_f = np.where(is_expert, -np.inf, z_final).max(axis=2)
    margin_f = expert_logit_f - max_excl_f

    rows = []
    for l in range(N_LAYERS):
        z_l = z_all[:, l]
        logZ_l = logZ_all[:, l]

        expert_logit = gather(z_l, expert_all)                 # [N,7]
        expert_logp = expert_logit - logZ_l                    # [N,7]
        # ExpertRank: 1 + #{k : z > z[t*]}
        rank = 1 + (z_l > expert_logit[:, :, None]).sum(axis=2)
        # ExpertMargin
        max_excl = np.where(is_expert, -np.inf, z_l).max(axis=2)
        margin = expert_logit - max_excl
        # Top1Acc
        top1 = (z_l.argmax(axis=2) == expert_all).mean()
        # Entropy (action-vocab conditional)
        p = softmax_256(z_l)
        entropy = -(p * np.log(p + 1e-12)).sum(axis=2)
        # ActionMass (full-vocab softmax mass on action bins)
        mass = np.exp(z_l - logZ_l[:, :, None]).sum(axis=2)

        # Maturity (final vs l)
        logp_gain = expert_logp_f - expert_logp                 # [N,7]
        margin_gain = margin_f - margin
        positive_rate = float((logp_gain > 0).mean())

        # Residual d_l = z_final - z_l (logit space)
        d_l = z_final - z_l                                    # [N,7,256]
        expert_d = gather(d_l, expert_all)                     # [N,7]
        # Control 1: ExpertSpecificGain (expert vs local neighbours)
        nb_idx = np.clip(expert_all[:, :, None] + OFFSETS[None, None, :], 0, 255)
        nb_val = np.take_along_axis(d_l, nb_idx, axis=2).mean(axis=2)
        specific = expert_d - nb_val
        # Control 2: PermutedExpertSpecificGain
        perm_expert = np.take_along_axis(perms, expert_all, axis=1)   # [N,7]
        perm_d = np.take_along_axis(d_l, perm_expert[:, :, None], axis=2)[:, :, 0]
        pnb_idx = np.clip(perm_expert[:, :, None] + OFFSETS[None, None, :], 0, 255)
        pnb_val = np.take_along_axis(d_l, pnb_idx, axis=2).mean(axis=2)
        perm_specific = perm_d - pnb_val
        # Residual direction: SignedExpertAdvantage
        signed = expert_d - (d_l.sum(axis=2) - expert_d) / 255.0

        def med(a):
            return float(np.median(a.astype(np.float64).ravel()))

        rows.append({
            "layer": l,
            "ExpertLP": med(expert_logp),
            "ExpertRank": med(rank),
            "ExpertMargin": med(margin),
            "Top1Acc": float(top1),
            "Entropy": med(entropy),
            "ActionMass": med(mass),
            "ExpertLogPGain": med(logp_gain),
            "ExpertMarginGain": med(margin_gain),
            "PositiveRate": positive_rate,
            "ExpertSpecificGain": med(specific),
            "PermutedExpertSpecificGain": med(perm_specific),
            "SignedExpertAdvantage": med(signed),
        })

    # per-task median ExpertLogPGain at each candidate layer (for reference)
    tasks_arr = np.asarray(tasks)
    per_task = {}
    for l in CANDIDATES:
        expert_logp_l = gather(z_all[:, l], expert_all) - logZ_all[:, l]
        logp_gain_l = (gather(z_final, expert_all) - logZ_final) - expert_logp_l  # [N,7]
        per_task[l] = {
            t: float(np.median(logp_gain_l[tasks_arr == t].astype(np.float64)))
            for t in sorted(set(tasks))
        }

    return rows, per_task


def select_layer(rows):
    """Deterministic selection rule over candidates. Returns (selected_row, passing, all_rows)."""
    by_layer = {r["layer"]: r for r in rows}
    candidates = [by_layer[l] for l in CANDIDATES]
    passing = [r for r in candidates
               if r["ExpertMarginGain"] > 0
               and r["PositiveRate"] > 0.60
               and r["ExpertSpecificGain"] > 0]
    if not passing:
        return None, [], candidates
    best_mg = max(r["ExpertMarginGain"] for r in passing)
    tier = [r for r in passing if abs(r["ExpertMarginGain"] - best_mg) <= 1e-12]
    best_sg = max(r["ExpertSpecificGain"] for r in tier)
    tier = [r for r in tier if abs(r["ExpertSpecificGain"] - best_sg) <= 1e-12]
    tier = sorted(tier, key=lambda r: -r["layer"])  # deeper layer wins
    return tier[0], passing, candidates


def _to_yaml(obj, indent=0):
    lines = []
    pad = "  " * indent
    for k, v in obj.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.extend(_to_yaml(v, indent + 1).splitlines())
        elif isinstance(v, (list, tuple)):
            lines.append(f"{pad}{k}:")
            for item in v:
                lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--max-states", type=int, default=100)
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()
    w = a.workspace.resolve()
    art = a.artifact.resolve()
    art.mkdir(parents=True, exist_ok=True)

    ckpt_dir = w / f"checkpoints/{CHECKPOINT}"
    low, high, mask = load_action_stats(ckpt_dir)

    split = json.loads((w / FROZEN_ARTIFACT / "frozen_state_split.json").read_text())
    selection_ids = split["selection"][: a.max_states]
    by_id = {}
    with (w / FROZEN_ARTIFACT / "state_manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            by_id[row["state_id"]] = row
    rows = [by_id[sid] for sid in selection_ids]
    n_states = len(rows)

    npz_path = art / "scan_z.npz"

    if not a.aggregate_only:
        set_determinism(20260822)
        model, processor = load_policy(ckpt_dir, w / "third_party/openvla/prismatic/extern/hf")

        z_all = np.zeros((n_states, N_LAYERS, 7, 256), dtype=np.float32)
        logZ_all = np.zeros((n_states, N_LAYERS, 7), dtype=np.float32)
        expert_all = np.zeros((n_states, 7), dtype=np.int32)
        tasks = [r["task_id"] for r in rows]
        state_ids = [r["state_id"] for r in rows]

        for si, row in enumerate(rows):
            language = row["language"]
            expert_raw = np.asarray(json.loads(row["expert_action"]), dtype=np.float64)
            fine_image = Image.open(w / FROZEN_ARTIFACT / row["image"]).convert("RGB")

            inputs = processor(build_prompt(language), fine_image).to(model.device, dtype=torch.bfloat16)
            base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
            pv = inputs["pixel_values"]

            expert_tokens = torch.tensor(
                encode_action_tokens(expert_raw, low, high, mask), dtype=base_ids.dtype, device=base_ids.device
            ).unsqueeze(0)  # [1, 7]
            teacher_ids = torch.cat([base_ids, expert_tokens[:, :-1]], dim=1)
            teacher_mask = torch.cat(
                [base_mask, torch.ones((1, 6), device=base_mask.device, dtype=base_mask.dtype)], dim=1)

            mm_embeds, mm_mask, n_visual = build_multimodal(model, teacher_ids, teacher_mask, pv)
            assert n_visual == 256
            seq_len = mm_embeds.shape[1]
            aq = list(range(seq_len - 7, seq_len))

            z, logZ = scan_layer_readouts(model, mm_embeds, mm_mask, aq)
            z_all[si] = z
            logZ_all[si] = logZ
            expert_all[si] = (expert_tokens[0].cpu().numpy() - ACTION_LO).astype(np.int32)

            if (si + 1) % 10 == 0 or si + 1 == n_states:
                print(json.dumps({"done": si + 1, "of": n_states}), flush=True)

        np.savez_compressed(
            npz_path, z_all=z_all, logZ_all=logZ_all, expert_all=expert_all,
            state_ids=np.asarray(state_ids), tasks=np.asarray(tasks),
        )
        print(json.dumps({"forward_done": n_states, "npz": str(npz_path)}), flush=True)

    # ---- aggregate ----
    data = np.load(npz_path)
    z_all = data["z_all"]; logZ_all = data["logZ_all"]; expert_all = data["expert_all"]
    tasks = list(data["tasks"]); state_ids = list(data["state_ids"])
    n_states = z_all.shape[0]

    metric_rows, per_task = compute_metrics(z_all, logZ_all, expert_all, tasks, state_ids)

    # write maturity curve CSV
    csv_path = art / "scan_maturity_curve.csv"
    fields = list(metric_rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=fields)
        wcsv.writeheader()
        for r in metric_rows:
            wcsv.writerow(r)

    selected, passing, candidates = select_layer(metric_rows)

    result = {
        "experiment": "AR_AMCD_ACTION_MATURITY_PHASE0_V1",
        "step": "layer_scan + selection",
        "n_states": n_states,
        "n_layers": N_LAYERS,
        "final_layer": FINAL,
        "candidates": CANDIDATES,
        "prefix": "teacher_forced_expert",
        "pooling": "per (state, position) median over N*7 values; PositiveRate/Top1Acc are fractions",
        "metric_rows": metric_rows,
        "per_task_ExpertLogPGain_at_candidates": per_task,
        "passing_candidates": [r["layer"] for r in passing],
        "candidate_rows": candidates,
        "selected": selected,
    }
    (art / "scan_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if selected is None:
        report = {
            "status": "NO_GO",
            "reason": "no candidate layer passed the selection filter "
                      "(median ExpertMarginGain>0 & PositiveRate>0.60 & median ExpertSpecificGain>0)",
            "candidates": CANDIDATES,
            "candidate_rows": candidates,
        }
        (art / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
        return

    lock = {
        "protocol_id": "AMCD-LAYER-LOCK-R1",
        "selected_layer": selected["layer"],
        "final_layer": FINAL,
        "candidates": CANDIDATES,
        "selection_rule": "filter(median ExpertMarginGain>0, PositiveRate>0.60, median ExpertSpecificGain>0) "
                          "-> max median ExpertMarginGain -> tie-break median ExpertSpecificGain -> deeper layer",
        "selected_metrics": {
            "ExpertMarginGain": selected["ExpertMarginGain"],
            "PositiveRate": selected["PositiveRate"],
            "ExpertSpecificGain": selected["ExpertSpecificGain"],
            "ExpertLogPGain": selected["ExpertLogPGain"],
            "PermutedExpertSpecificGain": selected["PermutedExpertSpecificGain"],
            "SignedExpertAdvantage": selected["SignedExpertAdvantage"],
        },
        "readout": "z_l(j) = lm_head(final_norm(h_l(j))); residual d_l = z_31 - z_l",
        "action_vocab_slice": [ACTION_LO, ACTION_LO + 255],
        "split_sha256": None,
    }
    (art / "AMCD_LAYER_LOCK.yaml").write_text(_to_yaml(lock) + "\n")
    sel_report = {"status": "SELECTED", "selected_layer": selected["layer"],
                  "selected_metrics": lock["selected_metrics"],
                  "passing_candidates": [r["layer"] for r in passing]}
    (art / "selection_report.json").write_text(json.dumps(sel_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(sel_report, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
