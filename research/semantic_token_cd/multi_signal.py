#!/usr/bin/env python3
"""MULTI_SIGNAL_SEMANTIC_CD Phase-0 (compute + aggregate).

Per Confirmation state, compute a per-group language score L_i = max_e cos(v_i, e)
(reusing Phase-2B group features v_i + entity embeddings e) and a per-group action
score A_i = l2_mean (Phase-0 action disruption). Min-max normalize each per state,
fuse S_i = α·L̂ + (1-α)·Â (α sweep), select G* = argmax, residual r, align cos(r,r_PCD).

Single-group fusion reuses masked_kmeans_8 (no forward). Intersection/union ablations
select sets of groups and use forward_masked for the multi-group union.

Oracle 2×2 analysis (Section 7 of CONFIG_LOCK): language top-1 vs action top-1
correctness (== g_obj_idx), to test whether the two signals are complementary.

No rollout, no training, no λ, no external VLM. object mask = evaluation only.

Run (smoke):
  cd /home/leju-suzhou/zjt_ws/token-cd
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/multi_signal.py \
      --artifact artifacts/multi_signal_semantic_cd_phase0_v1 \
      --split confirmation --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from research.cw_lpcd.core import PCD_ROOT, CHECKPOINT, load_model, action_token_slice
from research.cw_lpcd.engine import load_position_mean, random_seed_for
from research.cw_lpcd.metrics import per_position_cosine
from research.ar_cat_cd.core import forward_masked
from research.semantic_token_cd.entity_set import (
    extract_entities, embed_phrase, log_softmax_residual, _mean, _med)

STATES_LOCK = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl")
MEAN_PATH = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/position_conditioned_visual_mean.pt")
SEMANTIC_NPZ = Path("artifacts/semantic_token_cd_phase0_v1/semantic_npz")
SPLIT = Path("artifacts/cat_cd_phase0_v1/FROZEN_SPLIT.json")
H_REUSE = Path("artifacts/semantic_token_cd_phase2a_relation_grouping_v1/h_npz")
ACTION_NPZ = Path("artifacts/action_conditioned_semantic_cd_phase0_v1/action_npz")
ENTITY_NPZ = Path("artifacts/semantic_token_cd_phase2b_entity_set_v1/entity_npz")

K = 8
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TOPK = 2
HARD_TASKS = ["widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "google_robot_move_near",
              "google_robot_close_drawer", "google_robot_open_drawer",
              "google_robot_place_apple_in_closed_top_drawer"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def alpha_key(alpha: float) -> str:
    return f"a{int(round(alpha * 100)):03d}"


def group_means(h: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """group mean features v [K, 4096] (identical to Phase-2B)."""
    v = np.zeros((K, h.shape[1]), dtype=np.float64)
    for k in range(K):
        idx = np.flatnonzero(labels == k)
        v[k] = h[idx].mean(axis=0) if len(idx) else 0.0
    return v


def language_scores(v: np.ndarray, emb: list[np.ndarray]) -> np.ndarray:
    """L_i = max_e cos(v_i, e) over the task's entities. [K]."""
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    L = np.full(K, -1.0, dtype=np.float64)
    for e in emb:
        en = e / (np.linalg.norm(e) + 1e-8)
        c = vn @ en
        L = np.maximum(L, c)
    return L


def minmax01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def iou_group(labels: np.ndarray, g: int, obj: set) -> float:
    s = set(int(x) for x in np.flatnonzero(labels == g).tolist())
    inter = len(s & obj)
    union = len(s | obj)
    return inter / union if union else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pcd-root", type=Path, default=PCD_ROOT)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    p.add_argument("--limit", type=int)
    p.add_argument("--state-id", type=str)
    p.add_argument("--mean-path", type=Path, default=MEAN_PATH)
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()

    pcd_root = a.pcd_root.resolve()
    art = a.artifact.resolve()
    fus_dir = art / "fusion_npz"
    fus_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT.read_text())
    (art / "FROZEN_SPLIT.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    if a.state_id:
        state_ids = [a.state_id]
    else:
        key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
        state_ids = split[key][: a.limit] if a.limit else split[key]

    states = {r["state_id"]: r for r in read_jsonl(STATES_LOCK)}

    model = processor = tokenizer = None
    n_fwd = 0
    if not a.eval_only:
        torch.manual_seed(20260824)
        torch.cuda.manual_seed_all(20260824)
        model, processor = load_model(CHECKPOINT)
        tokenizer = processor.tokenizer
        mean = load_position_mean(a.mean_path.resolve()).to(model.device, dtype=torch.bfloat16)
        token_slice = action_token_slice(model)

        # entity embedding cache per unique instruction
        instr_set = {states[sid]["instruction"] for sid in state_ids}
        emb_cache = {}
        for instr in instr_set:
            ents = extract_entities(instr)
            emb_cache[instr] = [embed_phrase(model, tokenizer, e) for e in ents]

        t0 = time.time()
        for ordinal, sid in enumerate(state_ids):
            out_npz = fus_dir / f"{sid}.npz"
            if out_npz.exists():
                continue
            sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
            row = states[sid]
            instruction = row["instruction"]
            labels = sem["labels_kmeans_8"].astype(np.int64)
            clean_action = sem["clean_action"].astype(np.float64)
            clean_ids = torch.tensor(sem["clean_ids"], dtype=torch.long, device=model.device).unsqueeze(0)
            r_pixel = sem["r_pixel"].astype(np.float64)
            masked = sem["masked_kmeans_8"]           # [8,7,256]
            object_ids = set(int(x) for x in sem["object_ids"].tolist())
            g_obj = int(sem["g_obj_idx_kmeans_8"][0])

            h = np.load(H_REUSE / f"{sid}.npz")["h"].astype(np.float64)
            v = group_means(h, labels)
            L = language_scores(v, emb_cache[instruction])
            A = np.load(ACTION_NPZ / f"{sid}.npz")["l2_mean"].astype(np.float64)

            Lh = minmax01(L)
            Ah = minmax01(A)
            groups = [set(int(x) for x in np.flatnonzero(labels == k).tolist()) for k in range(K)]

            out = {"state_id": sid, "task": str(sem["task"]),
                   "L": L.astype(np.float32), "A": A.astype(np.float32),
                   "lang_top1": np.asarray([int(np.argmax(L))], dtype=np.int64),
                   "action_top1": np.asarray([int(np.argmax(A))], dtype=np.int64)}

            for alpha in ALPHAS:
                S = alpha * Lh + (1 - alpha) * Ah
                g = int(np.argmax(S))
                r = log_softmax_residual(clean_action, masked[g].astype(np.float64))
                out[f"fusion_sel_{alpha_key(alpha)}"] = np.asarray([g], dtype=np.int64)
                out[f"fusion_r_{alpha_key(alpha)}"] = r.astype(np.float32)
                out[f"fusion_iou_{alpha_key(alpha)}"] = np.asarray([iou_group(labels, g, object_ids)], dtype=np.float64)
                out[f"fusion_hit_{alpha_key(alpha)}"] = np.asarray([1 if g == g_obj else 0], dtype=np.int64)

            # ---- Ablation B: intersection (Lang top-K ∩ Action top-K) ----
            lang_topk = set(int(x) for x in np.argsort(-L)[:TOPK].tolist())
            action_topk = set(int(x) for x in np.argsort(-A)[:TOPK].tolist())
            inter = sorted(lang_topk & action_topk)
            if not inter:
                out["inter_empty"] = np.asarray([1], dtype=np.int64)
                out["inter_sel"] = np.asarray([], dtype=np.int64)
                out["inter_r"] = np.zeros((7, 256), dtype=np.float32)
                out["inter_iou"] = np.asarray([0.0], dtype=np.float64)
            else:
                out["inter_empty"] = np.asarray([0], dtype=np.int64)
                out["inter_sel"] = np.asarray(inter, dtype=np.int64)
                if len(inter) == 1:
                    branch = masked[inter[0]].astype(np.float64)
                else:
                    union_p = sorted(set().union(*[groups[g] for g in inter]))
                    ma, _ = forward_masked(model, processor, cv2.cvtColor(
                        cv2.imread(str(pcd_root / row["clean_path"])), cv2.COLOR_BGR2RGB),
                        instruction, clean_ids, union_p, mean)
                    branch = ma[:, token_slice].numpy().astype(np.float64)
                    n_fwd += 1
                r = log_softmax_residual(clean_action, branch)
                u = set().union(*[groups[g] for g in inter])
                inter = len(u & object_ids) / (len(u | object_ids) if (u | object_ids) else 1)
                out["inter_r"] = r.astype(np.float32)
                out["inter_iou"] = np.asarray([inter], dtype=np.float64)

            # ---- Ablation C: union (Lang top-K ∪ Action top-K) ----
            union_groups = sorted(lang_topk | action_topk)
            out["union_sel"] = np.asarray(union_groups, dtype=np.int64)
            union_p = sorted(set().union(*[groups[g] for g in union_groups]))
            ma, _ = forward_masked(model, processor, cv2.cvtColor(
                cv2.imread(str(pcd_root / row["clean_path"])), cv2.COLOR_BGR2RGB),
                instruction, clean_ids, union_p, mean)
            branch = ma[:, token_slice].numpy().astype(np.float64)
            n_fwd += 1
            r = log_softmax_residual(clean_action, branch)
            u = set().union(*[groups[g] for g in union_groups])
            un = len(u & object_ids) / (len(u | object_ids) if (u | object_ids) else 1)
            out["union_r"] = r.astype(np.float32)
            out["union_iou"] = np.asarray([un], dtype=np.float64)

            np.savez_compressed(out_npz, **out)
            dt = time.time() - t0
            if (ordinal + 1) % 20 == 0 or ordinal + 1 == len(state_ids):
                print(json.dumps({"done": ordinal + 1, "of": len(state_ids),
                                  "sec": round(dt, 1), "fwd": n_fwd}), flush=True)

        print(json.dumps({"DONE_COMPUTE": len(state_ids), "total_fwd": n_fwd,
                          "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    # ---------- aggregate ----------
    def align_pooled(r_field):
        vv = []
        for sid in state_ids:
            f = fus_dir / f"{sid}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
            r = d[r_field].astype(np.float64)
            vv.extend(per_position_cosine(r, sem["r_pixel"].astype(np.float64)).tolist())
        return _mean(vv)

    # fusion alpha sweep
    fusion_align = {alpha: align_pooled(f"fusion_r_{alpha_key(alpha)}") for alpha in ALPHAS}

    # baselines from action_npz / entity_npz
    def align_external(npz_dir, field):
        vv = []
        for sid in state_ids:
            f = npz_dir / f"{sid}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
            r = d[field].astype(np.float64)
            vv.extend(per_position_cosine(r, sem["r_pixel"].astype(np.float64)).tolist())
        return _mean(vv)

    action_align = align_external(ACTION_NPZ, "action_l2_r")
    random_align = align_external(ACTION_NPZ, "random_r")
    attention_align = align_external(ACTION_NPZ, "attention_r")
    oracle_single_align = align_external(ACTION_NPZ, "oracle_single_r")
    lang_es_align = align_external(ENTITY_NPZ, "entity_set_r_8")
    lang_ss_align = align_external(ENTITY_NPZ, "single_source_r_8")

    inter_align = align_pooled("inter_r")
    union_align = align_pooled("union_r")

    # per-state align for each alpha (for global best + task breakdown + statistical test)
    state_align = {}
    for sid in state_ids:
        f = fus_dir / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
        rp = sem["r_pixel"].astype(np.float64)
        for alpha in ALPHAS:
            r = d[f"fusion_r_{alpha_key(alpha)}"].astype(np.float64)
            state_align.setdefault(alpha, {})[sid] = float(np.mean(per_position_cosine(r, rp)))

    best_alpha = max(ALPHAS, key=lambda a_: fusion_align[a_])
    best_fusion = fusion_align[best_alpha]

    # ---- oracle 2x2 analysis ----
    four = {"both": 0, "lang_only": 0, "action_only": 0, "neither": 0}
    agree = 0
    n = 0
    for sid in state_ids:
        f = fus_dir / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
        g_obj = int(sem["g_obj_idx_kmeans_8"][0])
        lc = int(d["lang_top1"][0]) == g_obj
        ac = int(d["action_top1"][0]) == g_obj
        if lc == ac:
            agree += 1
        n += 1
        if lc and ac:
            four["both"] += 1
        elif lc and not ac:
            four["lang_only"] += 1
        elif not lc and ac:
            four["action_only"] += 1
        else:
            four["neither"] += 1
    oracle2x2 = {k: {"count": v, "frac": v / n if n else 0.0} for k, v in four.items()}
    oracle2x2["agree_rate"] = agree / n if n else 0.0
    oracle2x2["n"] = n

    # ---- task breakdown (global best α, action, language) ----
    def task_align_from_state(alpha, t):
        vv = []
        for sid in state_ids:
            f = fus_dir / f"{sid}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            if str(d["task"]) != t or sid not in state_align.get(alpha, {}):
                continue
            vv.append(state_align[alpha][sid])
        return _mean(vv)

    def task_align_external(npz_dir, field, t):
        vv = []
        for sid in state_ids:
            f = npz_dir / f"{sid}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            if str(d["task"]) != t:
                continue
            sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
            r = d[field].astype(np.float64)
            vv.append(float(np.mean(per_position_cosine(r, sem["r_pixel"].astype(np.float64)))))
        return _mean(vv)

    task_breakdown = []
    tasks = sorted({str(np.load(SEMANTIC_NPZ / f"{sid}.npz")["task"]) for sid in state_ids
                    if (SEMANTIC_NPZ / f"{sid}.npz").exists()})
    for t in tasks:
        fa = task_align_from_state(best_alpha, t)
        aa = task_align_external(ACTION_NPZ, "action_l2_r", t)
        la = task_align_external(ENTITY_NPZ, "entity_set_r_8", t)
        oa = task_align_external(ACTION_NPZ, "oracle_single_r", t)
        task_breakdown.append({"task": t, "fusion_align": fa, "action_align": aa,
                               "language_align": la, "oracle_align": oa})

    # ---- gates ----
    # Gate B statistical test (best fusion vs random, paired)
    d_ = np.asarray([state_align[best_alpha][s] for s in state_align[best_alpha]])
    rr = []
    for sid in state_ids:
        f = ACTION_NPZ / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
        rr.append(float(np.mean(per_position_cosine(
            d["random_r"].astype(np.float64), sem["r_pixel"].astype(np.float64)))))
    rr = np.asarray(rr)
    dd = d_ - rr
    nn = len(dd)
    se = dd.std(ddof=1) / np.sqrt(nn) if nn > 1 else float("inf")
    z = dd.mean() / se if se > 0 else 0.0
    from math import erfc, sqrt
    pval = float(erfc(abs(z) / sqrt(2.0)))

    gateA = best_fusion > 0.62
    gateB = (best_fusion > random_align) and (pval < 0.05)
    gateC = False
    htc = {}
    for t in HARD_TASKS:
        tb = next((x for x in task_breakdown if x["task"] == t), None)
        if tb is None:
            continue
        best_single = max(tb["action_align"], tb["language_align"])
        htc[t] = {"fusion": tb["fusion_align"], "action": tb["action_align"],
                  "language": tb["language_align"], "best_single": best_single,
                  "improved": tb["fusion_align"] > best_single + 0.02}
    gateC = any(htc[t]["improved"] for t in htc)

    stop_low = best_fusion <= 0.60
    verdict = "PASS_TO_ROLLOUT" if (gateA and gateB and gateC and not stop_low) \
        else "STOP_MULTI_SIGNAL_NO_GO"

    results = {
        "experiment": "MULTI_SIGNAL_SEMANTIC_CD_PHASE0_V1",
        "split": a.split, "n_states": len([s for s in state_ids if (fus_dir / f"{s}.npz").exists()]),
        "verdict": verdict, "stop_rule_hit": stop_low,
        "best_alpha": best_alpha, "best_fusion_align": best_fusion,
        "fusion_alpha_sweep": {alpha_key(al): fusion_align[al] for al in ALPHAS},
        "alignment": {
            "fusion_best": best_fusion,
            "action_only": action_align, "language_entity_set": lang_es_align,
            "language_single_source": lang_ss_align,
            "oracle_single": oracle_single_align, "random": random_align,
            "attention": attention_align, "cat_patch": 0.343, "oracle_set": 0.690,
            "intersection": inter_align, "union": union_align,
        },
        "gates": {
            "A_alignment": {"PASS": bool(gateA), "best_fusion": best_fusion, "threshold": 0.62},
            "B_over_random": {"PASS": bool(gateB), "best_fusion": best_fusion,
                              "random": random_align, "p_value": pval},
            "C_hard_task": {"PASS": bool(gateC), "hard_tasks": htc},
        },
        "ablations": {
            "A_simple_vs_weighted": {"alpha_050": fusion_align[0.5], "best_alpha": best_alpha,
                                     "best_align": best_fusion,
                                     "weighted_gt_equal": best_fusion > fusion_align[0.5]},
            "B_intersection": {"align": inter_align,
                               "empty_rate": _mean([int(np.load(fus_dir / f"{s}.npz")["inter_empty"][0])
                                                    for s in state_ids if (fus_dir / f"{s}.npz").exists()])},
            "C_union": {"align": union_align},
        },
        "oracle_2x2": oracle2x2,
        "stop_flag": {"fusion_leq_060": bool(stop_low)},
        "task_breakdown": task_breakdown,
    }
    (art / "selector_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "MULTI_SIGNAL_SEMANTIC_CD_PHASE0_V1",
        "status": verdict, "split": a.split, "date": "2026-08-24",
        "n_states": results["n_states"],
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "core_numbers": {
            "fusion_best_alpha": best_alpha, "align_fusion_best": best_fusion,
            "align_action": action_align, "align_language_entity_set": lang_es_align,
            "align_oracle_single": oracle_single_align, "align_random": random_align,
            "align_intersection": inter_align, "align_union": union_align,
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    # CSVs
    with (art / "alpha_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha", "align_pooled"])
        for alpha in ALPHAS:
            w.writerow([alpha, fusion_align[alpha]])
    with (art / "alignment_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "align"])
        w.writeheader()
        for name, val in results["alignment"].items():
            w.writerow({"method": name, "align": val})
    with (art / "oracle_recovery.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "count", "frac"])
        w.writeheader()
        for k, v in oracle2x2.items():
            if k in ("n", "agree_rate"):
                continue
            w.writerow({"case": k, "count": v["count"], "frac": v["frac"]})
        w.writerow({"case": "agree_rate", "count": "", "frac": oracle2x2["agree_rate"]})
    with (art / "signal_complementarity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "count", "frac"])
        w.writeheader()
        for k, v in oracle2x2.items():
            if k in ("n", "agree_rate"):
                continue
            w.writerow({"case": k, "count": v["count"], "frac": v["frac"]})
    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "fusion_align", "action_align", "language_align", "oracle_align"])
        w.writeheader()
        for tb in task_breakdown:
            w.writerow(tb)

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
