#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-0 evaluation: gates A/B/C + STOP RULE verdict.

Reads per-state semantic_npz from the compute loop, recomputes all residuals and
cosines offline (log_softmax space, matching r_pixel), aggregates over the
Confirmation split, writes decision JSON + CSVs + REPORT_ZH.md + group viz.

Run:
  cd /home/leju-suzhou/zjt_ws/token-cd
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/evaluate.py \
    --artifact artifacts/semantic_token_cd_phase0_v1 --split confirmation
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from research.cw_lpcd.metrics import per_position_cosine, state_cosine
from research.semantic_token_cd.grouping import group_compactness

# (method, K) combos, primary = kmeans K-ablation.
METHOD_KS = [("kmeans", 8), ("kmeans", 16), ("kmeans", 32), ("agglomerative", 16), ("spectral", 16)]
PRIMARY_KS = [8, 16, 32]  # kmeans K-ablation for gates B/C


def log_softmax_residual(clean: np.ndarray, branch: np.ndarray) -> np.ndarray:
    """r = log_softmax(clean[action]) - log_softmax(branch[action]); [7,256] float64."""
    ca = torch.tensor(clean, dtype=torch.float64)
    ba = torch.tensor(branch, dtype=torch.float64)
    return (torch.log_softmax(ca, -1) - torch.log_softmax(ba, -1)).numpy()


def _med(pool):
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    return float(np.median(pool)) if pool.size else float("nan")


def load_states(npz_dir: Path, state_ids: list[str]) -> dict:
    return {sid: np.load(npz_dir / f"{sid}.npz") for sid in state_ids
            if (npz_dir / f"{sid}.npz").exists()}


def colormap(k: int, K: int) -> tuple[int, int, int]:
    import colorsys
    if K <= 1:
        return (200, 200, 200)
    r, g, b = colorsys.hsv_to_rgb(k / K, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def render_state(sid: str, d, out_png: Path) -> None:
    """Render each combo's grouping (left) vs object mask (right) as one PNG."""
    obj = set(int(x) for x in d["object_ids"].tolist())
    panels = []
    for method, K in METHOD_KS:
        lab = d[f"labels_{method}_{K}"]
        # grouping panel
        gimg = Image.new("RGB", (16, 16))
        gpix = gimg.load()
        for p in range(256):
            gpix[p % 16, p // 16] = colormap(int(lab[p]), K)
        gimg = gimg.resize((224, 224), Image.NEAREST)
        # object-mask panel
        mimg = Image.new("RGB", (16, 16), (20, 20, 20))
        mpix = mimg.load()
        for p in range(256):
            if p in obj:
                mpix[p % 16, p // 16] = (255, 255, 255)
        mimg = mimg.resize((224, 224), Image.NEAREST)
        row = Image.new("RGB", (224 * 2 + 8, 224), (10, 10, 10))
        row.paste(gimg, (0, 0))
        row.paste(mimg, (232, 0))
        panels.append(row)
    canvas = Image.new("RGB", (224 * 2 + 8, 224 * len(panels)), (10, 10, 10))
    for i, row in enumerate(panels):
        canvas.paste(row, (0, i * 224))
    canvas.save(out_png)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    p.add_argument("--viz-states", type=int, default=6)
    a = p.parse_args()
    art = a.artifact.resolve()
    npz_dir = art / "semantic_npz"

    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
    state_ids = split[key]
    states = load_states(npz_dir, state_ids)
    n = len(states)
    missing = [s for s in state_ids if s not in states]
    print(json.dumps({"split": a.split, "n_loaded": n, "n_missing": len(missing), "missing_head": missing[:5]}))

    # ---------------- per (state) x (combo) records ----------------
    rows = []  # semantic (g_obj) + baselines, one row per combo per state
    pooled = {}  # combo -> flat list of per-position cosines vs r_pixel (over states x 7)
    pooled_attn = {}
    pooled_rnd = {}
    pooled_full = []
    group_assignment = []  # per state per combo, g_obj details
    labels_agg = {f"{m}_{K}": [] for m, K in METHOD_KS}

    for sid in state_ids:
        if sid not in states:
            continue
        d = states[sid]
        clean = d["clean_action"]
        r_pixel = d["r_pixel"].astype(np.float64)
        obj = set(int(x) for x in d["object_ids"].tolist())
        n_obj = len(obj)
        chance = n_obj / 256.0

        full_r = log_softmax_residual(clean, d["fullobj_masked"])
        pooled_full.extend(per_position_cosine(full_r, r_pixel).tolist())

        for method, K in METHOD_KS:
            keyn = f"{method}_{K}"
            lab = d[f"labels_{keyn}"]
            masked = d[f"masked_{keyn}"]            # [K,7,256]
            overlap = d[f"overlap_{keyn}"]          # [K,5]
            gi = int(d[f"g_obj_idx_{keyn}"][0])
            grp = np.flatnonzero(lab == gi).astype(int).tolist()
            s = len(grp)
            prec, rec, iou, cprec, ciou = overlap[gi]

            r_g = log_softmax_residual(clean, masked[gi])
            r_attn = log_softmax_residual(clean, d[f"attn_masked_{keyn}"])
            r_rnd = log_softmax_residual(clean, d[f"rnd_masked_{keyn}"])

            Dg = float(np.linalg.norm(r_g))
            Da = float(np.linalg.norm(r_attn))
            Dr = float(np.linalg.norm(r_rnd))
            pc_g = per_position_cosine(r_g, r_pixel)
            pc_a = per_position_cosine(r_attn, r_pixel)
            pc_r = per_position_cosine(r_rnd, r_pixel)

            rows.append({
                "state_id": sid, "task": str(d["task"]), "method": method, "K": K,
                "g_obj_size": s, "precision": prec, "recall": rec, "iou": iou,
                "chance_prec": cprec, "chance_iou": ciou, "compactness": group_compactness(grp),
                "D_semantic": Dg, "D_attn": Da, "D_random": Dr,
                "cos_pos_semantic": float(np.median(pc_g)), "cos_state_semantic": state_cosine(r_g, r_pixel),
                "cos_pos_attn": float(np.median(pc_a)), "cos_pos_random": float(np.median(pc_r)),
            })
            pooled.setdefault(keyn, []).extend(pc_g.tolist())
            pooled_attn.setdefault(keyn, []).extend(pc_a.tolist())
            pooled_rnd.setdefault(keyn, []).extend(pc_r.tolist())
            labels_agg[keyn].append(lab.astype(np.int64))

            group_assignment.append({
                "state_id": sid, "task": str(d["task"]), "method": method, "K": K,
                "g_obj_idx": gi, "g_obj_size": s,
                "precision": float(prec), "recall": float(rec), "iou": float(iou),
                "chance_prec": float(cprec), "chance_iou": float(ciou),
                "compactness": float(group_compactness(grp)),
            })

    # ---------------- aggregates per combo ----------------
    agg = {}
    for method, K in METHOD_KS:
        keyn = f"{method}_{K}"
        rr = [r for r in rows if r["method"] == method and r["K"] == K]
        agg[keyn] = {
            "D_semantic_median": _med([r["D_semantic"] for r in rr]),
            "D_attn_median": _med([r["D_attn"] for r in rr]),
            "D_random_median": _med([r["D_random"] for r in rr]),
            "cos_pos_semantic_pooled": _med(pooled[keyn]),
            "cos_pos_attn_pooled": _med(pooled_attn[keyn]),
            "cos_pos_random_pooled": _med(pooled_rnd[keyn]),
            "cos_state_semantic_median": _med([r["cos_state_semantic"] for r in rr]),
            "precision_median": _med([r["precision"] for r in rr]),
            "recall_median": _med([r["recall"] for r in rr]),
            "iou_median": _med([r["iou"] for r in rr]),
            "chance_prec_median": _med([r["chance_prec"] for r in rr]),
            "chance_iou_median": _med([r["chance_iou"] for r in rr]),
            "compactness_median": _med([r["compactness"] for r in rr]),
            "g_obj_size_median": _med([r["g_obj_size"] for r in rr]),
        }

    cos_full_pooled = _med(pooled_full)

    # ---------------- semantic specificity (question 2, kmeans only) ----------------
    # For each state & K: does the object group (argmax IoU) also rank top in PCD alignment?
    specificity = {}
    for K in PRIMARY_KS:
        ranks, obj_cos, best_cos, med_cos = [], [], [], []
        for sid in state_ids:
            if sid not in states:
                continue
            d = states[sid]
            clean = d["clean_action"]
            rp = d["r_pixel"].astype(np.float64)
            masked = d[f"masked_kmeans_{K}"]
            gi = int(d[f"g_obj_idx_kmeans_{K}"][0])
            cosg = np.asarray([float(np.median(per_position_cosine(
                log_softmax_residual(clean, masked[k]), rp))) for k in range(K)])
            order = np.argsort(-cosg)
            ranks.append(int(np.where(order == gi)[0][0]) + 1)
            obj_cos.append(cosg[gi]); best_cos.append(float(cosg.max())); med_cos.append(float(np.median(cosg)))
        specificity[f"kmeans_{K}"] = {
            "object_group_alignment_rank_median": _med(ranks),
            "frac_object_group_top1": float(np.mean(np.asarray(ranks) == 1)),
            "frac_object_group_top2": float(np.mean(np.asarray(ranks) <= 2)),
            "object_group_cos_median": _med(obj_cos),
            "best_group_cos_median": _med(best_cos),
            "median_group_cos_median": _med(med_cos),
        }

    # ---------------- gates ----------------
    # Gate A: kmeans discovers object at ALL K (median precision > chance AND median IoU > chance_iou)
    gateA_perK = {}
    for K in PRIMARY_KS:
        keyn = f"kmeans_{K}"
        gateA_perK[K] = bool(agg[keyn]["precision_median"] > agg[keyn]["chance_prec_median"]
                             and agg[keyn]["iou_median"] > agg[keyn]["chance_iou_median"])
    gateA = all(gateA_perK.values())

    # Gate B: pooled cos(r_g_obj, r_PCD) > pooled cos(r_attn, r_PCD) for >=2 of 3 K
    gateB_perK = {}
    for K in PRIMARY_KS:
        keyn = f"kmeans_{K}"
        gateB_perK[K] = bool(agg[keyn]["cos_pos_semantic_pooled"] > agg[keyn]["cos_pos_attn_pooled"])
    gateB = sum(gateB_perK.values()) >= 2

    # Gate C: median D_semantic > median D_random for >=2 of 3 K
    gateC_perK = {}
    for K in PRIMARY_KS:
        keyn = f"kmeans_{K}"
        gateC_perK[K] = bool(agg[keyn]["D_semantic_median"] > agg[keyn]["D_random_median"])
    gateC = sum(gateC_perK.values()) >= 2

    # stop-rule flags
    stop_semantic_random = not gateA            # no semantic structure
    stop_semantic_attn = gateA and not gateB    # structure but no improvement over attention proxy
    stop_not_close_pcd = False                   # judged below vs full-object ceiling

    # "alignment not close to PCD" — flag if best-K semantic cos is far below full-object ceiling
    best_cos = max(agg[f"kmeans_{K}"]["cos_pos_semantic_pooled"] for K in PRIMARY_KS)
    if best_cos < 0.5 * cos_full_pooled:
        stop_not_close_pcd = True

    stop_hit = stop_semantic_random or stop_semantic_attn or stop_not_close_pcd
    verdict = "PASS_TO_ROLLOUT" if (gateA and gateB and gateC and not stop_hit) else "STOP_SEMANTIC_TOKEN_CD_NO_GO"

    # ---------------- write CSVs ----------------
    with (art / "pcd_alignment.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "method", "K", "g_obj_size",
                                           "cos_pos_semantic", "cos_state_semantic", "precision", "recall", "iou"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    with (art / "baseline_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "method", "K",
                                           "D_semantic", "D_attn", "D_random",
                                           "cos_pos_semantic", "cos_pos_attn", "cos_pos_random"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # ---------------- semantic_groups.npz + group_assignment.json ----------------
    np.savez_compressed(
        art / "semantic_groups.npz",
        state_ids=np.asarray(state_ids),
        combos=np.asarray([f"{m}_{K}" for m, K in METHOD_KS]),
        **{f"labels_{m}_{K}": np.stack(labels_agg[f"{m}_{K}"]) for m, K in METHOD_KS},
    )
    (art / "group_assignment.json").write_text(json.dumps(group_assignment, indent=2) + "\n")

    # ---------------- visualization ----------------
    viz_dir = art / "group_visualization"
    viz_dir.mkdir(exist_ok=True)
    for sid in state_ids[: a.viz_states]:
        if sid in states:
            render_state(sid, states[sid], viz_dir / f"{sid}.png")

    # ---------------- decision JSON ----------------
    results = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE0_V1",
        "split": a.split, "n_states": n, "n_missing": len(missing),
        "verdict": verdict, "stop_rule_hit": stop_hit,
        "gates": {
            "A_structure": {"PASS": bool(gateA), "per_K": gateA_perK},
            "B_alignment": {"PASS": bool(gateB), "per_K": gateB_perK},
            "C_disruption": {"PASS": bool(gateC), "per_K": gateC_perK},
        },
        "stop_flags": {
            "semantic_approx_random": bool(stop_semantic_random),
            "semantic_lt_attention": bool(stop_semantic_attn),
            "alignment_not_close_pcd": bool(stop_not_close_pcd),
        },
        "aggregate_median": agg,
        "semantic_specificity": specificity,
        "full_object_ceiling": {"cos_pos_pooled": cos_full_pooled,
                                "note": "persistent 全对象 mask 的对齐上限（本 split 重算）"},
        "best_K_cos": best_cos,
        "historical_reference": {
            "persistent_object_pixel_cosine_median": 0.4765,
            "persistent_random_pixel_cosine_median": 0.0721,
            "note": "旧 Persistent Token-PCD 全对象 mask 对齐 0.4765 / random 0.0721（selection split）",
        },
    }
    (art / "evaluation_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE0_V1",
        "status": verdict,
        "split": a.split, "n_states": n,
        "date": "2026-08-23",
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "core_numbers": {
            "gateA_perK": {str(K): bool(v) for K, v in gateA_perK.items()},
            "gateB_perK": {str(K): bool(v) for K, v in gateB_perK.items()},
            "gateC_perK": {str(K): bool(v) for K, v in gateC_perK.items()},
            "cos_semantic_k8": agg["kmeans_8"]["cos_pos_semantic_pooled"],
            "cos_semantic_k16": agg["kmeans_16"]["cos_pos_semantic_pooled"],
            "cos_semantic_k32": agg["kmeans_32"]["cos_pos_semantic_pooled"],
            "cos_attn_k8": agg["kmeans_8"]["cos_pos_attn_pooled"],
            "cos_full_object_ceiling": cos_full_pooled,
            "D_semantic_k8": agg["kmeans_8"]["D_semantic_median"],
            "D_random_k8": agg["kmeans_8"]["D_random_median"],
            "iou_k8": agg["kmeans_8"]["iou_median"],
            "iou_k16": agg["kmeans_16"]["iou_median"],
            "iou_k32": agg["kmeans_32"]["iou_median"],
            "iou_chance_k8": agg["kmeans_8"]["chance_iou_median"],
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "phase0_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
