#!/usr/bin/env python3
"""ACTION_CONDITIONED_SEMANTIC_CD Phase-0 (compute, pure numpy/CPU, no model).

For each Confirmation state, reuse precomputed masked branches to score every
K=8 semantic group by its action-logit disruption, then select the argmax group
as the action-conditioned negative branch. Compare alignment cos(r_sel, r_PCD)
against language / attention / random / oracle baselines.

Disruption metrics (per action position j=0..6, over 256 action bins):
  D_l2[i,j]  = ||z_full[j] - z_{-G_i}[j]||_2
  D_kl[i,j]  = KL(softmax(z_full[j]) || softmax(z_{-G_i}[j]))
  D_exp[i,j] = softmax(z_full[j])[a*_j] - softmax(z_{-G_i}[j])[a*_j]

Selectors: argmax_i mean_j D (l2 / kl / exp), combined α·l2+(1-α)·exp, and l2@tok0.

No rollout, no training, no λ, no external VLM. object mask = evaluation only.

Run (smoke):
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/action_causal.py \
    --artifact artifacts/action_conditioned_semantic_cd_phase0_v1 \
    --split confirmation --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from research.cw_lpcd.metrics import per_position_cosine

SEMANTIC_NPZ = Path("artifacts/semantic_token_cd_phase0_v1/semantic_npz")
SPLIT = Path("artifacts/cat_cd_phase0_v1/FROZEN_SPLIT.json")
ENTITY_NPZ = Path("artifacts/semantic_token_cd_phase2b_entity_set_v1/entity_npz")
ACTION_VOCAB_OFFSET = 31744

K = 8
COMB_ALPHAS = (0.25, 0.5, 0.75)
HARD_TASKS = ["widowx_put_eggplant_in_basket", "widowx_spoon_on_towel"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _t(x, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(np.asarray(x), dtype=dtype)


def log_softmax_residual(clean: np.ndarray, branch: np.ndarray) -> np.ndarray:
    return (torch.log_softmax(_t(clean), -1) - torch.log_softmax(_t(branch), -1)).numpy()


def _mean(pool):
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    return float(np.mean(pool)) if pool.size else float("nan")


def _med(pool):
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    return float(np.median(pool)) if pool.size else float("nan")


def disruption_matrix(clean: np.ndarray, masked: np.ndarray, a_bins: np.ndarray,
                      valid: np.ndarray) -> dict[str, np.ndarray]:
    """clean [7,256], masked [8,7,256], a_bins [7], valid [7] -> {l2,kl,exp} each [8,7].

    exp[i,j] = NaN where clean_ids[j] is not an action-vocab token (e.g. EOS=2),
    since the demo action a* is undefined there."""
    c = _t(clean)                       # [7,256]
    lp = torch.log_softmax(c, -1)
    p = torch.exp(lp)
    l2 = np.zeros((K, 7), dtype=np.float64)
    kl = np.zeros((K, 7), dtype=np.float64)
    ex = np.full((K, 7), np.nan, dtype=np.float64)
    rows = torch.arange(7)
    a_clamped = np.clip(a_bins, 0, 255)
    for i in range(K):
        b = _t(masked[i])               # [7,256]
        l2[i] = torch.norm(c - b, dim=1).numpy()
        lq = torch.log_softmax(b, -1)
        q = torch.exp(lq)
        kl[i] = (p * (lp - lq)).sum(-1).numpy()
        drop = (p[rows, a_clamped] - q[rows, a_clamped]).numpy()
        ex[i, valid] = drop[valid]
    return {"l2": l2, "kl": kl, "exp": ex}


def iou_patchset(ids: np.ndarray, obj: set) -> float:
    s = set(int(x) for x in ids.tolist())
    inter = len(s & obj)
    union = len(s | obj)
    return inter / union if union else 0.0


def iou_group(labels: np.ndarray, g: int, obj: set) -> float:
    s = set(int(x) for x in np.flatnonzero(labels == g).tolist())
    inter = len(s & obj)
    union = len(s | obj)
    return inter / union if union else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    p.add_argument("--limit", type=int)
    p.add_argument("--state-id", type=str)
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()

    art = a.artifact.resolve()
    out_dir = art / "action_npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT.read_text())
    (art / "FROZEN_SPLIT.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    if a.state_id:
        state_ids = [a.state_id]
    else:
        key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
        state_ids = split[key][: a.limit] if a.limit else split[key]

    if not a.eval_only:
        n = 0
        for sid in state_ids:
            out_npz = out_dir / f"{sid}.npz"
            if out_npz.exists():
                continue
            sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
            clean = sem["clean_action"].astype(np.float64)          # [7,256]
            masked = sem["masked_kmeans_8"]                          # [8,7,256]
            clean_ids = sem["clean_ids"].astype(np.int64)            # [7]
            r_pixel = sem["r_pixel"].astype(np.float64)              # [7,256]
            object_ids = set(int(x) for x in sem["object_ids"].tolist())
            labels = sem["labels_kmeans_8"].astype(np.int64)
            g_obj = int(sem["g_obj_idx_kmeans_8"][0])
            attn_ids = sem["attn_ids_kmeans_8"]
            rnd_ids = sem["rnd_ids_kmeans_8"]

            a_bins = clean_ids - ACTION_VOCAB_OFFSET
            valid = (clean_ids >= ACTION_VOCAB_OFFSET) & (clean_ids < ACTION_VOCAB_OFFSET + 256)
            dm = disruption_matrix(clean, masked, a_bins, valid)
            l2_mean = dm["l2"].mean(axis=1)     # [8]
            kl_mean = dm["kl"].mean(axis=1)
            ex_mean = np.nanmean(dm["exp"], axis=1)     # [8], NaN only if all 7 invalid
            ex_mean = np.nan_to_num(ex_mean, nan=0.0)
            l2_tok0 = dm["l2"][:, 0]

            def norm01(x):
                lo, hi = x.min(), x.max()
                return (x - lo) / (hi - lo + 1e-12)

            l2n, exn = norm01(l2_mean), norm01(ex_mean)
            sel = {
                "action_l2": int(np.argmax(l2_mean)),
                "action_kl": int(np.argmax(kl_mean)),
                "action_exp": int(np.argmax(ex_mean)),
                "action_l2_tok0": int(np.argmax(l2_tok0)),
            }
            for alpha in COMB_ALPHAS:
                s = alpha * l2n + (1 - alpha) * exn
                sel[f"action_comb_{int(alpha * 100):03d}"] = int(np.argmax(s))

            out = {"state_id": sid, "task": str(sem["task"]),
                   "g_obj": np.asarray([g_obj], dtype=np.int64),
                   "l2_mean": l2_mean.astype(np.float32), "kl_mean": kl_mean.astype(np.float32),
                   "exp_mean": ex_mean.astype(np.float32),
                   "l2_tok0": l2_tok0.astype(np.float32)}

            for name, g in sel.items():
                branch = masked[g].astype(np.float64)
                r = log_softmax_residual(clean, branch)
                out[f"{name}_sel"] = np.asarray([g], dtype=np.int64)
                out[f"{name}_r"] = r.astype(np.float32)
                out[f"{name}_iou"] = np.asarray([iou_group(labels, g, object_ids)], dtype=np.float64)
                out[f"{name}_hit"] = np.asarray([1 if g == g_obj else 0], dtype=np.int64)

            # baselines (precomputed masked branches)
            for bname, bid, bmask in (
                ("random", rnd_ids, sem["rnd_masked_kmeans_8"]),
                ("attention", attn_ids, sem["attn_masked_kmeans_8"]),
            ):
                r = log_softmax_residual(clean, bmask.astype(np.float64))
                out[f"{bname}_r"] = r.astype(np.float32)
                out[f"{bname}_iou"] = np.asarray([iou_patchset(bid, object_ids)], dtype=np.float64)
            r = log_softmax_residual(clean, masked[g_obj].astype(np.float64))
            out["oracle_single_r"] = r.astype(np.float32)
            out["oracle_single_iou"] = np.asarray([iou_group(labels, g_obj, object_ids)], dtype=np.float64)

            np.savez_compressed(out_npz, **out)
            n += 1
        print(json.dumps({"DONE_COMPUTE": n, "of": len(state_ids)}), flush=True)

    # ---------- aggregate ----------
    def align(r, rp):
        return float(np.mean(per_position_cosine(r, rp)))

    action_selectors = ["action_l2", "action_kl", "action_exp", "action_l2_tok0"] + \
        [f"action_comb_{int(alpha * 100):03d}" for alpha in COMB_ALPHAS]
    baselines = ["random", "attention", "oracle_single"]

    # language baselines from Phase-2B entity_npz
    lang_fields = ["single_source", "single_goal", "entity_set", "oracle_set", "random_set"]

    rows = []
    pooled = {}   # (name) -> list of per-position cosine values
    for name in action_selectors + baselines:
        pooled[name] = []
    for name in lang_fields:
        pooled[f"lang_{name}"] = []

    for sid in state_ids:
        f = out_dir / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
        r_pixel = sem["r_pixel"].astype(np.float64)
        task = str(d["task"])
        for name in action_selectors:
            r = d[f"{name}_r"].astype(np.float64)
            pc = per_position_cosine(r, r_pixel)
            pooled[name].extend(pc.tolist())
            rows.append({"state_id": sid, "task": task, "selector": name,
                         "iou": float(d[f"{name}_iou"][0]), "hit": int(d[f"{name}_hit"][0]),
                         "align": float(np.mean(pc))})
        for name in baselines:
            r = d[f"{name}_r"].astype(np.float64)
            pc = per_position_cosine(r, r_pixel)
            pooled[name].extend(pc.tolist())
            rows.append({"state_id": sid, "task": task, "selector": name,
                         "iou": float(d[f"{name}_iou"][0]), "hit": 0,
                         "align": float(np.mean(pc))})
        # language baselines
        ef = ENTITY_NPZ / f"{sid}.npz"
        if ef.exists():
            ed = np.load(ef)
            for name in lang_fields:
                r = ed[f"{name}_r_8"].astype(np.float64)
                pc = per_position_cosine(r, r_pixel)
                pooled[f"lang_{name}"].extend(pc.tolist())
                rows.append({"state_id": sid, "task": task, "selector": f"lang_{name}",
                             "iou": float(ed[f"{name}_iou_8"][0]), "hit": 0,
                             "align": float(np.mean(pc))})

    align_pooled = {name: _mean(v) for name, v in pooled.items()}

    # ---- primary = action_l2 ----
    a2 = align_pooled["action_l2"]
    akl = align_pooled["action_kl"]
    aex = align_pooled["action_exp"]
    lang_es = align_pooled["lang_entity_set"]
    lang_ss = align_pooled["lang_single_source"]
    orc_s = align_pooled["oracle_single"]
    orc_set = align_pooled["lang_oracle_set"]
    rnd = align_pooled["random"]
    attn = align_pooled["attention"]

    # per-state align for the statistical test (Gate D)
    def state_aligns(name):
        m = {}
        for r in rows:
            if r["selector"] == name:
                m[r["state_id"]] = r["align"]
        return [m[s] for s in state_ids if s in m]

    aa = np.asarray(state_aligns("action_l2"))
    rr = np.asarray(state_aligns("random"))
    d_ = aa - rr
    n_ = len(d_)
    se = d_.std(ddof=1) / np.sqrt(n_) if n_ > 1 else float("inf")
    z = d_.mean() / se if se > 0 else 0.0
    # two-sided p via normal approx (n=135 large)
    from math import erfc, sqrt
    pval = float(erfc(abs(z) / sqrt(2.0)))

    # ---- gates ----
    gateA = a2 > 0.62
    gateB = False
    htb = {}
    for t in HARD_TASKS:
        sub = [r for r in rows if r["task"] == t and r["selector"] == "action_l2"]
        la = [r for r in rows if r["task"] == t and r["selector"] == "lang_entity_set"]
        htb[t] = {
            "action_align": _mean([r["align"] for r in sub]),
            "action_iou": _med([r["iou"] for r in sub]),
            "lang_iou": _med([r["iou"] for r in la]),
        }
    gateB = any(htb[t]["action_iou"] > 0 for t in HARD_TASKS)
    gateC = a2 > lang_es + 0.05
    gateD = (a2 > rnd) and (pval < 0.05)

    # ---- stop ----
    stop_approx_lang = abs(a2 - lang_es) <= 0.02
    stop_no_pcd = (a2 < 0.62) and (not gateB)
    # structure: action-selected group IoU vs random IoU (CAT repro = structure ~ chance)
    act_iou = _med([r["iou"] for r in rows if r["selector"] == "action_l2"])
    rnd_iou = _med([r["iou"] for r in rows if r["selector"] == "random"])
    stop_cat_repro = act_iou <= rnd_iou + 0.02
    stop_hit = stop_approx_lang or stop_no_pcd or stop_cat_repro
    verdict = "PASS_TO_ROLLOUT" if (gateA and gateB and gateC and gateD and not stop_hit) \
        else "STOP_ACTION_CONDITIONED_NO_GO"

    # ---- task breakdown ----
    tb = []
    for t in sorted({r["task"] for r in rows}):
        def mi(sel, field="iou"):
            v = [r[field] for r in rows if r["task"] == t and r["selector"] == sel]
            return _mean(v)
        tb.append({"task": t,
                   "action_l2_iou": mi("action_l2"), "action_l2_align": mi("action_l2", "align"),
                   "lang_entity_set_iou": mi("lang_entity_set"),
                   "oracle_single_iou": mi("oracle_single"),
                   "random_iou": mi("random")})

    results = {
        "experiment": "ACTION_CONDITIONED_SEMANTIC_CD_PHASE0_V1",
        "split": a.split, "n_states": len([s for s in state_ids if (out_dir / f"{s}.npz").exists()]),
        "verdict": verdict, "stop_rule_hit": stop_hit,
        "alignment_pooled": {k: v for k, v in align_pooled.items()},
        "gates": {
            "A_alignment": {"PASS": bool(gateA), "action_l2": a2, "oracle_single": orc_s},
            "B_hard_task": {"PASS": bool(gateB), **htb},
            "C_over_language": {"PASS": bool(gateC), "action_l2": a2, "lang_entity_set": lang_es},
            "D_over_random": {"PASS": bool(gateD), "action_l2": a2, "random": rnd, "p_value": pval},
        },
        "ablations": {
            "A_semantic_vs_patch": {"action_l2_group": a2, "cat_patch": 0.343, "group_gt_patch": a2 > 0.343},
            "B_logit_vs_attention": {"action_l2": a2, "attention": attn, "logit_gt_attention": a2 > attn},
            "C_token0_vs_mean": {"action_l2_mean": a2, "action_l2_tok0": align_pooled["action_l2_tok0"]},
        },
        "disruption": {
            "action_l2_mean": _med([np.mean(d["l2_mean"]) for d in
                                    [np.load(out_dir / f"{s}.npz") for s in state_ids if (out_dir / f"{s}.npz").exists()]]),
            "action_exp_mean": _med([np.mean(np.load(out_dir / f"{s}.npz")["exp_mean"]) for s in state_ids if (out_dir / f"{s}.npz").exists()]),
        },
        "stop_flags": {
            "action_approx_lang": bool(stop_approx_lang),
            "no_pcd_direction": bool(stop_no_pcd),
            "cat_reproduction": bool(stop_cat_repro),
        },
        "task_breakdown": tb,
    }
    (art / "selector_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "ACTION_CONDITIONED_SEMANTIC_CD_PHASE0_V1",
        "status": verdict, "split": a.split, "date": "2026-08-24",
        "n_states": results["n_states"],
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "core_numbers": {
            "align_action_l2": a2, "align_action_kl": akl, "align_action_exp": aex,
            "align_lang_entity_set": lang_es, "align_lang_single_source": lang_ss,
            "align_oracle_single": orc_s, "align_oracle_set": orc_set,
            "align_attention": attn, "align_random": rnd,
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    with (art / "selector_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "selector", "iou", "hit", "align"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tb[0].keys()))
        w.writeheader()
        for r in tb:
            w.writerow(r)

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
