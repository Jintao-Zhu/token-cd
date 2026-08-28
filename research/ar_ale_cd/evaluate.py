#!/usr/bin/env python3
"""ALE-CD Phase-0 evaluation: baselines B1/B2/B3, metrics M1-M4, gates A-E.

Offline token-level only; reads ale_traj_conf.npz (200 Confirmation states,
vanilla full-layer trajectory). Produces all spec artifacts + phase0_decision.json.

Run:
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/ar_ale_cd/evaluate.py \
    --workspace . --artifact artifacts/ar_ale_cd_phase0_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research.ar_ale_cd.trajectory import trajectory_score_raw, softmax_256

N_LAYERS = 32
FINAL = 31
AMCD_MID = 11
N_RAND_PERMS = 20
RAND_SEED = 20260822
LAMBDAS = [0.05, 0.10, 0.25]


def gather(z, idx):
    """z [...,256], idx [...] -> [...] element at idx along last axis."""
    return np.take_along_axis(z, idx[..., None], axis=-1)[..., 0]


def rank_of(z, idx):
    """1-based rank of z[idx] within its row (smaller = higher)."""
    val = gather(z, idx)
    return 1 + (z > val[..., None]).sum(axis=-1)


def cp_rank(dist, a_star, a_v, wrong):
    """Return (n_corrected, n_wrong) where corrected = dist(a*) > dist(a_v)."""
    d_star = gather(dist, a_star)
    d_v = gather(dist, a_v)
    m = wrong & (d_star > d_v)
    return int(m.sum()), int(wrong.sum())


def fix_harm(pred, a_v, a_star):
    fix = (a_v != a_star) & (pred == a_star)
    harm = (a_v == a_star) & (pred != a_star)
    return int(fix.sum()), int(harm.sum())


def frac(num, den):
    return float(num / den) if den else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    a = p.parse_args()
    w = a.workspace.resolve()
    art = a.artifact.resolve()

    d = np.load(art / "ale_traj_conf.npz")
    z = d["z_vanilla"]              # [N,32,7,256]
    expert = d["expert_local"]      # [N,7] local action-token index
    tasks = list(d["tasks"])
    state_ids = list(d["state_ids"])
    n = z.shape[0]

    z_fin = z[:, FINAL]
    p_fin = softmax_256(z_fin)
    a_v = z_fin.argmax(axis=2)      # vanilla final argmax
    a_star = expert
    wrong = a_v != a_star
    n_wrong = int(wrong.sum())
    n_pos = n * 7

    # ---- ALE latent distribution ----
    score, comps = trajectory_score_raw(z)   # [N,7,256]
    q = softmax_256(score)
    a_ale = q.argmax(axis=2)

    # Metric 1 / 1b
    nc_rank, _ = cp_rank(score, a_star, a_v, wrong)
    cp_rank_val = frac(nc_rank, n_wrong)
    n_corr_argmax = int((wrong & (a_ale == a_star)).sum())
    cp_argmax_val = frac(n_corr_argmax, n_wrong)

    # Metric 2 rank
    rank_fin = rank_of(z_fin, a_star)
    rank_q = rank_of(score, a_star)          # rank in q == rank in score (monotonic)
    rank_improvement = rank_fin - rank_q     # positive = improved

    # Metric 3 Fix/Harm (a_ale = argmax(q))
    fix_n, harm_n = fix_harm(a_ale, a_v, a_star)
    fh_ratio = float(fix_n / harm_n) if harm_n else float("inf")

    # ---- Metric 4: controls ----
    amcd_rows = []
    for lam in [1.0, 0.25]:
        z_amcd = z_fin + lam * (z_fin - z[:, AMCD_MID])
        a_amcd = z_amcd.argmax(axis=2)
        nc, _ = cp_rank(z_amcd, a_star, a_v, wrong)
        f, h = fix_harm(a_amcd, a_v, a_star)
        amcd_rows.append({
            "method": "AMCD", "lambda": lam,
            "cp_rank": frac(nc, n_wrong),
            "cp_argmax": frac(int((wrong & (a_amcd == a_star)).sum()), n_wrong),
            "fix": f, "harm": h, "fix_harm_ratio": float(f / h) if h else float("inf"),
        })

    rng = np.random.default_rng(RAND_SEED)
    rand_rows = []
    rand_cp = []
    for k in range(N_RAND_PERMS):
        perm = rng.permutation(N_LAYERS)
        score_r, _ = trajectory_score_raw(z, layer_perm=perm)
        a_rand = score_r.argmax(axis=2)
        nc, _ = cp_rank(score_r, a_star, a_v, wrong)
        f, h = fix_harm(a_rand, a_v, a_star)
        cp_r = frac(nc, n_wrong)
        rand_cp.append(cp_r)
        rand_rows.append({
            "trial": k, "cp_rank": cp_r,
            "cp_argmax": frac(int((wrong & (a_rand == a_star)).sum()), n_wrong),
            "fix": f, "harm": h,
        })
    rand_cp = np.asarray(rand_cp)
    rand_cp_mean = float(np.nanmean(rand_cp))
    rand_cp_std = float(np.nanstd(rand_cp))

    # ---- gates ----
    tasks_arr = np.asarray(tasks)
    task_rows = []
    n_task_pos = 0
    for t in sorted(set(tasks)):
        tm = (tasks_arr == t)[:, None]
        w_t = wrong & tm
        nc_t, nw_t = cp_rank(score, a_star, a_v, w_t)
        cp_t = frac(nc_t, nw_t)
        if not np.isnan(cp_t) and cp_t > 0:
            n_task_pos += 1
        task_rows.append({"task": t, "n_wrong": nw_t, "n_corrected": nc_t, "cp_rank": cp_t})

    dim_rows = []
    n_dim_pos = 0
    for j in range(7):
        w_j = wrong[:, j]
        nc_j, nw_j = cp_rank(score[:, j], a_star[:, j], a_v[:, j], w_j)
        cp_j = frac(nc_j, nw_j)
        if not np.isnan(cp_j) and cp_j > 0:
            n_dim_pos += 1
        dim_rows.append({"dim": j, "n_wrong": nw_j, "n_corrected": nc_j, "cp_rank": cp_j})

    gateA = cp_rank_val >= 0.60
    gateB = fh_ratio > 1.5
    gateC = (cp_rank_val - rand_cp_mean) >= 0.05
    gateD = n_task_pos >= 8
    gateE = n_dim_pos >= 5

    verdict_reason = []
    if cp_rank_val < 0.55:
        verdict = "NO_GO"
        verdict_reason.append("CorrectionPreference < 0.55 (literal STOP RULE)")
    elif gateA and gateB and gateC and gateD and gateE:
        verdict = "GO"
        verdict_reason.append("all gates A-E pass")
    elif (not gateB) or (not gateC):
        # taxonomy gap: CP>=0.60 but the two "is it a real signal?" gates fail.
        # Fix/Harm <= 1.5 (net-harmful argmax) or trajectory not beating random
        # => no usable correction signal -> NO_GO in substance.
        verdict = "NO_GO"
        verdict_reason.append(
            "Gate B (Fix/Harm) or Gate C (vs random trajectory) fails -> "
            "no usable correction signal, even though CP_rank >= 0.60")
    else:
        verdict = "WEAK"
        verdict_reason.append("marginal: CP in weak range or small advantage")

    # ---- secondary: contrastive z_new(lambda) ----
    contrastive = []
    logq = np.log(q + 1e-12)
    logpf = np.log(p_fin + 1e-12)
    for lam in LAMBDAS:
        z_new = z_fin + lam * (logq - logpf)
        a_new = z_new.argmax(axis=2)
        nc, _ = cp_rank(z_new, a_star, a_v, wrong)
        f, h = fix_harm(a_new, a_v, a_star)
        contrastive.append({
            "lambda": lam,
            "cp_rank": frac(nc, n_wrong),
            "cp_argmax": frac(int((wrong & (a_new == a_star)).sum()), n_wrong),
            "fix": f, "harm": h, "fix_harm_ratio": float(f / h) if h else float("inf"),
        })

    # ---- write artifacts ----
    np.save(art / "ale_scores.npy", score.astype(np.float32))
    np.save(art / "latent_distribution.npy", q.astype(np.float32))

    # correction_analysis.csv (per state x position)
    with (art / "correction_analysis.csv").open("w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["state_id", "task_id", "position", "vanilla_token", "expert_token",
                       "is_wrong", "ale_token", "score_expert", "score_vanilla",
                       "q_expert", "q_vanilla", "rank_final_expert", "rank_q_expert",
                       "q_recovers_expert", "ale_flips_to_expert"])
        for i in range(n):
            for j in range(7):
                wcsv.writerow([
                    state_ids[i], tasks[i], j,
                    int(a_v[i, j]), int(a_star[i, j]), int(wrong[i, j]),
                    int(a_ale[i, j]),
                    round(float(score[i, j, a_star[i, j]]), 6), round(float(score[i, j, a_v[i, j]]), 6),
                    round(float(q[i, j, a_star[i, j]]), 6), round(float(q[i, j, a_v[i, j]]), 6),
                    int(rank_fin[i, j]), int(rank_q[i, j]),
                    int(wrong[i, j] and score[i, j, a_star[i, j]] > score[i, j, a_v[i, j]]),
                    int(wrong[i, j] and a_ale[i, j] == a_star[i, j]),
                ])

    # fix_harm_analysis.csv
    with (art / "fix_harm_analysis.csv").open("w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["state_id", "task_id", "position", "vanilla_token", "expert_token",
                       "ale_token", "category"])
        for i in range(n):
            for j in range(7):
                cat = "neutral"
                if a_v[i, j] != a_star[i, j] and a_ale[i, j] == a_star[i, j]:
                    cat = "fix"
                elif a_v[i, j] == a_star[i, j] and a_ale[i, j] != a_star[i, j]:
                    cat = "harm"
                wcsv.writerow([state_ids[i], tasks[i], j,
                               int(a_v[i, j]), int(a_star[i, j]), int(a_ale[i, j]), cat])

    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=["task", "n_wrong", "n_corrected", "cp_rank"])
        wcsv.writeheader()
        for r in task_rows:
            wcsv.writerow(r)

    with (art / "dimension_breakdown.csv").open("w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=["dim", "n_wrong", "n_corrected", "cp_rank"])
        wcsv.writeheader()
        for r in dim_rows:
            wcsv.writerow(r)

    with (art / "random_control.csv").open("w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=["trial", "cp_rank", "cp_argmax", "fix", "harm"])
        wcsv.writeheader()
        for r in rand_rows:
            wcsv.writerow(r)

    with (art / "amcd_comparison.csv").open("w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=["method", "lambda", "cp_rank", "cp_argmax", "fix", "harm", "fix_harm_ratio"])
        wcsv.writeheader()
        wcsv.writerow({"method": "Vanilla_B1", "lambda": 0.0,
                       "cp_rank": 0.0, "cp_argmax": 0.0,
                       "fix": 0, "harm": 0, "fix_harm_ratio": 0.0})
        wcsv.writerow({"method": "ALE_CD", "lambda": "n/a",
                       "cp_rank": cp_rank_val, "cp_argmax": cp_argmax_val,
                       "fix": fix_n, "harm": harm_n, "fix_harm_ratio": fh_ratio})
        for r in amcd_rows:
            wcsv.writerow(r)

    results = {
        "experiment": "AR_ALE_CD_PHASE0_V1",
        "step": "offline token-level evaluation",
        "n_states": n, "n_positions": n_pos, "n_vanilla_wrong": n_wrong,
        "verdict": verdict,
        "verdict_reason": " | ".join(verdict_reason),
        "metrics": {
            "correction_preference_rank": cp_rank_val,
            "correction_preference_argmax": cp_argmax_val,
            "fix_count": fix_n, "harm_count": harm_n, "fix_harm_ratio": fh_ratio,
            "mean_rank_final": float(np.mean(rank_fin)),
            "mean_rank_q": float(np.mean(rank_q)),
            "mean_rank_improvement": float(np.mean(rank_improvement)),
            "mean_rank_improvement_wrong": float(np.mean(rank_improvement[wrong])) if n_wrong else float("nan"),
        },
        "controls": {
            "amcd": amcd_rows,
            "random_cp_mean": rand_cp_mean, "random_cp_std": rand_cp_std,
            "random_cp_min": float(np.nanmin(rand_cp)), "random_cp_max": float(np.nanmax(rand_cp)),
            "ale_minus_random": float(cp_rank_val - rand_cp_mean),
        },
        "contrastive_update": contrastive,
        "gates": {
            "A": {"value": cp_rank_val, "threshold": 0.60, "PASS": bool(gateA)},
            "B": {"value": fh_ratio, "threshold": 1.5, "PASS": bool(gateB)},
            "C": {"value": float(cp_rank_val - rand_cp_mean), "threshold": 0.05, "PASS": bool(gateC)},
            "D": {"value": f"{n_task_pos}/10", "threshold": ">=8", "PASS": bool(gateD)},
            "E": {"value": f"{n_dim_pos}/7", "threshold": ">=5", "PASS": bool(gateE)},
        },
        "task_breakdown": task_rows,
        "dimension_breakdown": dim_rows,
        "component_medians": {
            "A_expert_minus_mean": float(np.mean(gather(comps["A"], a_star) - comps["A"].mean(axis=2))),
            "B_expert_minus_mean": float(np.mean(gather(comps["B"], a_star) - comps["B"].mean(axis=2))),
            "C_expert_minus_mean": float(np.mean(gather(comps["C"], a_star) - comps["C"].mean(axis=2))),
        },
    }
    (art / "evaluation_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "AR_ALE_CD_PHASE0_V1",
        "status": verdict,
        "verdict": {"GO": "ALE_CD_PHASE0_GO", "WEAK": "ALE_CD_PHASE0_WEAK", "NO_GO": "STOP_ALE_CD_NO_GO"}[verdict],
        "verdict_reason": " | ".join(verdict_reason),
        "date": "2026-08-23",
        "confirmation_n": n,
        "n_vanilla_wrong": n_wrong,
        "core_numbers": {
            "CorrectionPreference_rank": cp_rank_val,
            "CorrectionPreference_argmax": cp_argmax_val,
            "Fix_Harm_ratio": fh_ratio,
            "fix_count": fix_n, "harm_count": harm_n,
            "random_cp_mean": rand_cp_mean,
            "ale_minus_random": float(cp_rank_val - rand_cp_mean),
            "per_task_positive": f"{n_task_pos}/10",
            "per_dim_positive": f"{n_dim_pos}/7",
        },
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "closed_loop": "NOT_RUN",
        "amcd_baseline_cp_rank_lam1": amcd_rows[0]["cp_rank"],
    }
    (art / "phase0_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
