"""Phase 1A Global Spatial Merge CD — gate + diagnostics.

Primary (decision-driving), per CD arm vs vanilla on the 3 tasks:
    SR_t(a)    = mean success
    R_t(a)     = #{seed : vanilla fails AND a succeeds}   (rescue)
    H_t(a)     = #{seed : vanilla succeeds AND a fails}   (harm)
    Net_t(a)   = R_t(a) - H_t(a)
    dSR_t(a)   = SR_t(a) - SR_t(vanilla)
    dSR_macro  = mean over tasks of dSR_t
    McNemar    = paired significance test on discordant pairs (1 df, continuity corr.)
    95% CI     = paired-proportions CI on dSR_t

Gate (Phase 1A GO) — evaluated independently per merge strength eta in
{gsm_025, gsm_050, gsm_100}:
    G1  #tasks with Net > 0            >= 2
    G2  dSR_macro                     >= +0.03
    G3  min_t dSR_t                    >  -0.10
If NO eta satisfies all three -> "Fixed Global Spatial Merge NO-GO" (do not enter 1B).
If one or more eta passes, pick the argmax dSR_macro (tie-break: Net, then min dSR_t).

Secondary (diagnostic), from the stored per-step trace + logit arrays:
    mean_residual_norm  = mean ||r_t|| over the episode's CD steps
    action_flip_rate    = P(final_token_ids != positive_token_ids)
    first_step_cos_gsm_attn = cos(r_0^gsm, r_0^attn)  (shared initial state)
    within_arm_temporal_cos = mean cos(r_t, r_{t-1})  (per CD arm)
    first_step_action_div   = ||a_0^arm - a_0^vanilla||

Usage:
  <venv>/bin/python research/semantic_token_cd/analyze_global_merge.py \
    --artifact artifacts/attn_global_merge_v1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
VANILLA = "vanilla"
ATTN_ARM = "semantic_attn_cd"
GSM_ARMS = ("gsm_025", "gsm_050", "gsm_100")
ETAS = {"gsm_025": 0.25, "gsm_050": 0.5, "gsm_100": 1.0}
CD_ARMS = (ATTN_ARM, *GSM_ARMS)


def _scipy_mcnemar(r: int, h: int) -> float:
    """Two-sided McNemar p-value with continuity correction, using scipy if present."""
    try:
        from scipy.stats import chi2
    except Exception:
        # Fallback: no continuity correction, standard McNemar chi-square.
        disc = r + h
        if disc == 0:
            return 1.0
        stat = (abs(r - h)) ** 2 / disc
        return float(math.exp(-stat / 2) * 2) if stat > 0 else 1.0  # crude, not exact
    disc = r + h
    if disc == 0:
        return 1.0
    stat = (abs(r - h) - 1.0) ** 2 / disc if (r + h) > 0 else 0.0
    return float(chi2.sf(stat, df=1))


def paired_dsr_ci(r: int, h: int, n: int) -> tuple[float, float]:
    """Standard paired-proportions 95% CI for dSR = (R - H)/n."""
    d = (r - h) / n if n else 0.0
    if n == 0:
        return 0.0, 0.0
    var = (r + h) / (n * n) - ((r - h) ** 2) / (n ** 3)
    var = max(var, 0.0)
    se = math.sqrt(var)
    z = 1.96
    return d - z * se, d + z * se


def _load_summaries(artifact: Path, task: str, arm: str) -> dict[int, dict]:
    d = {}
    adir = artifact / "episodes" / task / arm
    if not adir.exists():
        return d
    for f in adir.glob("episode_*_summary.json"):
        seed = int(f.name.split("_")[1])
        d[seed] = json.loads(f.read_text())
    return d


def _load_arrays(artifact: Path, task: str, arm: str, seed: int) -> dict | None:
    p = artifact / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=False))


def _residual(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Residual vectors [7,256] -> flattened float32 (log_softmax difference)."""
    eps = 1e-12
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)

    def ls(x):
        m = x.max(axis=-1, keepdims=True)
        e = np.exp(x - m)
        return x - m - np.log(e.sum(axis=-1, keepdims=True))

    return (ls(pos) - ls(neg)).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--min_seed", type=int, default=200)
    parser.add_argument("--max_seed", type=int, default=299)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    seed_range = range(args.min_seed, args.max_seed + 1)

    out: dict = {
        "artifact": str(artifact),
        "tasks": {},
        "gate": {},
    }

    # ---- primary per-task SR / R / H / Net + CI ----
    for task in TASKS:
        v = _load_summaries(artifact, task, VANILLA)
        paired = [s for s in seed_range if s in v and all(
            s in _load_summaries(artifact, task, a) for a in CD_ARMS
        )]
        if not paired:
            continue
        n = len(paired)
        t: dict = {"n_paired": n, "n_vanilla": len(v), "sr": {}}
        succ = {}
        for arm in (VANILLA, *CD_ARMS):
            m = _load_summaries(artifact, task, arm)
            succ[arm] = {s: bool(m[s]["success"]) for s in paired}
            t["sr"][arm] = float(sum(succ[arm].values()) / n)
        for arm in CD_ARMS:
            r = sum(1 for s in paired if not succ[VANILLA][s] and succ[arm][s])
            h = sum(1 for s in paired if succ[VANILLA][s] and not succ[arm][s])
            lo, hi = paired_dsr_ci(r, h, n)
            t[arm] = {
                "rescue": r, "harm": h, "net": r - h,
                "delta_sr": float(t["sr"][arm] - t["sr"][VANILLA]),
                "delta_sr_ci95": [round(lo, 4), round(hi, 4)],
                "mcnemar_p": round(_scipy_mcnemar(r, h), 6),
                "rescue_rate": float(r / n), "harm_rate": float(h / n),
            }
        out["tasks"][task] = t

    present = [k for k in out["tasks"]]
    if not present:
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    # ---- secondary diagnostics (internal metrics) ----
    diag: dict = {}
    for task in present:
        diag[task] = {}
        # first-step cross-arm cosine + per-arm temporal cos + flip rate
        arm_meta = {a: _load_summaries(artifact, task, a) for a in CD_ARMS}
        # iterate the shared paired seeds
        vsum = _load_summaries(artifact, task, VANILLA)
        paired = [s for s in seed_range if s in vsum and all(
            s in arm_meta[a] for a in CD_ARMS
        )]
        for arm in CD_ARMS:
            norms = []
            flips = []
            temporal = []
            for s in paired:
                tr = arm_meta[arm][s].get("selector_trace", [])
                if not tr:
                    continue
                cd = [x for x in tr if not x.get("oracle_empty", False)]
                for x in cd:
                    norms.append(float(x.get("residual_norm", 0.0)))
                    if "positive_token_ids" in x and "final_token_ids" in x:
                        flips.append(1 if x["positive_token_ids"] != x["final_token_ids"] else 0)
            # temporal cosine from npz arrays (arm vs itself)
            for s in paired:
                arr = _load_arrays(artifact, task, arm, s)
                if not arr or "negative_logits" not in arr:
                    continue
                pos = arr["positive_logits"]
                neg = arr["negative_logits"]
                T = min(pos.shape[0], neg.shape[0])
                for t in range(1, T):
                    temporal.append(_cos(
                        _residual(pos[t], neg[t]), _residual(pos[t - 1], neg[t - 1])
                    ))
            diag[task][arm] = {
                "mean_residual_norm": float(np.mean(norms)) if norms else None,
                "action_flip_rate": float(np.mean(flips)) if flips else None,
                "within_arm_temporal_cos": float(np.mean(temporal)) if temporal else None,
            }
        # cross-arm first-step cosine: gsm_* vs semantic_attn_cd at t=0 (shared state)
        for garm in GSM_ARMS:
            coses = []
            divs = []
            for s in paired:
                g = _load_arrays(artifact, task, garm, s)
                a = _load_arrays(artifact, task, ATTN_ARM, s)
                vv = _load_arrays(artifact, task, VANILLA, s)
                if not (g and a and "negative_logits" in g and "negative_logits" in a):
                    continue
                r_g = _residual(g["positive_logits"][0], g["negative_logits"][0])
                r_a = _residual(a["positive_logits"][0], a["negative_logits"][0])
                coses.append(_cos(r_g, r_a))
                if vv is not None and vv["executed_actions"].shape[0] > 0 and g["executed_actions"].shape[0] > 0:
                    divs.append(float(np.linalg.norm(
                        g["executed_actions"][0] - vv["executed_actions"][0]
                    )))
            diag[task][f"{garm}_vs_attn_first_step_cos"] = float(np.mean(coses)) if coses else None
            diag[task][f"{garm}_first_step_action_div"] = float(np.mean(divs)) if divs else None
    out["diagnostics"] = diag

    # ---- gate evaluation ----
    gate: dict = {"phase": "1A", "eta_arms": {}, "verdict": None}
    for garm in GSM_ARMS:
        nets = [out["tasks"][task][garm]["net"] for task in present]
        dsrs = [out["tasks"][task][garm]["delta_sr"] for task in present]
        d_macro = float(np.mean(dsrs))
        g1 = sum(1 for x in nets if x > 0)
        g2 = d_macro >= 0.03
        g3 = min(dsrs) > -0.10
        gate["eta_arms"][garm] = {
            "eta": ETAS[garm],
            "per_task_net": {task: out["tasks"][task][garm]["net"] for task in present},
            "per_task_delta_sr": {task: out["tasks"][task][garm]["delta_sr"] for task in present},
            "delta_sr_macro": round(d_macro, 4),
            "g1_tasks_net_gt0": g1,
            "g2_macro_sr_ge_3pp": g2,
            "g3_min_delta_sr_gt_m10pp": g3,
            "pass": g1 >= 2 and g2 and g3,
        }
    passing = [g for g in GSM_ARMS if gate["eta_arms"][g]["pass"]]
    if passing:
        best = max(passing, key=lambda g: (
            gate["eta_arms"][g]["delta_sr_macro"],
            sum(gate["eta_arms"][g]["per_task_net"].values()),
            min(gate["eta_arms"][g]["per_task_delta_sr"].values()),
        ))
        gate["verdict"] = "GO"
        gate["best_eta"] = best
        gate["best_eta_value"] = ETAS[best]
    else:
        gate["verdict"] = "NO-GO"
        gate["best_eta"] = None
        gate["best_eta_value"] = None
        gate["note"] = ("Fixed Global Spatial Merge NO-GO: no eta passes G1&G2&G3; "
                        "do not enter Phase 1B.")
    out["gate"] = gate

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
