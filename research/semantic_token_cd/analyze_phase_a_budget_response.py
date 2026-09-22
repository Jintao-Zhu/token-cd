"""Phase A analysis: does m_matched track the state's effective counterfactual budget?

Outputs REPORT.md + RESULTS.json under the artifact root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")


def knee_breakpoint(ms: np.ndarray, D: np.ndarray):
    """Two-segment piecewise-linear fit; return (breakpoint_m, info)."""
    n = len(ms)
    if n < 6:
        return None, {"reason": "too_few_points"}
    best = None
    for i in range(2, n - 2):            # at least 3 points per side
        xl, yl = ms[:i + 1], D[:i + 1]
        xr, yr = ms[i:], D[i:]
        if len(set(xl)) < 2 or len(set(xr)) < 2:
            continue
        cl = np.polyfit(xl, yl, 1)
        cr = np.polyfit(xr, yr, 1)
        sse = float(np.sum((yl - np.polyval(cl, xl)) ** 2) + np.sum((yr - np.polyval(cr, xr)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, i, cl, cr)
    if best is None:
        return None, {"reason": "no_candidate"}
    sse, i, cl, cr = best
    c1 = np.polyfit(ms, D, 1)
    sse1 = float(np.sum((D - np.polyval(c1, ms)) ** 2))
    r2_one = 1.0 - sse1 / max(1e-12, float(np.sum((D - D.mean()) ** 2)))
    r2_two = 1.0 - sse / max(1e-12, float(np.sum((D - D.mean()) ** 2)))
    slope_l, slope_r = float(cl[0]), float(cr[0])
    gain = r2_two - r2_one
    valid = bool(slope_l > 0 and slope_r < 0.35 * slope_l and gain > 0.05)
    return int(ms[i]), {"slope_left": slope_l, "slope_right": slope_r,
                        "r2_single_line": r2_one, "r2_two_segment": r2_two,
                        "r2_gain": gain, "valid": valid}


def m80(ms: np.ndarray, D: np.ndarray):
    lo, hi = float(D.min()), float(D.max())
    if hi - lo < 1e-9:
        return None
    thr = lo + 0.8 * (hi - lo)
    idx = np.flatnonzero(D >= thr)
    return int(ms[idx[0]]) if idx.size else None


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or len(set(a[ok])) < 2 or len(set(b[ok])) < 2:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(a[ok], b[ok])
    return float(r), float(p)


def cluster_bootstrap(rows, key_a, key_b, n_boot=5000, seed=20260918):
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    seeds = sorted(by_seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        aa, bb = [], []
        for s in pick:
            for r in by_seed[s]:
                va, vb = r.get(key_a), r.get(key_b)
                if va is None or vb is None:
                    continue
                aa.append(va); bb.append(vb)
        rr, _p = spearman(aa, bb)
        if np.isfinite(rr):
            out.append(rr)
    if not out:
        return (float("nan"), float("nan"), float("nan"))
    return (float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    args = ap.parse_args()
    root = args.artifact.resolve()
    pa = root / "phase_a"

    rows = []
    per_state = []
    for task in TASKS:
        for f in sorted((pa / task).glob("seed_*.json")):
            d = json.loads(f.read_text())
            seed = int(d["seed"])
            ep_ms, ep_md = [], []
            for st in d["states"]:
                ms = np.array([c["m"] for c in st["curve"]], float)
                D = np.array([c["D"] for c in st["curve"]], float)
                order = np.argsort(ms); ms, D = ms[order], D[order]
                b, info = knee_breakpoint(ms, D)
                m80v = m80(ms, D)
                rec = {
                    "task": task, "seed": seed, "step": int(st["step"]),
                    "progress": float(st["progress"]),
                    "matched_m": int(st["matched_m"]),
                    "topp80_m": int(st.get("topp80_m", -1)),
                    "topp85_m": int(st.get("topp85_m", -1)),
                    "knee_m": (int(b) if b is not None else None),
                    "valid_knee": bool(info.get("valid", False)),
                    "r2_gain": float(info.get("r2_gain", float("nan"))),
                    "m80": (int(m80v) if m80v is not None else None),
                    "D_max": float(D.max()), "D_min": float(D.min()),
                    "curve_m": ms.astype(int).tolist(),
                    "curve_D": D.astype(float).tolist(),
                }
                per_state.append(rec)
                if rec["knee_m"] is not None and rec["valid_knee"]:
                    rows.append(rec)
                    ep_ms.append(rec["matched_m"]); ep_md.append(rec["knee_m"])
            # episode-internal centred pairs
            if len(ep_ms) >= 2:
                cm = np.array(ep_ms, float) - np.mean(ep_ms)
                cd = np.array(ep_md, float) - np.mean(ep_md)
                for r, a, b in zip([x for x in rows if x["seed"] == seed and x["task"] == task], cm, cd):
                    r["matched_centered"] = float(a); r["knee_centered"] = float(b)

    def fmt(x, nd=3):
        return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

    L = ["# Phase A: Matched 预算 vs 有效反事实预算", "",
         f"状态总数: **{len(per_state)}**，其中有明确 knee 的: **{len(rows)}**", ""]
    if rows:
        m_matched = np.array([r["matched_m"] for r in rows], float)
        m_knee = np.array([r["knee_m"] for r in rows], float)
        r_pool, p_pool = spearman(m_matched, m_knee)
        L += ["## 总相关（pooled）", "",
              f"- Spearman(ρ) = **{fmt(r_pool)}** (p = {fmt(p_pool, 4)}), n = {len(rows)}", ""]
        if any("matched_centered" in r for r in rows):
            ca = [r.get("matched_centered", np.nan) for r in rows]
            cb = [r.get("knee_centered", np.nan) for r in rows]
            r_c, p_c = spearman(ca, cb)
            mean_b, lo_b, hi_b = cluster_bootstrap(rows, "matched_centered", "knee_centered")
            L += ["## episode 内中心化（真正的机制检验）", "",
                  f"- Spearman(Δmatched, Δknee) = **{fmt(r_c)}** (p = {fmt(p_c, 4)})",
                  f"- episode-cluster bootstrap(5000): mean = {fmt(mean_b)}, 95% CI = [{fmt(lo_b)}, {fmt(hi_b)}]", ""]
    L += ["## 逐任务", "", "| 任务 | n | Spearman(matched, knee) | mean |matched−knee| |", "|---|---:|---:|---:|"]
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task]
        if not sub:
            L.append(f"| {task} | 0 | — | — |"); continue
        a = [r["matched_m"] for r in sub]; b = [r["knee_m"] for r in sub]
        rr, _ = spearman(a, b)
        mad = float(np.mean(np.abs(np.array(a, float) - np.array(b, float))))
        L.append(f"| {task} | {len(sub)} | {fmt(rr)} | {mad:.2f} |")
    L += ["", "## 方法对比：|m_method − m_knee|", "",
          "| 方法 | mean | median | Spearman | |", "|---|---:|---:|---:|---|"]
    for name, key in (("matched", "matched_m"), ("TopP80", "topp80_m"), ("TopP85", "topp85_m"),
                      ("fixed32", None), ("m80(robust)", "m80")):
        vals, ks, ms_ = [], [], []
        for r in rows:
            v = 32 if key is None else r.get(key)
            k = r["knee_m"]
            if v is None or v < 0 or k is None:
                continue
            vals.append(abs(float(v) - float(k))); ks.append(k); ms_.append(float(v))
        if not vals:
            L.append(f"| {name} | — | — | — |"); continue
        rr, _ = spearman(ms_, ks)
        L.append(f"| {name} | {np.mean(vals):.2f} | {np.median(vals):.2f} | {fmt(rr)} | |")

    res = {"n_states": len(per_state), "n_valid_knee": len(rows),
           "valid_knee_rate": (len(rows) / len(per_state) if per_state else 0.0),
           "per_state": per_state}
    if rows:
        res["pooled_spearman"] = spearman([r["matched_m"] for r in rows], [r["knee_m"] for r in rows])
        res["centered_spearman"] = spearman([r.get("matched_centered", np.nan) for r in rows],
                                            [r.get("knee_centered", np.nan) for r in rows])
    (root / "PHASE_A_RESULTS.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (root / "PHASE_A_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
