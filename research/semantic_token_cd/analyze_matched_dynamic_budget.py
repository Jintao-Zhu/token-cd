"""Matched dynamic-budget mechanism analysis over existing 800 states + shuffle episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
SHORT = {t: t.replace("google_robot_", "") for t in TASKS}
METRIC_COLS = (
    "bbox_area", "mean_spatial_distance", "components", "density",
    "cluster_intra_cosine_distance", "cluster_separation_margin",
    "semantic_centroid_cosine", "semantic_margin", "cluster_size_entropy",
    "matched_cluster_size_rank_mean",
)


def top_k(scores: np.ndarray, k: int) -> set[int]:
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(values)), -values))
    return set(int(x) for x in order[: max(1, min(int(k), len(values)))])


def top_p_count(scores: np.ndarray, tau: float) -> int:
    values = np.asarray(scores, dtype=np.float64)
    p = values / (values.sum() + 1e-12)
    order = np.lexsort((np.arange(len(p)), -p))
    return int(np.searchsorted(np.cumsum(p[order]), float(tau) - 1e-12) + 1)


def spearman(a, b):
    vals = []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        try:
            xv, yv = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xv) and np.isfinite(yv):
            vals.append((xv, yv))
    if len(vals) < 3 or len({x for x, _ in vals}) < 2 or len({y for _, y in vals}) < 2:
        return float("nan"), float("nan")
    x, y = zip(*vals)
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def load_metrics(root: Path):
    rows = []
    for task in TASKS:
        for path in sorted((root / "metrics" / task).glob("seed_*_step_*.json")):
            d = json.loads(path.read_text())
            rows.append(d)
    if len(rows) != 800:
        raise RuntimeError(f"expected 800 metric files, got {len(rows)}")
    return rows


def phase1(root: Path, lines: list[str]) -> dict:
    rows = load_metrics(root)
    report = {}
    lines += ["## Phase 1: What distinguishes small-m and large-m states?", "",
              "Spearman rho between matched m and each state metric. Computed per task.", "",
              "| Metric | open | close | pick | move |", "|---|---:|---:|---:|---:|"]
    for col in METRIC_COLS:
        vals = []
        for task in TASKS:
            sub = [r for r in rows if r["task"] == task]
            rho, _ = spearman([r["m_matched"] for r in sub], [r[col] for r in sub])
            vals.append(rho)
        lines.append(f"| {col} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} |")
        report[col] = dict(zip(TASKS, vals))
    lines += ["", "### Small / Middle / Large states: metric means", "",
              "Within each task, Small = lowest 25% m, Large = highest 25% m.", ""]
    for col in METRIC_COLS:
        lines += [f"#### {col}", "", "| Task | Small | Middle | Large |", "|---|---:|---:|---:|"]
        for task in TASKS:
            sub = sorted([r for r in rows if r["task"] == task], key=lambda r: r["m_matched"])
            n = len(sub); lo = max(1, n // 4); hi = min(n - 1, 3 * n // 4)
            groups = [sub[:lo], sub[lo:hi], sub[hi:]]
            means = [np.mean([r[col] for r in g]) if g else float("nan") for g in groups]
            lines.append(f"| {SHORT[task]} | {means[0]:.3f} | {means[1]:.3f} | {means[2]:.3f} |")
        lines.append("")
    return {"rows": rows, "correlations": report}


def phase2(root: Path, cal_root: Path, calibration: dict, lines: list[str]) -> dict:
    tau_entity = float(calibration["tau"]["entity"])
    rows = []
    metric_rows = load_metrics(root)
    # deterministic within-seed permutation for the 4 calibration states
    shuffle_k = {}
    for task in TASKS:
        by_seed = {}
        for r in metric_rows:
            if r["task"] == task:
                by_seed.setdefault(int(r["seed"]), []).append(r)
        task_index = TASKS.index(task)
        for seed, rs in by_seed.items():
            rs = sorted(rs, key=lambda r: r["step"])
            perm = np.random.default_rng(20260920 + task_index * 100003 + seed).permutation(len(rs))
            for i, r in enumerate(rs):
                shuffle_k[(task, seed, r["step"])] = int(rs[int(perm[i])]["m_matched"])
    for r in metric_rows:
        key = (r["task"], int(r["seed"]), int(r["step"]))
        mpath = root / "metrics" / r["task"] / f"seed_{int(r['seed']):03d}_step_{int(r['step']):03d}.npz"
        cpath = cal_root / "states" / r["task"] / f"seed_{int(r['seed']):03d}_step_{int(r['step']):03d}.npz"
        if not cpath.exists():
            cpath = cal_root / r["task"] / f"seed_{int(r['seed']):03d}_step_{int(r['step']):03d}.npz"
        if not mpath.exists() or not cpath.exists():
            continue
        npz = np.load(mpath); cal = np.load(cpath)
        G = set(int(x) for x in npz["G"])
        a_full = np.asarray(cal["a_full"], dtype=np.float64)
        a_entity = np.asarray(cal["a_entity"], dtype=np.float64)
        methods = {
            "matched": int(r["m_matched"]),
            "fixed34": 34,
            "entity_top_p": top_p_count(a_entity, tau_entity),
            "shuffle": shuffle_k[key],
        }
        row = {"task": r["task"], "seed": int(r["seed"]), "step": int(r["step"]), "m": int(r["m_matched"])}
        for name, k in methods.items():
            S = top_k(a_full, k)
            row[f"{name}_k"] = int(k)
            row[f"{name}_recall"] = len(S & G) / max(1, len(G))
            row[f"{name}_precision"] = len(S & G) / max(1, len(S))
        rows.append(row)
    report = {"tau_entity": tau_entity}
    lines += ["## Phase 2: Does matched m cover the KMeans task-related support?", "",
              "For each state, `G` is the matched KMeans token set and `S_k` is full-L11 Top-k.", "",
              "| Method | Recall | Precision |", "|---|---:|---:|"]
    for name in ("matched", "fixed34", "entity_top_p", "shuffle"):
        vals_r = [r[f"{name}_recall"] for r in rows]
        vals_p = [r[f"{name}_precision"] for r in rows]
        report[name] = {"recall": float(np.mean(vals_r)), "precision": float(np.mean(vals_p))}
        lines.append(f"| {name} | {np.mean(vals_r):.3f} | {np.mean(vals_p):.3f} |")
    lines += ["", "### By matched-m quartile", "", "| Task | Size | Matched recall | Fixed34 recall | Entity recall | Shuffle recall |", "|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        sub = sorted([r for r in rows if r["task"] == task], key=lambda r: r["m"])
        n = len(sub); third1 = n // 3; third2 = 2 * n // 3
        for label, group in (("small", sub[:third1]), ("middle", sub[third1:third2]), ("large", sub[third2:])):
            if not group: continue
            lines.append(f"| {SHORT[task]} | {label} | {np.mean([x['matched_recall'] for x in group]):.3f} | "
                         f"{np.mean([x['fixed34_recall'] for x in group]):.3f} | "
                         f"{np.mean([x['entity_top_p_recall'] for x in group]):.3f} | "
                         f"{np.mean([x['shuffle_recall'] for x in group]):.3f} |")
    return {"rows": rows, "summary": report}


def phase3(shuffle_root: Path, matched_root: Path, lines: list[str]) -> dict:
    rows = []
    seeds = range(100, 200)
    for task in TASKS:
        for seed in seeds:
            mp = matched_root / "episodes" / task / "l11_matched" / f"episode_{seed:03d}_summary.json"
            sp = shuffle_root / "episodes" / task / "matched_shuffle" / f"episode_{seed:03d}_summary.json"
            if not (mp.exists() and sp.exists()):
                continue
            m = json.loads(mp.read_text()); s = json.loads(sp.read_text())
            denom = max(1.0, float(s.get("mean_matched_m_t", 1.0)))
            d = float(s.get("mean_abs_delta_vs_current", 0.0)) / denom
            signed = (float(s.get("mean_m_t", 0.0)) - float(s.get("mean_matched_m_t", 0.0))) / denom
            matched_win = bool(m["success"]) and not bool(s["success"])
            shuffle_win = bool(s["success"]) and not bool(m["success"])
            rows.append({"task": task, "seed": seed, "D": d, "signed": signed,
                         "matched_win": matched_win, "shuffle_win": shuffle_win,
                         "matched_success": bool(m["success"]), "shuffle_success": bool(s["success"])})
    report = {"n": len(rows)}
    lines += ["", "## Phase 3: Does shuffle harm grow with budget mismatch?", "",
              "`D = mean |m_shuffle - m_matched| / m_matched`. Episode-level matched-vs-shuffle outcomes.", "",
              "| D bin | n | Matched | Shuffle | Matched-only | Shuffle-only | Net (matched-only − shuffle-only) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    bins = [("D < 20%", 0.0, 0.20), ("20–40%", 0.20, 0.40), ("D ≥ 40%", 0.40, float("inf"))]
    for label, lo, hi in bins:
        sub = [r for r in rows if lo <= r["D"] < hi]
        if not sub: continue
        mo = sum(r["matched_win"] for r in sub); so = sum(r["shuffle_win"] for r in sub)
        lines.append(f"| {label} | {len(sub)} | {sum(r['matched_success'] for r in sub)}/{len(sub)} | "
                     f"{sum(r['shuffle_success'] for r in sub)}/{len(sub)} | {mo} | {so} | {mo-so:+d} |")
        report[label] = {"n": len(sub), "matched_only": mo, "shuffle_only": so, "net": mo-so}
    lines += ["", "### Under- vs over-budget direction", "",
              "| Direction | n | Matched-only | Shuffle-only | Net |", "|---|---:|---:|---:|---:|"]
    directions = [("under (< −10%)", -float("inf"), -0.10), ("balanced", -0.10, 0.10), ("over (> +10%)", 0.10, float("inf"))]
    for label, lo, hi in directions:
        sub = [r for r in rows if lo <= r["signed"] < hi]
        if not sub: continue
        mo = sum(r["matched_win"] for r in sub); so = sum(r["shuffle_win"] for r in sub)
        lines.append(f"| {label} | {len(sub)} | {mo} | {so} | {mo-so:+d} |")
        report[label] = {"n": len(sub), "matched_only": mo, "shuffle_only": so, "net": mo-so}
    return {"rows": rows, "summary": report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--calibration-results", type=Path, required=True)
    ap.add_argument("--matched", type=Path, required=True)
    ap.add_argument("--shuffle", type=Path, required=True)
    args = ap.parse_args()
    root = args.artifact.resolve()
    lines = ["# Matched dynamic-budget mechanism analysis", "",
             "All analyses reuse existing states/episodes. No new rollout episodes.", ""]
    calibration = json.loads(args.calibration_results.read_text())
    p1 = phase1(root, lines)
    p2 = phase2(root, args.calibration.resolve(), calibration, lines)
    p3 = phase3(args.shuffle.resolve(), args.matched.resolve(), lines)
    out = {"protocol_id": "MATCHED_DYNAMIC_BUDGET_MECHANISM_ANALYSIS_V1",
           "phase1": p1["correlations"], "phase2": p2["summary"], "phase3": p3["summary"]}
    (root / "ANALYSIS_RESULTS.json").write_text(json.dumps(out, indent=2) + "\n")
    (root / "ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
