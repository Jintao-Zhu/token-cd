"""Calibrate global Top-P tau for Full/Entity/Generic L11 attention budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
VARIANTS = {"full": "a_full", "entity": "a_entity", "generic": "a_generic"}


def top_p_count(scores: np.ndarray, tau: float) -> int:
    p = np.asarray(scores, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    order = np.lexsort((np.arange(len(p)), -p))
    return int(np.searchsorted(np.cumsum(p[order]), tau - 1e-12) + 1)


def load_rows(root: Path):
    rows = []
    for task in TASKS:
        for meta_path in sorted((root / "states" / task).glob("seed_*_step_*.json")):
            npz_path = meta_path.with_suffix(".npz")
            if not npz_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            arrays = np.load(npz_path)
            rows.append({
                "task": task,
                "seed": int(meta["seed"]),
                "step": int(meta["step"]),
                "m_matched": int(meta["m_matched"]),
                "a_full": np.asarray(arrays["a_full"], dtype=np.float64),
                "a_entity": np.asarray(arrays["a_entity"], dtype=np.float64),
                "a_generic": np.asarray(arrays["a_generic"], dtype=np.float64),
            })
    return rows


def spearman(a, b):
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(a, b)
    return float(r), float(p)


def desc(values):
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
        "min": float(v.min()), "max": float(v.max()),
        "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    args = ap.parse_args()
    root = args.artifact.resolve()
    rows = load_rows(root)
    if len(rows) != 800:
        raise SystemExit(f"expected 800 states, got {len(rows)}")

    target_mean = float(np.mean([r["m_matched"] for r in rows]))
    # Fine enough to hit the target mean within a fraction of a token.
    taus = np.arange(0.500, 0.9995 + 1e-9, 0.0005)
    results = {}
    chosen = {}
    for name, key in VARIANTS.items():
        counts = []
        for tau in taus:
            values = [top_p_count(r[key], float(tau)) for r in rows]
            counts.append(float(np.mean(values)))
        counts = np.asarray(counts)
        idx = int(np.argmin(np.abs(counts - target_mean)))
        tau = float(taus[idx])
        for r in rows:
            r[f"m_{name}"] = top_p_count(r[key], tau)
        chosen[name] = tau
        results[name] = {
            "tau": tau,
            "mean": float(np.mean([r[f"m_{name}"] for r in rows])),
            "dist": desc([r[f"m_{name}"] for r in rows]),
            "rho_pooled": spearman([r["m_matched"] for r in rows], [r[f"m_{name}"] for r in rows]),
        }

    report = {"n_states": len(rows), "mean_matched": target_mean, "tau": chosen, "results": results}
    lines = ["# Entity-Conditioned Budget Calibration", "",
             f"States: **{len(rows)}** (4 tasks × 50 seeds × 4 progress states)", "",
             f"Calibration target: mean matched m = **{target_mean:.3f}**", "",
             "| Estimator | tau | mean m | std | median | P10 | P90 | rho vs matched |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("full", "entity", "generic"):
        r = results[name]
        d = r["dist"]
        lines.append(f"| {name} | {r['tau']:.4f} | {r['mean']:.2f} | {d['std']:.2f} | "
                     f"{d['median']:.1f} | {d['p10']:.0f} | {d['p90']:.0f} | {r['rho_pooled'][0]:.3f} |")
    lines += ["", "## Per-task Spearman", "", "| Task | full | entity | generic |", "|---|---:|---:|---:|"]
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task]
        vals = []
        for name in ("full", "entity", "generic"):
            vals.append(spearman([r["m_matched"] for r in sub], [r[f"m_{name}"] for r in sub])[0])
        lines.append(f"| {task.replace('google_robot_','')} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |")

    lines += ["", "## Within-episode centered Spearman", "", "| Estimator | rho | p |", "|---|---:|---:|"]
    for name in ("full", "entity", "generic"):
        a, b = [], []
        for task in TASKS:
            for seed in sorted({r["seed"] for r in rows if r["task"] == task}):
                sub = sorted([r for r in rows if r["task"] == task and r["seed"] == seed], key=lambda r: r["step"])
                if len(sub) < 2:
                    continue
                ma = np.asarray([r["m_matched"] for r in sub], float)
                mb = np.asarray([r[f"m_{name}"] for r in sub], float)
                a.extend((ma - ma.mean()).tolist()); b.extend((mb - mb.mean()).tolist())
        rho, p = spearman(a, b)
        lines.append(f"| {name} | {rho:.3f} | {p:.4f} |")
        report[f"within_episode_{name}"] = {"rho": rho, "p": p}

    (root / "CALIBRATION_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "CALIBRATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
