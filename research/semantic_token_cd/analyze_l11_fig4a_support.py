"""Offline Fig. 4a support-scale analysis from the completed Full L11-Matched runs."""
from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import statistics

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL = ROOT / "artifacts/l11_table2_main_ablation_v1/formal/full"
OUT = ROOT / "artifacts/l11_matched_dose_response_fig4_v1/fig4a"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SEEDS = range(100, 200)
DIAGONAL = math.sqrt(15**2 + 15**2)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = average_ranks(x), average_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def support_metrics(ids):
    ids = sorted(set(int(i) for i in ids))
    if not ids or min(ids) < 0 or max(ids) >= 256:
        raise ValueError(f"invalid/empty support ids: {ids[:8]}")
    xy = np.asarray([(i % 16, i // 16) for i in ids], dtype=np.float64)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    bbox = int((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1))
    centroid = xy.mean(axis=0)
    mean_dist = float(np.linalg.norm(xy - centroid, axis=1).mean())
    return len(ids), bbox, bbox / 256.0, mean_dist / DIAGONAL


def sample_episode_rows(task: str, seed: int):
    path = FULL / "episodes" / task / "full" / f"episode_{seed}_summary.json"
    data = json.loads(path.read_text())
    trace = data.get("selector_trace", [])
    if not trace:
        raise RuntimeError(f"empty Full selector trace: {path}")
    split = max(1, len(trace) // 2)
    halves = (("early", 0, trace[:split]),
              ("late", split, trace[split:] or trace))
    rows = []
    for phase, start_index, part in halves:
        selected_index = start_index + len(part) // 2
        step = part[len(part) // 2]
        ids = step.get("reference_shr_token_ids") or []
        support_count, bbox, bbox_norm, dist_norm = support_metrics(ids)
        m = int(step["m_t"])
        if m != support_count:
            raise RuntimeError(f"Full m/support mismatch at {task}:{seed}:{phase}: {m}!={support_count}")
        rows.append({
            "task": task,
            "seed": seed,
            "episode_id": data.get("episode_id", seed),
            "phase": phase,
            "step_id": int(step.get("step", step.get("step_id", selected_index))),
            "m_t": m,
            "support_count": support_count,
            "bbox_area": bbox,
            "bbox_area_normalized": bbox_norm,
            "mean_spatial_distance_normalized": dist_norm,
        })
    return rows


def episode_bootstrap(rows, metric: str, n_boot=1000, seed=20261005):
    by_episode = {}
    for row in rows:
        by_episode.setdefault((row["task"], row["seed"]), []).append(row)
    episodes = list(by_episode.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sampled = [episodes[rng.randrange(len(episodes))] for _ in range(len(episodes))]
        flat = [row for ep in sampled for row in ep]
        vals.append(spearman(
            np.asarray([r["m_t"] for r in flat], dtype=np.float64),
            np.asarray([r[metric] for r in flat], dtype=np.float64),
        ))
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task in TASKS:
        for seed in SEEDS:
            all_rows.extend(sample_episode_rows(task, seed))
    csv_path = OUT / "support_scale_states.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    result = {"protocol": "FIG4A_MATCHED_SUPPORT_SCALE_V1", "sampled_states": len(all_rows), "episodes": 400, "states_per_episode": 2, "tasks": {}, "metrics": {}}
    for task in TASKS:
        rows = [r for r in all_rows if r["task"] == task]
        task_name = task.replace("google_robot_", "")
        result["tasks"][task_name] = {"episodes": 100, "states": len(rows)}
        for metric in ("bbox_area_normalized", "mean_spatial_distance_normalized"):
            rho = spearman(np.asarray([r["m_t"] for r in rows]), np.asarray([r[metric] for r in rows]))
            result["metrics"].setdefault(metric, {})[task_name] = {
                "spearman_rho": rho,
                "episode_bootstrap_95ci": episode_bootstrap(rows, metric),
            }
    result_path = OUT / "support_scale_correlations.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for task, color in zip(TASKS, colors):
            rows = [r for r in all_rows if r["task"] == task]
            name = task.replace("google_robot_", "")
            axes[0].scatter([r["bbox_area_normalized"] for r in rows], [r["m_t"] for r in rows], s=13, alpha=.38, color=color, label=name)
            axes[1].scatter([r["mean_spatial_distance_normalized"] for r in rows], [r["m_t"] for r in rows], s=13, alpha=.38, color=color, label=name)
        axes[0].set(xlabel="Normalized support bounding-box area", ylabel="$m_t$", title="Matched budget vs support area")
        axes[1].set(xlabel="Normalized mean distance from support centroid", ylabel="$m_t$", title="Matched budget vs support spread")
        for ax in axes:
            ax.grid(alpha=.2)
        axes[1].legend(frameon=False, fontsize=8)
        fig.savefig(OUT / "fig4a_support_scale.png", dpi=220)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({"csv": str(csv_path), "summary": str(result_path), "plot": str(OUT / "fig4a_support_scale.png"), "states": len(all_rows), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
