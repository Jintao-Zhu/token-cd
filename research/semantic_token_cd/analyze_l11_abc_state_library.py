#!/usr/bin/env python3
"""Analyze OpenVLA L11 A/B state library against independent SAPIEN masks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata, spearmanr


def values_int(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from values_int(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from values_int(value)
    elif isinstance(obj, int) and not isinstance(obj, bool):
        yield obj


def component_count(mask: np.ndarray) -> int:
    todo = set(map(int, np.flatnonzero(mask)))
    count = 0
    while todo:
        count += 1
        stack = [todo.pop()]
        while stack:
            i = stack.pop()
            r, c = divmod(i, 16)
            for j in ((r - 1) * 16 + c if r else -1,
                      (r + 1) * 16 + c if r < 15 else -1,
                      r * 16 + c - 1 if c else -1,
                      r * 16 + c + 1 if c < 15 else -1):
                if j in todo:
                    todo.remove(j)
                    stack.append(j)
    return count


def overlap_grid(seg: np.ndarray, ids: list[int]) -> np.ndarray:
    binary = np.isin(seg, ids).astype(np.float32)
    # OpenVLA's SIMPLER preprocessing resizes the 512x640 camera image directly
    # to 224x224 with INTER_AREA; preserve fractional pixels through 14x14 patch pooling.
    resized = cv2.resize(binary, (224, 224), interpolation=cv2.INTER_AREA)
    return resized.reshape(16, 14, 16, 14).mean(axis=(1, 3)).reshape(256)


def summarize_region(task: str, seg: np.ndarray, entities: dict) -> dict:
    if task in {"google_robot_open_drawer", "google_robot_close_drawer"}:
        groups = {"target": list(values_int(entities.get("target_drawer_link", {})))}
    elif task == "google_robot_pick_coke_can":
        groups = {"target": list(values_int(entities.get("target_object", {})))}
    else:
        groups = {
            "source": list(values_int(entities.get("source_object", {}))),
            "target": list(values_int(entities.get("target_object", {}))),
        }
        groups["target_union"] = groups["source"] + groups["target"]
    grids = {name: overlap_grid(seg, ids) for name, ids in groups.items() if ids}
    if "target" in grids:
        primary = grids["target"]
    else:
        primary = grids["target_union"]
    occupied = primary > 0
    coords = np.argwhere(occupied.reshape(16, 16))
    if len(coords) == 0:
        raise ValueError(f"no target patches for {task}; IDs={groups}")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    centroid = coords.mean(axis=0)
    dist = np.linalg.norm(coords - centroid, axis=1)
    target_ids = list(values_int(groups))
    visible_pixels = int(np.isin(seg, target_ids).sum())
    bbox_pixels = int((r1 - r0 + 1) * (c1 - c0 + 1) * 14 * 14)
    return {
        "gt_visible_patches": int(occupied.sum()),
        "gt_bbox_area_patches": int((r1 - r0 + 1) * (c1 - c0 + 1)),
        "gt_bbox_area_fraction": float((r1 - r0 + 1) * (c1 - c0 + 1) / 256),
        "gt_mask_area_fraction_model_view": float(primary.sum() / 256),
        "gt_mean_spatial_distance_grid": float(dist.mean() / np.sqrt(2 * 15**2)),
        "gt_bbox_fill_fraction": float(occupied.sum() / ((r1 - r0 + 1) * (c1 - c0 + 1))),
        "gt_visible_pixel_count_source_resolution": visible_pixels,
        "gt_bbox_pixel_count_model_view_approx": bbox_pixels,
        "region_patch_overlap": primary,
        "region_groups": grids,
    }


def partial_rank_corr(rows: list[dict]) -> float:
    """Rank partial association after task and phase controls.

    Visible area is itself the support-size measurement under test; controlling
    for it would remove the target variable mechanically. An amodal object mask
    is unavailable, so occlusion-adjusted visibility is not claimed.
    """
    x = rankdata([r["m_matched"] for r in rows])
    y = rankdata([r["gt_visible_patches"] for r in rows])
    task_names = sorted({r["task"] for r in rows})
    task_cols = [[float(r["task"] == name) for r in rows] for name in task_names[1:]]
    phase = [float(r["phase_quartile"]) for r in rows]
    design = np.column_stack([np.ones(len(rows)), *task_cols, phase])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_spearman(rows: list[dict], xkey: str, ykey: str, reps: int, rng) -> tuple[float, float, float]:
    x = np.asarray([r[xkey] for r in rows], dtype=float)
    y = np.asarray([r[ykey] for r in rows], dtype=float)
    rho = float(spearmanr(x, y).statistic)
    episodes = sorted({(r["task"], r["seed"]) for r in rows})
    lookup = {(r["task"], r["seed"]): [] for r in rows}
    for r in rows:
        lookup[(r["task"], r["seed"])].append(r)
    vals = []
    for _ in range(reps):
        picked = rng.integers(0, len(episodes), len(episodes))
        sample = [r for j in picked for r in lookup[episodes[j]]]
        b = spearmanr([r[xkey] for r in sample], [r[ykey] for r in sample]).statistic
        if np.isfinite(b):
            vals.append(float(b))
    ci = np.quantile(vals, [0.025, 0.975]) if vals else (float("nan"), float("nan"))
    return rho, float(ci[0]), float(ci[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261009)
    rows = []
    for meta_path in sorted((args.library / "states").glob("*/*.json")):
        meta = json.loads(meta_path.read_text())
        arr_path = meta_path.with_suffix(".npz")
        with np.load(arr_path, allow_pickle=False) as arr:
            seg = arr["segmentation_instance"]
        reg = summarize_region(meta["task"], seg, meta["target_entities"])
        overlap = reg.pop("region_patch_overlap")
        groups = reg.pop("region_groups")
        selected = np.asarray(meta["selected_token_ids"], dtype=int)
        selected_mask = np.zeros(256, dtype=bool)
        selected_mask[selected] = True
        target = overlap > 0
        target_precision = float((selected_mask & target).sum() / max(selected_mask.sum(), 1))
        target_recall = float((selected_mask & target).sum() / max(target.sum(), 1))
        enrichment = target_precision / max(float(target.mean()), 1e-12)
        sm = selected_mask.reshape(16, 16)
        boundary = 0
        for r in range(16):
            for c in range(16):
                if not sm[r, c]:
                    continue
                boundary += int(r == 0 or not sm[r - 1, c])
                boundary += int(r == 15 or not sm[r + 1, c])
                boundary += int(c == 0 or not sm[r, c - 1])
                boundary += int(c == 15 or not sm[r, c + 1])
        row = {
            "task": meta["task"], "seed": int(meta["seed"]),
            "step": int(meta["step"]), "phase_quartile": int(meta["phase_quartile"]),
            "episode_success": bool(meta["episode_success"]), "m_matched": int(meta["m_matched"]),
            "target_precision": target_precision, "target_recall": target_recall,
            "target_enrichment": enrichment,
            "selected_component_count_4n": component_count(sm.reshape(-1)),
            "selected_boundary_edges_4n": boundary,
            "source_visible_patches": int((groups.get("source", np.zeros(256)) > 0).sum()),
            "target_visible_patches": int((groups.get("target", np.zeros(256)) > 0).sum()),
            "source_target_union_patches": int((groups.get("target_union", overlap) > 0).sum()),
            "initial_state_sha256": meta["initial_state_sha256"],
            "initial_rgb_sha256": meta["initial_rgb_sha256"],
            "state_rgb_sha256": meta["state_rgb_sha256"],
        }
        row.update(reg)
        rows.append(row)
    if not rows:
        raise SystemExit("no state library metadata found")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "state_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"protocol_id": "OPENVLA_L11_ABC_A1_A2_B1_V1", "state_count": len(rows),
              "expected_state_count": 1600, "bootstrap_replicates": args.bootstrap,
              "occlusion_ratio": "not measured: simulator provides visible instance masks but no amodal mask",
              "task_results": {}, "aggregate": {}}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task]
        rho, lo, hi = boot_spearman(rs, "m_matched", "gt_visible_patches", args.bootstrap, rng)
        episode_corrs = []
        for seed in sorted({r["seed"] for r in rs}):
            er = sorted([r for r in rs if r["seed"] == seed], key=lambda x: x["phase_quartile"])
            if len(er) >= 3:
                v = spearmanr([x["m_matched"] for x in er], [x["gt_visible_patches"] for x in er]).statistic
                if np.isfinite(v): episode_corrs.append(float(v))
        report["task_results"][task] = {
            "states": len(rs), "episodes": len({r['seed'] for r in rs}),
            "spearman_m_vs_target_patch_count": rho, "episode_bootstrap_95ci": [lo, hi],
            "median_within_episode_spearman": float(np.median(episode_corrs)) if episode_corrs else None,
            "mean_m": float(np.mean([r['m_matched'] for r in rs])),
            "mean_target_patches": float(np.mean([r['gt_visible_patches'] for r in rs])),
            "mean_target_precision": float(np.mean([r['target_precision'] for r in rs])),
            "mean_target_recall": float(np.mean([r['target_recall'] for r in rs])),
            "mean_target_enrichment": float(np.mean([r['target_enrichment'] for r in rs])),
        }
    rho, lo, hi = boot_spearman(rows, "m_matched", "gt_visible_patches", args.bootstrap, rng)
    all_episode_corrs = []
    for task, seed in sorted({(r["task"], r["seed"]) for r in rows}):
        er = sorted([r for r in rows if r["task"] == task and r["seed"] == seed],
                    key=lambda x: x["phase_quartile"])
        if len(er) >= 3:
            value = spearmanr([x["m_matched"] for x in er],
                              [x["gt_visible_patches"] for x in er]).statistic
            if np.isfinite(value):
                all_episode_corrs.append(float(value))
    report["aggregate"] = {
        "spearman_m_vs_target_patch_count": rho,
        "episode_bootstrap_95ci": [lo, hi],
        "partial_rank_correlation_control_task_and_phase": partial_rank_corr(rows),
        "median_within_episode_spearman": float(np.median(all_episode_corrs)) if all_episode_corrs else None,
    }
    (args.out / "analysis_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"state_count": len(rows), "expected": 1600,
                      "tasks": {k: v["states"] for k, v in report["task_results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
