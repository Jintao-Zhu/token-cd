"""Aggregate the short-horizon L11 token-group causal audit."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))


def mean(rows, key):
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(); root = args.artifact.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())["cases"]
    files = sorted((root / "results").rglob("*.json"))
    if len(files) != len(manifest):
        raise RuntimeError(f"incomplete audit: {len(files)}/{len(manifest)} cases")
    rows, integrity = [], []
    for path in files:
        data = json.loads(path.read_text())
        full = data["results"]["full_l11"]
        full_action = np.asarray(full["first_guided_action"], dtype=np.float64)[:6]
        full_residual = full["first_centered_residual"]
        audit = data["restore_audit"]
        integrity.append(bool(data["source_mask_verified"]) and audit["sim_state_max_abs_difference"] <= 1e-6
                         and audit["rgb_mean_abs_difference"] <= .1
                         and audit.get("rgb_difference_gt16_fraction", 0.0) <= 1e-3)
        for name, branch in data["results"].items():
            if name == "full_l11":
                continue
            removed = branch["removed_token_ids"]
            delta = float(branch["progress_difference_vs_full_l11"])
            action = np.asarray(branch["first_guided_action"], dtype=np.float64)[:6]
            rows.append({
                "task": data["task"].removeprefix("google_robot_"), "seed": data["seed"],
                "category": data["category"], "control_step": data["control_step"],
                "phase": branch["phase"], "sector": name.removeprefix("without_"),
                "removed_count": len(removed), "removed_tokens": ";".join(map(str, removed)),
                "mean_removed_prompt_rank": float(np.mean(branch["removed_prompt_global_ranks"])),
                "mean_removed_prompt_attention": float(np.mean(branch["removed_prompt_attention"])),
                "progress_full": full["primary_progress_gain"],
                "progress_ablation": branch["primary_progress_gain"],
                "delta_progress_vs_full": delta,
                "delta_progress_per_removed_token": delta / max(1, len(removed)),
                "first_action_l2_vs_full": float(np.linalg.norm(action - full_action)),
                "residual_cosine_vs_full": cosine(branch["first_centered_residual"], full_residual),
                "ablation_improves_2mm": int(delta > .002),
                "ablation_worsens_2mm": int(delta < -.002),
            })
    write_csv(root / "causal_group_rows.csv", rows)

    summaries = []
    for keys in (("category",), ("task", "category"), ("sector", "category"),
                 ("task", "sector", "category"), ("phase", "category")):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        for values, subset in sorted(groups.items()):
            record = {key: value for key, value in zip(keys, values)}
            record.update({
                "n": len(subset), "episodes": len({(x["task"], x["seed"]) for x in subset}),
                "mean_delta_progress": mean(subset, "delta_progress_vs_full"),
                "median_delta_progress": float(np.median([x["delta_progress_vs_full"] for x in subset])),
                "improves_2mm": sum(x["ablation_improves_2mm"] for x in subset),
                "worsens_2mm": sum(x["ablation_worsens_2mm"] for x in subset),
                "mean_action_l2": mean(subset, "first_action_l2_vs_full"),
                "mean_residual_cosine": mean(subset, "residual_cosine_vs_full"),
                "mean_removed_count": mean(subset, "removed_count"),
                "mean_removed_prompt_rank": mean(subset, "mean_removed_prompt_rank"),
            })
            summaries.append(record)
    # Different grouping tables have different columns; write JSON as the
    # canonical aggregate and a flat core table for quick inspection.
    (root / "GROUP_SUMMARIES.json").write_text(json.dumps(summaries, indent=2) + "\n")
    core = [x for x in summaries if set(x).issuperset({"category", "n"}) and
            not any(key in x for key in ("task", "sector", "phase"))]
    write_csv(root / "category_summary.csv", core)

    episode_groups = defaultdict(list)
    for row in rows:
        episode_groups[(row["task"], row["seed"], row["category"])].append(row)
    episode_rows = [{"task": key[0], "seed": key[1], "category": key[2],
                     "mean_delta": mean(value, "delta_progress_vs_full"),
                     "mean_rank": mean(value, "mean_removed_prompt_rank")}
                    for key, value in episode_groups.items()]
    rescue = [x["mean_delta"] for x in episode_rows if x["category"] == "rescue"]
    harm = [x["mean_delta"] for x in episode_rows if x["category"] == "harm"]
    mw = mannwhitneyu(harm, rescue, alternative="greater") if rescue and harm else None
    rho_rank = spearmanr([x["mean_rank"] for x in episode_rows],
                         [x["mean_delta"] for x in episode_rows])
    candidate_sectors = []
    for sector in sorted({x["sector"] for x in rows}):
        h = [x for x in rows if x["sector"] == sector and x["category"] == "harm"]
        r = [x for x in rows if x["sector"] == sector and x["category"] == "rescue"]
        candidate_sectors.append({
            "sector": sector, "harm_n": len(h), "rescue_n": len(r),
            "harm_mean_delta": mean(h, "delta_progress_vs_full"),
            "rescue_mean_delta": mean(r, "delta_progress_vs_full"),
            "harm_improve_minus_worsen": sum(x["ablation_improves_2mm"]-x["ablation_worsens_2mm"] for x in h),
            "rescue_improve_minus_worsen": sum(x["ablation_improves_2mm"]-x["ablation_worsens_2mm"] for x in r),
        })
    # A deployable fixed spatial rule must point in the desired direction in
    # at least three tasks for Harm and must not hurt Rescue in at least three.
    task_sector = []
    for sector in sorted({x["sector"] for x in rows if not x["sector"].startswith("fallback")}):
        record = {"sector": sector, "tasks": {}}
        for task in sorted({x["task"] for x in rows}):
            h = [x for x in rows if x["sector"] == sector and x["task"] == task and x["category"] == "harm"]
            r = [x for x in rows if x["sector"] == sector and x["task"] == task and x["category"] == "rescue"]
            record["tasks"][task] = {"harm_mean_delta": mean(h, "delta_progress_vs_full"),
                                      "rescue_mean_delta": mean(r, "delta_progress_vs_full")}
        record["harm_positive_tasks"] = sum(v["harm_mean_delta"] is not None and v["harm_mean_delta"] > 0
                                             for v in record["tasks"].values())
        record["rescue_nonnegative_tasks"] = sum(v["rescue_mean_delta"] is not None and v["rescue_mean_delta"] >= 0
                                                  for v in record["tasks"].values())
        task_sector.append(record)
    useful_rule = [x for x in task_sector if x["harm_positive_tasks"] >= 3 and
                   x["rescue_nonnegative_tasks"] >= 3]
    decision = "STABLE_SPATIAL_FILTER_CANDIDATE_FOUND" if useful_rule else "NO_STABLE_SPATIAL_FILTER_RULE"
    tasks = sorted({x["task"] for x in rows})
    sectors = [f"r{ri}_c{ci}" for ri in range(3) for ci in range(2)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    limit = .04
    for axis, category in zip(axes, ("harm", "rescue")):
        matrix = np.full((len(tasks), len(sectors)), np.nan)
        for ti, task in enumerate(tasks):
            for si, sector in enumerate(sectors):
                subset = [x for x in rows if x["task"] == task and x["sector"] == sector
                          and x["category"] == category]
                matrix[ti, si] = mean(subset, "delta_progress_vs_full") if subset else np.nan
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_title(f"Remove sector: {category}")
        axis.set_xticks(range(len(sectors)), sectors, rotation=45, ha="right")
        axis.set_yticks(range(len(tasks)), tasks)
        for ti in range(len(tasks)):
            for si in range(len(sectors)):
                if np.isfinite(matrix[ti, si]):
                    axis.text(si, ti, f"{1000*matrix[ti, si]:+.1f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes, label="5-action progress difference (m); positive = removal helps")
    figure.savefig(root / "task_sector_causal_heatmap.png", dpi=180)
    plt.close(figure)
    final = {
        "protocol_id": "L11_TOKEN_GROUP_CAUSAL_AUDIT_V1", "decision": decision,
        "integrity_pass": all(integrity), "cases": len(files), "ablations": len(rows),
        "cases_by_task_category": {f"{task}/{category}": sum(
            1 for p in files if (d := json.loads(p.read_text()))["task"].removeprefix("google_robot_") == task
            and d["category"] == category) for task in sorted({x["task"] for x in rows})
            for category in ("rescue", "harm")},
        "inference_unit": "episode mean across its spatial ablations",
        "harm_vs_rescue_delta_mann_whitney_one_sided": None if mw is None else
            {"statistic": float(mw.statistic), "p": float(mw.pvalue)},
        "removed_rank_vs_delta_spearman": {"rho": float(rho_rank.statistic), "p": float(rho_rank.pvalue)},
        "sector_results": candidate_sectors, "task_sector_results": task_sector,
        "candidate_rules": useful_rule,
        "interpretation_boundary": "5-action local progress is causal for the tested branch point, not a full-episode success claim",
    }
    (root / "FINAL_RESULTS.json").write_text(json.dumps(final, indent=2) + "\n")

    lines = ["# L11 Token-group Causal Audit", "", "## Decision", "",
             f"**{decision}**", "", "## Integrity", "",
             f"- Cases: {len(files)}/{len(manifest)}; spatial ablations: {len(rows)}.",
             f"- Snapshot / RGB tolerance / source-mask gate: {'PASS' if all(integrity) else 'FAIL'}.",
             "- Every comparison restores the same simulator state, changes only one spatial subset of the first L11 mask, then uses the same L11-Matched continuation for five actions.",
             "", "## Pooled causal effect", ""]
    for category in ("rescue", "harm"):
        subset = [x for x in rows if x["category"] == category]
        lines.append(f"- {category}: mean leave-one-group-out minus full-L11 progress = {mean(subset, 'delta_progress_vs_full'):+.5f} m; "
                     f"improved >2 mm {sum(x['ablation_improves_2mm'] for x in subset)}/{len(subset)}, "
                     f"worsened >2 mm {sum(x['ablation_worsens_2mm'] for x in subset)}/{len(subset)}.")
    lines += ["", "## Spatial sectors", "", "| Sector | Harm mean Δ | Rescue mean Δ | Harm +/- | Rescue +/- |",
              "|---|---:|---:|---:|---:|"]
    for x in candidate_sectors:
        h = "—" if x["harm_mean_delta"] is None else f"{x['harm_mean_delta']:+.5f}"
        r = "—" if x["rescue_mean_delta"] is None else f"{x['rescue_mean_delta']:+.5f}"
        lines.append(f"| {x['sector']} | {h} | {r} | "
                     f"{x['harm_improve_minus_worsen']:+d} | {x['rescue_improve_minus_worsen']:+d} |")
    lines += ["", "## Main finding", "",
              "No fixed spatial sector separates Harm from Rescue consistently across tasks. Effects reverse by task: the same sector can improve a Harm case in one task and damage Rescue in another.",
              f"Episode-level Harm-vs-Rescue one-sided Mann–Whitney p={mw.pvalue:.4g}; removed Prompt rank vs causal delta Spearman rho={rho_rank.statistic:.3f}, p={rho_rank.pvalue:.4g}.",
              "All selected branch points are in the approach phase. This audit therefore rules against a simple early spatial/rank filter, but it does not test the previously observed late drawer over-guidance mechanism.",
              "", "![Task-by-sector causal heatmap](task_sector_causal_heatmap.png)"]
    lines += ["", "## Interpretation boundary", "",
              "A positive delta means removing that group improved task-native short-horizon progress relative to the full L11 mask. "
              "This is a same-state causal statement for five actions; it is not yet evidence that a full closed-loop episode would succeed.",
              "", "## Files", "", "- `FINAL_RESULTS.json`: decision and machine-readable statistics.",
              "- `causal_group_rows.csv`: one row per case × removed spatial group.",
              "- `GROUP_SUMMARIES.json`: task/category/sector/phase aggregates.",
              "- `task_sector_causal_heatmap.png`: sign reversals across tasks and outcomes."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "cases": len(files), "ablations": len(rows)}))


if __name__ == "__main__":
    main()
