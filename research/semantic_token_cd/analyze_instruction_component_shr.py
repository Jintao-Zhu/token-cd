"""Final paired and component analysis for IC-SHR v1."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = (
    "google_robot_close_drawer", "google_robot_open_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, required=True)
    args = ap.parse_args()
    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    rows = []
    by_task = {}
    for task in TASKS:
        records = []
        for seed in range(300):
            ic_path = artifact / "episode_summary" / task / "ic_shr" / f"episode_{seed:03d}_summary.json"
            if not ic_path.exists():
                raise RuntimeError(f"missing IC-SHR result: {task} seed={seed}")
            ic = json.loads(ic_path.read_text())
            refs = {}
            for arm in ("vanilla", "semantic_recon_k8_m10", "shr_harmonic"):
                refs[arm] = json.loads((source / "episodes" / task / arm /
                                        f"episode_{seed:03d}_summary.json").read_text())
            record = {
                "task": task, "seed": seed,
                "vanilla_success": bool(refs["vanilla"]["success"]),
                "recon_success": bool(refs["semantic_recon_k8_m10"]["success"]),
                "shr_success": bool(refs["shr_harmonic"]["success"]),
                "ic_shr_success": bool(ic["success"]),
                "mean_num_components": ic["mean_num_components"],
                "mean_selected_num_components": ic["mean_selected_num_components"],
                "mean_filtered_num_components": ic["mean_filtered_num_components"],
                "mean_mask_tokens_before": ic["mean_mask_tokens_before"],
                "mean_mask_tokens_after": ic["mean_mask_tokens_after"],
                "deleted_token_fraction": ic["deleted_token_fraction"],
                "technical_pass": bool(ic["technical_pass"]),
                "canonical_snapshot_sha256": ic["canonical_snapshot_sha256"],
            }
            rows.append(record)
            records.append(record)
        def success(key):
            count = sum(r[key] for r in records)
            return {"count": count, "rate": count / len(records)}
        rescue = sum(r["ic_shr_success"] and not r["shr_success"] for r in records)
        harm = sum(not r["ic_shr_success"] and r["shr_success"] for r in records)
        by_task[task] = {
            "n": len(records),
            "success": {key: success(key) for key in (
                "vanilla_success", "recon_success", "shr_success", "ic_shr_success")},
            "ic_vs_shr": {"rescue": rescue, "harm": harm, "net": rescue - harm},
            "components": {
                "mean_num_components": sum(r["mean_num_components"] for r in records) / len(records),
                "mean_selected_num_components": sum(r["mean_selected_num_components"] for r in records) / len(records),
                "mean_filtered_num_components": sum(r["mean_filtered_num_components"] for r in records) / len(records),
                "mean_mask_tokens_before": sum(r["mean_mask_tokens_before"] for r in records) / len(records),
                "mean_mask_tokens_after": sum(r["mean_mask_tokens_after"] for r in records) / len(records),
                "mean_episode_deleted_token_fraction": sum(r["deleted_token_fraction"] for r in records) / len(records),
                "episodes_with_filtering": sum(r["deleted_token_fraction"] > 0 for r in records),
            },
            "technical_pass": sum(r["technical_pass"] for r in records),
        }
    overall = {
        "n": len(rows),
        "success": {key: {"count": sum(r[key] for r in rows), "rate": sum(r[key] for r in rows) / len(rows)}
                    for key in ("vanilla_success", "recon_success", "shr_success", "ic_shr_success")},
        "ic_vs_shr": {
            "rescue": sum(r["ic_shr_success"] and not r["shr_success"] for r in rows),
            "harm": sum(not r["ic_shr_success"] and r["shr_success"] for r in rows),
        },
        "technical_pass": sum(r["technical_pass"] for r in rows),
    }
    overall["ic_vs_shr"]["net"] = overall["ic_vs_shr"]["rescue"] - overall["ic_vs_shr"]["harm"]
    stats = {"complete": len(rows) == 1200, "overall": overall, "by_task": by_task}
    stats_dir = artifact / "component_stats"
    paired_dir = artifact / "paired_results"
    stats_dir.mkdir(exist_ok=True); paired_dir.mkdir(exist_ok=True)
    (stats_dir / "summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    with (paired_dir / "paired_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# IC-SHR v1 report", "", "All 1200 episodes reuse the released canonical snapshots.", "",
             "## Success and paired result", "",
             "| Task | Vanilla | Recon | SHR | IC-SHR | Delta pp | Rescue | Harm | Net |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task, stat in by_task.items():
        s = stat["success"]; pair = stat["ic_vs_shr"]
        delta = 100 * (s["ic_shr_success"]["rate"] - s["shr_success"]["rate"])
        lines.append(f"| {task.replace('google_robot_', '')} | {s['vanilla_success']['rate']:.3%} | "
                     f"{s['recon_success']['rate']:.3%} | {s['shr_success']['rate']:.3%} | "
                     f"{s['ic_shr_success']['rate']:.3%} | {delta:+.2f} | {pair['rescue']} | "
                     f"{pair['harm']} | {pair['net']:+d} |")
    lines.extend(["", "## Component diagnostics", "",
                  "| Task | Mean components | Mean selected | Mean filtered | Tokens before | Tokens after | Deleted | Filtered episodes |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for task, stat in by_task.items():
        c = stat["components"]
        lines.append(f"| {task.replace('google_robot_', '')} | {c['mean_num_components']:.2f} | "
                     f"{c['mean_selected_num_components']:.2f} | {c['mean_filtered_num_components']:.2f} | "
                     f"{c['mean_mask_tokens_before']:.1f} | {c['mean_mask_tokens_after']:.1f} | "
                     f"{c['mean_episode_deleted_token_fraction']:.2%} | {c['episodes_with_filtering']}/300 |")
    (artifact / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "episodes": len(rows), **overall["ic_vs_shr"]}), flush=True)


if __name__ == "__main__":
    main()
