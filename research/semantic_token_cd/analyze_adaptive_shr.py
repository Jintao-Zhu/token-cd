"""Analyze the 2700 exactly paired Adaptive-SHR v1 episodes."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from research.semantic_token_cd.semantic_recon_rollout import TASKS


REFERENCE_ARMS = ("vanilla", "semantic_recon_k8_m10", "shr_harmonic")
ADAPTIVE_ARM = "adaptive_shr"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def rate(n: int, d: int) -> float:
    return n / d if d else 0.0


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    counts = {
        arm: sum(bool(row[f"{arm}_success"]) for row in rows)
        for arm in (*REFERENCE_ARMS, ADAPTIVE_ARM)
    }
    adaptive_rescue = sum(not row["shr_harmonic_success"] and row["adaptive_shr_success"] for row in rows)
    adaptive_harm = sum(row["shr_harmonic_success"] and not row["adaptive_shr_success"] for row in rows)
    triggered = [row for row in rows if row["adapt_trigger"]]
    triggered_rescue = sum(not row["shr_harmonic_success"] and row["adaptive_shr_success"] for row in triggered)
    triggered_harm = sum(row["shr_harmonic_success"] and not row["adaptive_shr_success"] for row in triggered)
    vanilla_shr_rescue = sum(not row["vanilla_success"] and row["shr_harmonic_success"] for row in rows)
    vanilla_shr_harm = sum(row["vanilla_success"] and not row["shr_harmonic_success"] for row in rows)
    vanilla_adaptive_rescue = sum(not row["vanilla_success"] and row["adaptive_shr_success"] for row in rows)
    vanilla_adaptive_harm = sum(row["vanilla_success"] and not row["adaptive_shr_success"] for row in rows)
    return {
        "n": n,
        "success": {
            arm: {"count": counts[arm], "rate": rate(counts[arm], n)}
            for arm in counts
        },
        "adaptive_vs_shr": {
            "rescue_shr_fail_adaptive_success": adaptive_rescue,
            "harm_shr_success_adaptive_fail": adaptive_harm,
            "net": adaptive_rescue - adaptive_harm,
        },
        "lambda_episode_selection": {
            "lambda_0_5": n - len(triggered),
            "lambda_0_25": len(triggered),
            "trigger_rate": rate(len(triggered), n),
        },
        "triggered_subset": {
            "n": len(triggered),
            "shr_success": sum(row["shr_harmonic_success"] for row in triggered),
            "adaptive_success": sum(row["adaptive_shr_success"] for row in triggered),
            "rescue": triggered_rescue,
            "harm": triggered_harm,
            "net": triggered_rescue - triggered_harm,
        },
        "rescue_harm_vs_vanilla": {
            "shr": {"rescue": vanilla_shr_rescue, "harm": vanilla_shr_harm},
            "adaptive_shr": {"rescue": vanilla_adaptive_rescue, "harm": vanilla_adaptive_harm},
        },
        "control_steps": {
            "total": sum(row["control_steps"] for row in rows),
            "triggered": sum(row["triggered_control_steps"] for row in rows),
        },
    }


def markdown_report(overall: dict, by_task: dict[str, dict]) -> str:
    success = overall["success"]
    pair = overall["adaptive_vs_shr"]
    trigger = overall["lambda_episode_selection"]
    subset = overall["triggered_subset"]
    lines = [
        "# Adaptive-SHR v1 report",
        "",
        "## Method",
        "",
        "The negative branch is the locked SHR implementation: semantic KMeans K=8 selection and beta=0 four-neighbor harmonic reconstruction. At every control step, the candidate SHR action uses lambda=0.5. The shift ratio is the fraction of changed tokens over action dimensions 0..5 (gripper excluded). If S > 0.33, the final action is recomputed with lambda=0.25; otherwise lambda=0.5 is retained.",
        "",
        "All Adaptive-SHR episodes restore the exact prior Vanilla/Recon/SHR canonical snapshot and verify snapshot, initial-state, and initial-RGB hashes.",
        "",
        "## Overall success",
        "",
        "| Arm | Success | Rate |",
        "|---|---:|---:|",
    ]
    for arm in ("vanilla", "semantic_recon_k8_m10", "shr_harmonic", "adaptive_shr"):
        item = success[arm]
        lines.append(f"| {arm} | {item['count']}/{overall['n']} | {item['rate']:.3%} |")
    lines += [
        "",
        "## Adaptive-SHR vs SHR paired transitions",
        "",
        f"- Rescue (SHR fail, Adaptive success): {pair['rescue_shr_fail_adaptive_success']}",
        f"- Harm (SHR success, Adaptive fail): {pair['harm_shr_success_adaptive_fail']}",
        f"- Net paired gain: {pair['net']:+d}",
        "",
        "## Lambda triggering",
        "",
        f"- Episode lambda=0.5 (no step triggered): {trigger['lambda_0_5']}",
        f"- Episode lambda=0.25 (at least one step triggered): {trigger['lambda_0_25']}",
        f"- Episode trigger rate: {trigger['trigger_rate']:.3%}",
        f"- Triggered subset: SHR {subset['shr_success']}/{subset['n']}, Adaptive {subset['adaptive_success']}/{subset['n']}, rescue {subset['rescue']}, harm {subset['harm']}, net {subset['net']:+d}",
        "",
        "## Task-level results",
        "",
        "| Task | N | SHR | Adaptive | Δ pp | Rescue | Harm | Triggered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = by_task[task]
        shr = item["success"]["shr_harmonic"]["rate"]
        adaptive = item["success"]["adaptive_shr"]["rate"]
        p = item["adaptive_vs_shr"]
        tr = item["lambda_episode_selection"]
        lines.append(
            f"| {task} | {item['n']} | {shr:.3%} | {adaptive:.3%} | "
            f"{100 * (adaptive - shr):+.2f} | {p['rescue_shr_fail_adaptive_success']} | "
            f"{p['harm_shr_success_adaptive_fail']} | {tr['lambda_0_25']} |"
        )
    lines += [
        "",
        "Episode-level lambda selection means whether any control step triggered. Exact step-level lambda, shift ratio, and all three token actions remain in each episode JSON.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    rows: list[dict] = []

    for task in TASKS:
        for seed in range(300):
            adaptive_path = artifact / "episode_json" / task / ADAPTIVE_ARM / f"episode_{seed:03d}_summary.json"
            if not adaptive_path.exists():
                if args.allow_partial:
                    continue
                raise FileNotFoundError(adaptive_path)
            adaptive = load_json(adaptive_path)
            references = {
                arm: load_json(source / "episodes" / task / arm / f"episode_{seed:03d}_summary.json")
                for arm in REFERENCE_ARMS
            }
            all_summaries = [adaptive, *references.values()]
            for field in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
                if len({item.get(field) for item in all_summaries}) != 1:
                    raise RuntimeError(f"pairing mismatch {field}: {task} seed={seed}")
            if not adaptive.get("technical_pass", False):
                raise RuntimeError(f"technical audit failed: {adaptive_path}")
            rows.append({
                "task": task,
                "seed": seed,
                **{f"{arm}_success": bool(summary["success"]) for arm, summary in references.items()},
                "adaptive_shr_success": bool(adaptive["success"]),
                "adapt_trigger": bool(adaptive["adapt_trigger"]),
                "final_lambda": float(adaptive["final_lambda"]),
                "shift_ratio": float(adaptive["shift_ratio"]),
                "control_steps": int(adaptive["control_steps"]),
                "triggered_control_steps": int(adaptive["triggered_control_steps"]),
                "canonical_snapshot_sha256": adaptive["canonical_snapshot_sha256"],
            })

    if not rows:
        raise RuntimeError("no Adaptive-SHR results found")
    if not args.allow_partial and len(rows) != 2700:
        raise RuntimeError(f"expected 2700 pairs, found {len(rows)}")

    paired_dir = artifact / "paired_results"
    stats_dir = artifact / "statistics"
    paired_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    with (paired_dir / "paired_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (paired_dir / "paired_episodes.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    overall = summarize(rows)
    by_task = {task: summarize([row for row in rows if row["task"] == task]) for task in TASKS}
    output = {"complete": len(rows) == 2700, "overall": overall, "by_task": by_task}
    tmp = stats_dir / f".summary.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, stats_dir / "summary.json")
    report = markdown_report(overall, by_task)
    (artifact / "report.md").write_text(report)
    print(json.dumps({"pairs": len(rows), "complete": len(rows) == 2700, **overall["adaptive_vs_shr"]}))


if __name__ == "__main__":
    main()
