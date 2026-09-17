"""Analyze four-task L11 Adaptive-lambda development and confirmation results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from scipy.stats import binomtest

from research.semantic_token_cd.prompt_attn_l11_adaptive_lambda_rollout import TASKS


BASELINE_CSV = Path(
    "/home/leju-suzhou/zjt_ws/token-cd/artifacts/"
    "prompt_attn_l11_matched_full_9task_0_299_v1/paired_results_0_299.csv"
)


def load_baselines() -> dict[tuple[str, int], dict]:
    rows = {}
    with BASELINE_CSV.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["task"], int(row["seed"]))
            rows[key] = {
                "shr": row["shr"] == "True",
                "l11_fixed_05": row["l11_matched"] == "True",
            }
    return rows


def exact_p(rescue: int, harm: int) -> float:
    discordant = rescue + harm
    return float(binomtest(min(rescue, harm), discordant, 0.5).pvalue) if discordant else 1.0


def summarize(rows: list[dict], arm: str) -> dict:
    rescue = sum(row[arm] and not row["l11_fixed_05"] for row in rows)
    harm = sum(row["l11_fixed_05"] and not row[arm] for row in rows)
    vs_shr_rescue = sum(row[arm] and not row["shr"] for row in rows)
    vs_shr_harm = sum(row["shr"] and not row[arm] for row in rows)
    l11_rescues = [row for row in rows if row["l11_fixed_05"] and not row["shr"]]
    l11_harms = [row for row in rows if row["shr"] and not row["l11_fixed_05"]]
    return {
        "n": len(rows),
        "successes": sum(row[arm] for row in rows),
        "success_rate": sum(row[arm] for row in rows) / len(rows),
        "vs_l11_fixed_05": {
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "exact_mcnemar_p": exact_p(rescue, harm),
        },
        "vs_shr": {
            "rescue": vs_shr_rescue,
            "harm": vs_shr_harm,
            "net": vs_shr_rescue - vs_shr_harm,
            "exact_mcnemar_p": exact_p(vs_shr_rescue, vs_shr_harm),
        },
        "l11_rescue_preservation": {
            "preserved": sum(row[arm] for row in l11_rescues),
            "total": len(l11_rescues),
            "rate": sum(row[arm] for row in l11_rescues) / len(l11_rescues) if l11_rescues else None,
        },
        "l11_harm_repair": {
            "repaired": sum(row[arm] for row in l11_harms),
            "total": len(l11_harms),
            "rate": sum(row[arm] for row in l11_harms) / len(l11_harms) if l11_harms else None,
        },
        "mean_episode_lambda": sum(row[f"{arm}_mean_lambda"] for row in rows) / len(rows),
    }


def collect(artifact: Path, seeds: range, arms: tuple[str, ...]) -> list[dict]:
    baselines = load_baselines()
    rows = []
    for task in TASKS:
        for seed in seeds:
            baseline = baselines[(task, seed)]
            row = {"task": task, "seed": seed, **baseline}
            hashes = set()
            for arm in arms:
                path = artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                data = json.loads(path.read_text())
                if not data.get("technical_pass", False):
                    raise RuntimeError(f"technical audit failed: {path}")
                row[arm] = bool(data["success"])
                row[f"{arm}_mean_lambda"] = float(data["mean_final_lambda"])
                hashes.add((
                    data["canonical_snapshot_sha256"],
                    data["initial_state_sha256"],
                    data["initial_rgb_sha256"],
                ))
            if len(hashes) != 1:
                raise RuntimeError(f"adaptive arm hash mismatch: {task} seed={seed}")
            rows.append(row)
    return rows


def development_selection(artifact: Path) -> dict:
    candidates = ("l11_adaptive_raw", "l11_adaptive_ema")
    rows = collect(artifact, range(100), ("l11_fixed_025", *candidates))
    results = {}
    for arm in ("l11_fixed_025", *candidates):
        results[arm] = {
            "overall": summarize(rows, arm),
            "by_task": {
                task: summarize([row for row in rows if row["task"] == task], arm)
                for task in TASKS
            },
        }
    ranked = []
    positive_tasks = {"google_robot_close_drawer", "google_robot_pick_coke_can"}
    for arm in candidates:
        positive_rows = [row for row in rows if row["task"] in positive_tasks]
        positive = summarize(positive_rows, arm)
        preservation = positive["l11_rescue_preservation"]["rate"]
        overall = results[arm]["overall"]
        ranked.append((
            preservation is not None and preservation >= 0.9,
            overall["vs_l11_fixed_05"]["net"],
            overall["l11_harm_repair"]["repaired"],
            arm == "l11_adaptive_ema",
            arm,
        ))
    ranked.sort(reverse=True)
    selected = ranked[0][-1]
    output = {
        "protocol_id": "PROMPT_ATTN_L11_ADAPTIVE_LAMBDA_4TASK_V1",
        "development_seeds": [0, 99],
        "selection_rule": "preservation>=0.9, then paired net vs L11-F05, harm repairs, EMA tie-break",
        "selected_arm": selected,
        "selected_constraint_pass": ranked[0][0],
        "results": results,
    }
    path = artifact / "DEVELOPMENT_SELECTION.json"
    if path.exists() and json.loads(path.read_text()) != output:
        raise RuntimeError("development selection lock differs")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return output


def final_analysis(artifact: Path, selected_arm: str) -> dict:
    arms = ("l11_fixed_025", selected_arm)
    rows = collect(artifact, range(100, 300), arms)
    output = {
        "protocol_id": "PROMPT_ATTN_L11_ADAPTIVE_LAMBDA_4TASK_V1",
        "complete": True,
        "confirmation_seeds": [100, 299],
        "selected_arm": selected_arm,
        "overall": {arm: summarize(rows, arm) for arm in arms},
        "by_task": {
            task: {
                arm: summarize([row for row in rows if row["task"] == task], arm)
                for arm in arms
            }
            for task in TASKS
        },
    }
    (artifact / "FINAL_RESULTS_100_299.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    selected = output["overall"][selected_arm]
    fixed = output["overall"]["l11_fixed_025"]
    lines = [
        "# L11 Adaptive-lambda four-task confirmation",
        "",
        f"Selected development arm: `{selected_arm}`.",
        "",
        "| Arm | Success | vs L11-F05 net | exact p | Rescue preservation | Harm repair | Mean lambda |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, summary in ((selected_arm, selected), ("l11_fixed_025", fixed)):
        preservation = summary["l11_rescue_preservation"]
        repair = summary["l11_harm_repair"]
        lines.append(
            f"| {arm} | {summary['successes']}/{summary['n']} ({100 * summary['success_rate']:.1f}%) | "
            f"{summary['vs_l11_fixed_05']['net']:+d} | {summary['vs_l11_fixed_05']['exact_mcnemar_p']:.6g} | "
            f"{preservation['preserved']}/{preservation['total']} | "
            f"{repair['repaired']}/{repair['total']} | {summary['mean_episode_lambda']:.4f} |"
        )
    lines += ["", "## Selected Adaptive arm by task", "", "| Task | Success | vs L11-F05 net | vs SHR net | Mean lambda |", "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        summary = output["by_task"][task][selected_arm]
        lines.append(
            f"| {task} | {summary['successes']}/{summary['n']} ({100 * summary['success_rate']:.1f}%) | "
            f"{summary['vs_l11_fixed_05']['net']:+d} | {summary['vs_shr']['net']:+d} | "
            f"{summary['mean_episode_lambda']:.4f} |"
        )
    (artifact / "FINAL_REPORT_100_299.md").write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--select-development", action="store_true")
    parser.add_argument("--selected-arm")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if args.select_development:
        print(json.dumps(development_selection(artifact), indent=2))
        return
    if not args.selected_arm:
        selection = json.loads((artifact / "DEVELOPMENT_SELECTION.json").read_text())
        args.selected_arm = selection["selected_arm"]
    result = final_analysis(artifact, args.selected_arm)
    print(json.dumps({"complete": True, "selected_arm": args.selected_arm, "overall": result["overall"][args.selected_arm]}, indent=2))


if __name__ == "__main__":
    main()
