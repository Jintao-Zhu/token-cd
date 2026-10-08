#!/usr/bin/env python3
"""Paired success analysis for the locked L11 Ranking Value Study."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = ROOT / "artifacts/l11_ranking_value_simpler_6task_0_99_v1"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
    "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel",
)
ARMS = ("top32", "mid32", "bottom32", "random32", "top96_random32")
CANON = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
RT0 = ROOT / "artifacts/simpler_rt_depth0_7task_4200_v1"
DEPTH8 = {"google_robot_open_drawer", "google_robot_close_drawer"}


def exact_binomial_two_sided(b: int, h: int) -> float:
    n = b + h
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, h) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired(a: list[bool], b: list[bool]) -> dict:
    rescue = sum((not x) and y for x, y in zip(a, b))
    harm = sum(x and (not y) for x, y in zip(a, b))
    return {
        "first_only_success": rescue,
        "second_only_success": harm,
        "net_second_minus_first": rescue - harm,
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(rescue, harm),
    }


def baseline_root(task: str) -> Path:
    return CANON if task in DEPTH8 else RT0


def main() -> None:
    rows = {}
    completeness = {}
    for task in TASKS:
        base_rows = []
        for seed in range(100):
            path = baseline_root(task) / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json"
            baseline = json.loads(path.read_text())
            if baseline.get("canonical_snapshot_sha256") is None or baseline.get("initial_rgb_sha256") is None:
                raise RuntimeError(f"baseline identity metadata missing: {path}")
            base_rows.append(bool(baseline["success"]))
        rows[task] = {"vanilla": base_rows}
        for arm in ARMS:
            values = []
            missing = []
            for seed in range(100):
                path = ART / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                array_path = ART / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"
                if not path.is_file() or not array_path.is_file():
                    missing.append(seed)
                    continue
                summary = json.loads(path.read_text())
                if summary.get("protocol_id") != "L11_RANKING_VALUE_SIMPLER_6TASK_5ARM_SEEDS0_99_V1":
                    raise RuntimeError(f"protocol id mismatch: {path}")
                if summary.get("renderer_gpu") not in (1, 2, 3) or summary.get("renderer_gpu") != summary.get("inference_gpu"):
                    raise RuntimeError(f"unexpected GPU assignment: {path}")
                if summary.get("K") != 32 or summary.get("lambda") != 0.5 or summary.get("beta") != 0.0:
                    raise RuntimeError(f"configuration mismatch: {path}")
                values.append(bool(summary["success"]))
            completeness[f"{task}/{arm}"] = {"n": len(values), "missing_seeds": missing}
            if missing:
                continue
            rows[task][arm] = values

    complete = all(value["n"] == 100 for value in completeness.values())
    result = {
        "protocol_id": "L11_RANKING_VALUE_SIMPLER_6TASK_5ARM_SEEDS0_99_V1",
        "complete": complete,
        "completeness": completeness,
        "primary_arm_vs_vanilla": {},
        "primary_arm_vs_arm": {},
        "task_results": {},
    }
    if not complete:
        result["status"] = "INCOMPLETE"
        (ART / "analysis").mkdir(parents=True, exist_ok=True)
        (ART / "analysis" / "PAIRED_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return

    for task in TASKS:
        base = rows[task]["vanilla"]
        task_result = {"vanilla_successes": sum(base), "vanilla_rate": sum(base) / 100,
                       "arms": {}, "arm_pairwise": {}}
        for arm in ARMS:
            arm_rows = rows[task][arm]
            p = paired(base, arm_rows)
            doses = []
            for seed in range(100):
                summary = json.loads((ART / "episodes" / task / arm / f"episode_{seed:03d}_summary.json").read_text())
                doses.append(summary["mean_reconstruction_dose"])
            task_result["arms"][arm] = {
                "successes": sum(arm_rows), "rate": sum(arm_rows) / 100,
                "rescue": p["first_only_success"], "harm": p["second_only_success"],
                "net_rescue_minus_harm": p["net_second_minus_first"],
                "mcnemar_exact_two_sided_p": p["mcnemar_exact_two_sided_p"],
                "mean_episode_step_dose": sum(doses) / len(doses),
            }
        for arm in ARMS:
            task_result["arm_pairwise"][f"top32_vs_{arm}"] = paired(rows[task][arm], rows[task]["top32"])
        result["task_results"][task] = task_result

    for arm in ARMS:
        vanilla_all = [value for task in TASKS for value in rows[task]["vanilla"]]
        arm_all = [value for task in TASKS for value in rows[task][arm]]
        result["primary_arm_vs_vanilla"][arm] = paired(vanilla_all, arm_all)
        result["primary_arm_vs_vanilla"][arm]["vanilla_successes"] = sum(vanilla_all)
        result["primary_arm_vs_vanilla"][arm]["arm_successes"] = sum(arm_all)
        result["primary_arm_vs_vanilla"][arm]["vanilla_rate"] = sum(vanilla_all) / len(vanilla_all)
        result["primary_arm_vs_vanilla"][arm]["arm_rate"] = sum(arm_all) / len(arm_all)
    for arm in ("random32", "bottom32", "top96_random32", "mid32"):
        first = [value for task in TASKS for value in rows[task][arm]]
        second = [value for task in TASKS for value in rows[task]["top32"]]
        result["primary_arm_vs_arm"][f"{arm}_vs_top32"] = paired(first, second)
        result["primary_arm_vs_arm"][f"{arm}_vs_top32"]["{arm}_successes"] = sum(first)
        result["primary_arm_vs_arm"][f"{arm}_vs_top32"]["top32_successes"] = sum(second)
        result["primary_arm_vs_arm"][f"{arm}_vs_top32"]["{arm}_rate"] = sum(first) / len(first)
        result["primary_arm_vs_arm"][f"{arm}_vs_top32"]["top32_rate"] = sum(second) / len(second)
    result["macro_success_rate"] = {
        "vanilla": sum(sum(rows[t]["vanilla"]) / 100 for t in TASKS) / len(TASKS),
        **{arm: sum(sum(rows[t][arm]) / 100 for t in TASKS) / len(TASKS) for arm in ARMS},
    }
    result["status"] = "COMPLETE"
    analysis = ART / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "PAIRED_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# L11 Ranking Value Study — Paired Results", "", "Status: COMPLETE", "",
             "All values use seed-paired historical Vanilla; each intervention arm has n=100 per task.", "",
             "## Success by task", "",
             "| Task | Vanilla | Top32 | Mid32 | Bottom32 | Random32 | Top96-Random32 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        tr = result["task_results"][task]
        lines.append("| " + task + " | " + f"{tr['vanilla_successes']}/100" + " | " + " | ".join(
            f"{tr['arms'][arm]['successes']}/100" for arm in ARMS
        ) + " |")
    lines += ["", "## Pooled paired comparisons", "", "| Comparison | First success | Second success | First-only | Second-only | Net (second-first) | Exact McNemar p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        r = result["primary_arm_vs_vanilla"][arm]
        lines.append(f"| Vanilla vs {arm} | {r['vanilla_successes']}/600 | {r['arm_successes']}/600 | {r['first_only_success']} | {r['second_only_success']} | {r['net_second_minus_first']} | {r['mcnemar_exact_two_sided_p']:.4g} |")
    for name, r in result["primary_arm_vs_arm"].items():
        lines.append(f"| {name} | {r['{arm}_successes']}/600 | {r['top32_successes']}/600 | {r['first_only_success']} | {r['second_only_success']} | {r['net_second_minus_first']} | {r['mcnemar_exact_two_sided_p']:.4g} |")
    lines += ["", "## Macro success rate", "", "| Arm | Six-task macro |", "|---|---:|"]
    for arm, rate in result["macro_success_rate"].items():
        lines.append(f"| {arm} | {rate:.1%} |")
    (analysis / "PAIRED_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "macro": result["macro_success_rate"],
                      "pooled": result["primary_arm_vs_vanilla"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
