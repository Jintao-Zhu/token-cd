"""Final paired report for pure and DTP-positive high-lambda L11 arms."""
from __future__ import annotations

import json
import math
import pickle

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.l11_high_lambda_protocol import (
    ARMS, ARM_CONFIG, ARTIFACT, CANONICAL, MATCHED_ROOT, SEEDS, TASKS, atomic_json,
)

OLD_DTP_COMBO = ARTIFACT.parent / "dtp_fixed_positive_l11_matched_cd_v1/closed_loop/episodes"
DTP_FIXED = ARTIFACT.parent / "dtp_openvla_calibration_v1/formal_fixed400/episodes"


def exact_p(rescue: int, harm: int) -> float:
    n = rescue + harm
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(rescue, harm) + 1)) / 2**n)


def main() -> None:
    result = {"complete": True, "new_episodes": 1600, "tasks": {}}
    totals = {arm: {"n": 0, "success": 0, "rescue_vs_family05": 0, "harm_vs_family05": 0} for arm in ARMS}
    baseline_totals = {"vanilla": 0, "dtp_fixed": 0, "l11_lambda_05": 0, "dtp_l11_lambda_05": 0}
    for task in TASKS:
        rows = []
        for seed in SEEDS:
            with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                expected = snapshot_sha(pickle.load(handle))
            paths = {
                "vanilla": CANONICAL / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json",
                "dtp_fixed": DTP_FIXED / task / "v2_fix_k64_t05" / f"episode_{seed:03d}_summary.json",
                "l11_lambda_05": MATCHED_ROOT[task] / f"episode_{seed:03d}_summary.json",
                "dtp_l11_lambda_05": OLD_DTP_COMBO / task / "dtp_l11_lambda_0p5" / f"episode_{seed:03d}_summary.json",
            }
            base = {key: json.loads(path.read_text()) for key, path in paths.items()}
            if any(value.get("canonical_snapshot_sha256") != expected for value in base.values()):
                raise RuntimeError(f"baseline snapshot mismatch: {task} seed={seed}")
            current = {}
            for arm in ARMS:
                path = ARTIFACT / "closed_loop/episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                item = json.loads(path.read_text())
                if item.get("technical_pass") is not True or item.get("canonical_snapshot_sha256") != expected:
                    raise RuntimeError(f"new arm audit failure: {path}")
                current[arm] = bool(item["success"])
            rows.append(({key: bool(value["success"]) for key, value in base.items()}, current))
        task_result = {"n": 100, "baselines": {}, "arms": {}}
        for name in baseline_totals:
            success = sum(row[0][name] for row in rows)
            task_result["baselines"][name] = success
            baseline_totals[name] += success
        for arm in ARMS:
            family_ref = "l11_lambda_05" if ARM_CONFIG[arm]["family"] == "pure_l11" else "dtp_l11_lambda_05"
            success = sum(row[1][arm] for row in rows)
            rescue = sum(row[1][arm] and not row[0][family_ref] for row in rows)
            harm = sum(row[0][family_ref] and not row[1][arm] for row in rows)
            task_result["arms"][arm] = {"success": success, "lambda": ARM_CONFIG[arm]["lambda"],
                                         "family_reference": family_ref, "rescue": rescue, "harm": harm,
                                         "net": rescue-harm, "p_exact": exact_p(rescue, harm)}
            totals[arm]["n"] += 100; totals[arm]["success"] += success
            totals[arm]["rescue_vs_family05"] += rescue; totals[arm]["harm_vs_family05"] += harm
        result["tasks"][task] = task_result
    result["overall"] = {"baselines": {"n": 400, **baseline_totals}}
    for arm, row in totals.items():
        row["lambda"] = ARM_CONFIG[arm]["lambda"]
        row["family"] = ARM_CONFIG[arm]["family"]
        row["net_vs_family05"] = row["rescue_vs_family05"] - row["harm_vs_family05"]
        row["p_exact_vs_family05"] = exact_p(row["rescue_vs_family05"], row["harm_vs_family05"])
        result["overall"][arm] = row
    atomic_json(ARTIFACT / "FINAL_RESULTS.json", result)
    lines = ["# L11 high-lambda extension", "",
             "| Task | L11 .50 | L11 .60 | L11 .75 | DTP+L11 .50 | DTP+L11 .60 | DTP+L11 .75 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        x = result["tasks"][task]
        lines.append(f"| {task.removeprefix('google_robot_')} | {x['baselines']['l11_lambda_05']}/100 | {x['arms']['l11_matched_lambda_0p6']['success']}/100 | {x['arms']['l11_matched_lambda_0p75']['success']}/100 | {x['baselines']['dtp_l11_lambda_05']}/100 | {x['arms']['dtp_l11_lambda_0p6']['success']}/100 | {x['arms']['dtp_l11_lambda_0p75']['success']}/100 |")
    b = result["overall"]["baselines"]
    lines.append(f"| overall | {b['l11_lambda_05']}/400 | {totals['l11_matched_lambda_0p6']['success']}/400 | {totals['l11_matched_lambda_0p75']['success']}/400 | {b['dtp_l11_lambda_05']}/400 | {totals['dtp_l11_lambda_0p6']['success']}/400 | {totals['dtp_l11_lambda_0p75']['success']}/400 |")
    lines += ["", "## Paired against each family's lambda=.50", "",
              "| Arm | Rescue | Harm | Net | exact p |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        x = result["overall"][arm]
        lines.append(f"| {arm} | {x['rescue_vs_family05']} | {x['harm_vs_family05']} | {x['net_vs_family05']:+d} | {x['p_exact_vs_family05']:.5g} |")
    (ARTIFACT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

