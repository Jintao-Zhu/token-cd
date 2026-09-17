"""Final paired analysis for DTP-Fixed-positive/L11-negative lambda sweep."""
from __future__ import annotations

import json
import math
import pickle

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.dtp_l11_cd_protocol import (
    ARMS, ARM_TO_LAMBDA, ARTIFACT, CANONICAL, MATCHED_ROOT, SEEDS, TASKS, atomic_json,
)

DTP_ROOT = ARTIFACT.parent / "dtp_openvla_calibration_v1/formal_fixed400/episodes"


def exact_p(rescue: int, harm: int) -> float:
    n = rescue + harm
    if not n:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(rescue, harm) + 1)) / 2**n)


def main() -> None:
    result = {"complete": True, "new_episodes": 1200, "tasks": {}}
    baseline_totals = {"vanilla_success": 0, "dtp_fixed_success": 0, "l11_matched_success": 0}
    pooled = {arm: {"n": 0, "success": 0, "vs_dtp_rescue": 0, "vs_dtp_harm": 0,
                    "vs_l11_rescue": 0, "vs_l11_harm": 0} for arm in ARMS}
    for task in TASKS:
        task_result = {"n": 100, "arms": {}}
        vanilla_success = dtp_success = l11_success = 0
        rows = []
        for seed in SEEDS:
            with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                expected = snapshot_sha(pickle.load(handle))
            vanilla = json.loads((CANONICAL / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json").read_text())
            dtp = json.loads((DTP_ROOT / task / "v2_fix_k64_t05" / f"episode_{seed:03d}_summary.json").read_text())
            l11 = json.loads((MATCHED_ROOT[task] / f"episode_{seed:03d}_summary.json").read_text())
            if any(x.get("canonical_snapshot_sha256") != expected for x in (vanilla, dtp, l11)):
                raise RuntimeError(f"baseline snapshot mismatch: {task} seed={seed}")
            current = {}
            for arm in ARMS:
                path = ARTIFACT / "closed_loop/episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                item = json.loads(path.read_text())
                if item.get("technical_pass") is not True or item.get("canonical_snapshot_sha256") != expected:
                    raise RuntimeError(f"new-arm audit failure: {path}")
                current[arm] = bool(item["success"])
            v, d, l = bool(vanilla["success"]), bool(dtp["success"]), bool(l11["success"])
            vanilla_success += v; dtp_success += d; l11_success += l
            rows.append((v, d, l, current))
        task_result.update({"vanilla_success": vanilla_success, "dtp_fixed_success": dtp_success,
                            "l11_matched_success": l11_success})
        baseline_totals["vanilla_success"] += vanilla_success
        baseline_totals["dtp_fixed_success"] += dtp_success
        baseline_totals["l11_matched_success"] += l11_success
        for arm in ARMS:
            success = sum(row[3][arm] for row in rows)
            dr = sum(row[3][arm] and not row[1] for row in rows)
            dh = sum(row[1] and not row[3][arm] for row in rows)
            lr = sum(row[3][arm] and not row[2] for row in rows)
            lh = sum(row[2] and not row[3][arm] for row in rows)
            arm_result = {
                "lambda": ARM_TO_LAMBDA[arm], "success": success,
                "vs_dtp": {"rescue": dr, "harm": dh, "net": dr-dh, "p_exact": exact_p(dr, dh)},
                "vs_l11": {"rescue": lr, "harm": lh, "net": lr-lh, "p_exact": exact_p(lr, lh)},
            }
            task_result["arms"][arm] = arm_result
            p = pooled[arm]; p["n"] += 100; p["success"] += success
            p["vs_dtp_rescue"] += dr; p["vs_dtp_harm"] += dh
            p["vs_l11_rescue"] += lr; p["vs_l11_harm"] += lh
        result["tasks"][task] = task_result
    result["overall"] = {"baselines": {"n": 400, **baseline_totals}}
    for arm, p in pooled.items():
        p["lambda"] = ARM_TO_LAMBDA[arm]
        p["vs_dtp_net"] = p["vs_dtp_rescue"] - p["vs_dtp_harm"]
        p["vs_l11_net"] = p["vs_l11_rescue"] - p["vs_l11_harm"]
        p["vs_dtp_p_exact"] = exact_p(p["vs_dtp_rescue"], p["vs_dtp_harm"])
        p["vs_l11_p_exact"] = exact_p(p["vs_l11_rescue"], p["vs_l11_harm"])
        result["overall"][arm] = p
    atomic_json(ARTIFACT / "FINAL_RESULTS.json", result)
    lines = ["# DTP-Fixed positive + L11-Matched negative lambda sweep", "",
             "| Task | Vanilla | DTP-Fixed | L11-Matched | lambda=.10 | lambda=.25 | lambda=.50 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        x = result["tasks"][task]
        values = [x["arms"][arm]["success"] for arm in ARMS]
        lines.append(f"| {task.removeprefix('google_robot_')} | {x['vanilla_success']}/100 | {x['dtp_fixed_success']}/100 | {x['l11_matched_success']}/100 | " + " | ".join(f"{v}/100" for v in values) + " |")
    values = [result["overall"][arm]["success"] for arm in ARMS]
    lines.append(
        f"| overall | {baseline_totals['vanilla_success']}/400 | "
        f"{baseline_totals['dtp_fixed_success']}/400 | {baseline_totals['l11_matched_success']}/400 | "
        + " | ".join(f"{v}/400" for v in values) + " |"
    )
    lines += ["", "## Overall paired transitions", "",
              "| lambda | vs DTP rescue/harm/net | vs L11 rescue/harm/net |",
              "|---:|---:|---:|"]
    for arm in ARMS:
        x = result["overall"][arm]
        lines.append(f"| {x['lambda']:.2f} | {x['vs_dtp_rescue']}/{x['vs_dtp_harm']}/{x['vs_dtp_net']:+d} | {x['vs_l11_rescue']}/{x['vs_l11_harm']}/{x['vs_l11_net']:+d} |")
    (ARTIFACT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
