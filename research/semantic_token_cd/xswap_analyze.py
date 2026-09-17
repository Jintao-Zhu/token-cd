"""Aggregate XSWAP-V1 closed-loop results into per-task/pooled tables.

Units: physical scenes (deduped seeds).  A scene is usable only when all five
arms have summaries.  Primary comparisons (pre-specified): Correct vs Swapped
and Correct vs Paraphrase; Random/Vanilla are interpretive baselines.

Outputs:
  REPORT.md + REPORT.json under artifacts/prompt_attn_instr_swap_v1/
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.semantic_token_cd.xswap_protocol import ARMS, ARTIFACT, TASKS


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return value is not None


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    try:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
    except Exception:
        pass
    return obj


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0 or b == c:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p)


def load_scenes() -> dict[str, list[int]]:
    return json.loads((ARTIFACT / "scene_manifest.json").read_text())["scenes"]


def read_summary(task: str, seed: int, arm: str) -> dict | None:
    path = ARTIFACT / "runs/episodes" / task / arm / f"episode_{seed:03d}_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACT / "REPORT.md")
    args = parser.parse_args()
    scenes = load_scenes()
    report: dict = {"protocol": "PROMPT_ATTN_INSTR_SWAP_V1", "per_task": {},
                    "pooled": None, "coverage": {}, "offline": {}}
    lines: list[str] = []
    add = lines.append

    pair_primary = (("correct", "swapped"), ("correct", "paraphrase"))
    pair_baseline = (("correct", "random"), ("correct", "vanilla"),
                     ("paraphrase", "swapped"))

    all_pairs = [("correct", "swapped"), ("correct", "paraphrase"),
                 ("correct", "random"), ("correct", "vanilla"), ("paraphrase", "swapped")]

    pooled_tables: dict = {}
    for task in TASKS:
        scene_list = scenes[task]
        valid = []
        for seed in scene_list:
            arms = {a: read_summary(task, seed, a) for a in ARMS}
            if all(v is not None for v in arms.values()):
                valid.append((seed, arms))
        n_total = len(scene_list)
        n_valid = len(valid)
        per_arm = {}
        for arm in ARMS:
            ok = [arms[arm]["success"] for _, arms in valid]
            n = len(ok)
            k = sum(ok)
            lo, hi = wilson(k, n) if n else (0.0, 0.0)
            m_vals = [arms[arm]["mean_m_t"] for _, arms in valid if finite(arms[arm].get("mean_m_t"))]
            steps = [arms[arm]["control_steps"] for _, arms in valid]
            per_arm[arm] = {
                "n_scenes": n, "successes": k,
                "success_rate": round(k / n, 4) if n else None,
                "wilson_95": [round(lo, 4), round(hi, 4)] if n else None,
                "mean_m": round(float(np.mean(m_vals)), 2) if m_vals else None,
                "mean_control_steps": round(float(np.mean(steps)), 1) if steps else None,
            }
        pairs = {}
        for a, b in all_pairs:
            n01 = sum(1 for _, arms in valid if arms[a]["success"] and not arms[b]["success"])
            n10 = sum(1 for _, arms in valid if not arms[a]["success"] and arms[b]["success"])
            n00 = sum(1 for _, arms in valid if not arms[a]["success"] and not arms[b]["success"])
            n11 = sum(1 for _, arms in valid if arms[a]["success"] and arms[b]["success"])
            p = mcnemar_p(n01, n10)
            pairs[f"{a}_vs_{b}"] = {
                "n01_a_win_b_lose": n01, "n10_a_lose_b_win": n10,
                "both_success": n11, "both_fail": n00,
                "net_rescue_minus_harm": n01 - n10,
                "mcnemar_p": round(p, 4),
            }
        report["per_task"][task] = {"n_scenes_total": n_total, "n_scenes_valid": n_valid,
                                    "per_arm": per_arm, "pairs": pairs}
        add(f"## {task}")
        add(f"- dedup scenes in manifest: {n_total}; scenes with all five arms: {n_valid}")
        add(f"- success rate (Wilson 95% CI):")
        for arm in ARMS:
            v = per_arm[arm]
            ci = f"[{v['wilson_95'][0]}, {v['wilson_95'][1]}]" if v["wilson_95"] else "-"
            add(f"  - {arm:10s} {v['successes']:3d}/{v['n_scenes']:3d} = {v['success_rate']}  CI {ci}"
                f"  mean_m {v['mean_m']}  mean_steps {v['mean_control_steps']}")
        add("- paired (scene-level), rescue/harm:")
        for a, b in all_pairs:
            v = pairs[f"{a}_vs_{b}"]
            add(f"  - {a:9s} vs {b:9s}: {a}-win={v['n01_a_win_b_lose']}, {a}-lose={v['n10_a_lose_b_win']}, "
                f"net={v['net_rescue_minus_harm']:+d}, McNemar p={v['mcnemar_p']}")

    # pooled over tasks at the scene level
    per_arm_pool = defaultdict(list)
    pair_pool: dict = {}
    valid_scenes = []
    for task in TASKS:
        for seed in scenes[task]:
            arms = {a: read_summary(task, seed, a) for a in ARMS}
            if all(v is not None for v in arms.values()):
                valid_scenes.append((task, seed, arms))
    for arm in ARMS:
        per_arm_pool[arm] = [arms[arm]["success"] for _, _, arms in valid_scenes]
    pooled = {}
    for arm in ARMS:
        ok = per_arm_pool[arm]
        n = len(ok)
        k = sum(ok)
        lo, hi = wilson(k, n)
        m_vals = [arms[arm]["mean_m_t"] for _, _, arms in valid_scenes if finite(arms[arm].get("mean_m_t"))]
        steps = [arms[arm]["control_steps"] for _, _, arms in valid_scenes]
        pooled[f"arm_{arm}"] = {
            "n_scenes": n, "successes": k, "success_rate": round(k / n, 4),
            "wilson_95": [round(lo, 4), round(hi, 4)],
            "mean_m": round(float(np.mean(m_vals)), 2) if m_vals else None,
            "mean_control_steps": round(float(np.mean(steps)), 1),
        }
    for a, b in all_pairs:
        n01 = sum(1 for _, _, arms in valid_scenes if arms[a]["success"] and not arms[b]["success"])
        n10 = sum(1 for _, _, arms in valid_scenes if not arms[a]["success"] and arms[b]["success"])
        n00 = sum(1 for _, _, arms in valid_scenes if not arms[a]["success"] and not arms[b]["success"])
        n11 = sum(1 for _, _, arms in valid_scenes if arms[a]["success"] and arms[b]["success"])
        pair_pool[f"{a}_vs_{b}"] = {
            "n01": n01, "n10": n10, "both_success": n11, "both_fail": n00,
            "net_rescue_minus_harm": n01 - n10, "mcnemar_p": round(mcnemar_p(n01, n10), 4),
        }
    report["pooled"] = {"n_valid_scenes": len(valid_scenes), "per_arm": pooled, "pairs": pair_pool}
    report["coverage"] = {
        "manifest_scenes_per_task": {t: len(s) for t, s in scenes.items()},
        "total_manifest_scenes": sum(len(s) for s in scenes.values()),
        "total_episodes_nominal": sum(len(s) for s in scenes.values()) * len(ARMS),
        "valid_scenes_all_arms": len(valid_scenes),
    }

    add("")
    add("## Pooled (scene-level across the four tasks)")
    add(f"- valid scenes with all five arms: {len(valid_scenes)}")
    for arm in ARMS:
        v = pooled[f"arm_{arm}"]
        add(f"  - {arm:10s} {v['successes']:4d}/{v['n_scenes']:4d} = {v['success_rate']}"
            f"  CI {v['wilson_95']}  mean_m {v['mean_m']}  mean_steps {v['mean_control_steps']}")
    add("- paired (scene-level), rescue/harm:")
    for a, b in all_pairs:
        v = pair_pool[f"{a}_vs_{b}"]
        add(f"  - {a:9s} vs {b:9s}: {a}-win={v['n01']}, {a}-lose={v['n10']}, "
            f"net={v['net_rescue_minus_harm']:+d}, McNemar p={v['mcnemar_p']}")

    # offline same-state summary if present
    for task in TASKS:
        summary_path = ARTIFACT / "runs/offline" / task / "offline_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        report["offline"][task] = summary
    if report["offline"]:
        add("")
        add("## Offline same-state branch checks")
        for task, summary in report["offline"].items():
            add(f"- {task}: states={summary['n_states']}, clean_equal={summary['clean_equal_all_states']}, "
                f"m_equal={summary['m_equal_all_states']}, mean_m={summary['mean_m']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    json_out = args.out.with_suffix(".json")
    json_out.write_text(json.dumps(sanitize(report), indent=1, sort_keys=True) + "\n")
    print(json.dumps({"written": str(args.out), "valid_scenes": len(valid_scenes),
                      "per_arm_pooled": report["pooled"]["per_arm"],
                      "pairs_pooled": report["pooled"]["pairs"]}, indent=1))


if __name__ == "__main__":
    main()
