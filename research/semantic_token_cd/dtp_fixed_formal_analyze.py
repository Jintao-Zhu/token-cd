"""Analyze the 400-episode fixed-mask DTP formal run against current-harness Vanilla."""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import binomtest


REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
ROOT = REPO / "artifacts/dtp_openvla_calibration_v1/formal_fixed400"
VANILLA = REPO / "artifacts/vla_pruner_openvla_reproduction/formal_1200/episodes"
ARM = "v2_fix_k64_t05"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
HASH_KEYS = ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    result = {"arm": ARM, "tasks": {}, "issues": []}
    total = {"n": 0, "vanilla_success": 0, "fixed_success": 0,
             "rescue": 0, "harm": 0, "trigger_steps": 0,
             "control_steps": 0, "pruned_tokens": 0}
    for task in TASKS:
        rows = []
        arm_dir = ROOT / "episodes" / task / ARM
        for seed in range(100):
            fp = arm_dir / f"episode_{seed:03d}_summary.json"
            vp = VANILLA / task / "vanilla" / f"episode_{seed:03d}_summary.json"
            if not fp.exists():
                continue
            fixed, vanilla = load(fp), load(vp)
            if not fixed.get("technical_pass", False):
                result["issues"].append(f"{task}:{seed}: technical_pass=false")
            for key in HASH_KEYS:
                if fixed.get(key) != vanilla.get(key):
                    result["issues"].append(f"{task}:{seed}: {key} mismatch")
            rows.append((fixed, vanilla))
        n = len(rows)
        fs = sum(bool(f["success"]) for f, _ in rows)
        vs = sum(bool(v["success"]) for _, v in rows)
        rescue = sum(bool(f["success"]) and not bool(v["success"]) for f, v in rows)
        harm = sum(not bool(f["success"]) and bool(v["success"]) for f, v in rows)
        trigger = sum(int(f.get("prune_activation_steps", 0)) for f, _ in rows)
        steps = sum(int(f.get("control_steps", 0)) for f, _ in rows)
        pruned = sum(int(f.get("total_pruned_tokens", 0)) for f, _ in rows)
        p = float(binomtest(min(rescue, harm), rescue + harm, 0.5).pvalue) if rescue + harm else 1.0
        result["tasks"][task] = {
            "n": n, "vanilla_success": vs, "fixed_success": fs,
            "rescue": rescue, "harm": harm, "net": rescue - harm,
            "mcnemar_exact_p": p,
            "trigger_steps": trigger, "control_steps": steps,
            "trigger_rate": trigger / steps if steps else 0.0,
            "pruned_tokens": pruned,
        }
        for key, value in (("n", n), ("vanilla_success", vs), ("fixed_success", fs),
                           ("rescue", rescue), ("harm", harm),
                           ("trigger_steps", trigger), ("control_steps", steps),
                           ("pruned_tokens", pruned)):
            total[key] += value
    total["net"] = total["rescue"] - total["harm"]
    total["mcnemar_exact_p"] = (
        float(binomtest(min(total["rescue"], total["harm"]),
                        total["rescue"] + total["harm"], 0.5).pvalue)
        if total["rescue"] + total["harm"] else 1.0
    )
    total["trigger_rate"] = total["trigger_steps"] / total["control_steps"] if total["control_steps"] else 0.0
    result["overall"] = total
    result["complete"] = total["n"] == 400 and not result["issues"]
    atomic_write(ROOT / "FINAL_RESULTS.json", json.dumps(result, indent=2, sort_keys=True) + "\n")

    def cell(x: int, n: int) -> str:
        return f"{x}/{n} ({100*x/n:.1f}%)" if n else "—"

    lines = [
        "# DTP Fixed-mask L11/k64/tau0.5 formal 400", "",
        f"Completed: {total['n']}/400; issues={len(result['issues'])}", "",
        "| Task | N | Current Vanilla | DTP-Fixed | Delta | Rescue | Harm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        d = result["tasks"][task]
        lines.append(
            f"| {task} | {d['n']} | {cell(d['vanilla_success'], d['n'])} | "
            f"{cell(d['fixed_success'], d['n'])} | {d['fixed_success']-d['vanilla_success']:+d} | "
            f"{d['rescue']} | {d['harm']} |"
        )
    lines += [
        f"| OVERALL | {total['n']} | {cell(total['vanilla_success'], total['n'])} | "
        f"{cell(total['fixed_success'], total['n'])} | "
        f"{total['fixed_success']-total['vanilla_success']:+d} | {total['rescue']} | {total['harm']} |",
        "", f"Overall exact paired p={total['mcnemar_exact_p']:.4g}; "
        f"trigger rate={100*total['trigger_rate']:.2f}%; pruned tokens={total['pruned_tokens']}.",
    ]
    atomic_write(ROOT / "FINAL_REPORT.md", "\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
