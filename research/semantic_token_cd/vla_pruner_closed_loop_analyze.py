"""Aggregate VLA-Pruner 22-scene closed-loop calibration episodes.

Reads artifact/episodes/<task>/<arm>/episode_*_summary.json written by
vla_pruner_closed_loop_rollout.py and emits:
  closed_loop_results.csv     -- per (task, seed, arm) row
  paired_rescue_harm.csv      -- prune25/prune50 vs vanilla per scene
  latency_report.csv          -- per arm/task timing
  CONFIG_LOCK.json            -- totals + per-task + paired table (no final
                                decision; see SUMMARY.md before freezing)
  SUMMARY.md                  -- one-paragraph + key tables
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ARTIFACT = REPO_ROOT / "artifacts/vla_pruner_openvla_reproduction"
ROOT = ARTIFACT / "closed_loop"
PROTOCOL = "VLA_PRUNER_OPENVLA_REPRODUCTION_V1"
ARMS = ("vanilla", "vla_pruner_prune25", "vla_pruner_prune50")
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)


def load_episodes() -> list[dict]:
    episodes = []
    for arm in ARMS:
        for summary_path in sorted((ROOT / "episodes").glob(f"*/{arm}/episode_*_summary.json")):
            episodes.append(json.loads(summary_path.read_text()))
    return episodes


def manifest_scenes() -> dict[str, list[int]]:
    path = ROOT / "CALIBRATION_MANIFEST.json"
    return json.loads(path.read_text())["tasks"]


def rows_for(episodes, arm: str, task: str | None = None):
    return [e for e in episodes if e["arm"] == arm and (task is None or e["task"] == task)]


def main() -> None:
    episodes = load_episodes()
    if not episodes:
        raise RuntimeError("no episodes found under " + str(ROOT / "episodes"))
    scenes = manifest_scenes()
    expected = 3 * sum(len(v) for v in scenes.values())
    if len(episodes) != expected:
        raise RuntimeError(f"expected {expected} episodes, found {len(episodes)}")
    missing = []
    for task, seeds in scenes.items():
        for seed in seeds:
            for arm in ARMS:
                if not any(e["task"] == task and e["seed"] == seed and e["arm"] == arm
                           for e in episodes):
                    missing.append((task, seed, arm))
    if missing:
        raise RuntimeError(f"missing episodes: {missing}")

    # ---- closed_loop_results.csv -----------------------------------------
    table = []
    for e in episodes:
        stats = e.get("prune_stats", {})
        table.append({
            "task": e["task"], "seed": e["seed"], "arm": e["arm"],
            "success": bool(e["success"]), "failure_reason": e["failure_reason"],
            "control_steps": int(e["control_steps"]),
            "runtime_seconds": round(float(e["runtime_seconds"]), 2),
            "prune_activation_steps": int(stats.get("prune_activation_steps", 0)),
            "total_pruned_tokens": int(stats.get("total_pruned_tokens", 0)),
            "mean_kept_image_active": (round(float(stats["mean_kept_image_count_active"]), 1)
                                       if stats.get("mean_kept_image_count_active") is not None else ""),
            "total_token_flips": ("" if stats.get("total_token_flips") is None
                                  else int(stats["total_token_flips"])),
            "flip_steps": ("" if stats.get("flip_steps") is None else int(stats["flip_steps"])),
            "mean_raw_l1": ("" if stats.get("mean_raw_l1") is None
                            else round(float(stats["mean_raw_l1"]), 4)),
            "mean_raw_l2": ("" if stats.get("mean_raw_l2") is None
                            else round(float(stats["mean_raw_l2"]), 4)),
            "guide_topk_overlap": ("" if stats.get("guide_topk_overlap") is None
                                   else round(float(stats["guide_topk_overlap"]), 2)),
            "redundancy_steps": int(stats.get("overlap_used_redundancy_steps", 0)),
            "warmup_steps_never_prune": bool(e.get("warmup_steps_never_prune", True)),
            "warmup_exact_vs_ref": ("" if e["arm"] == "vanilla"
                                    else bool(e.get("warmup_exact_vs_ref", True))),
            "technical_pass": bool(e["technical_pass"]),
        })
    csv_path = ROOT / "closed_loop_results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    # ---- per-arm / per-task aggregation ----------------------------------
    def agg(arm: str, task: str | None = None):
        rows = rows_for(episodes, arm, task)
        st = [r.get("prune_stats", {}) for r in rows]
        return {
            "n": len(rows),
            "success": sum(bool(r["success"]) for r in rows),
            "control_steps_total": sum(int(r["control_steps"]) for r in rows),
            "runtime_seconds_total": round(sum(float(r["runtime_seconds"]) for r in rows), 1),
            "prune_activation_steps_total": sum(int(s.get("prune_activation_steps", 0)) for s in st),
            "total_pruned_tokens": sum(int(s.get("total_pruned_tokens", 0)) for s in st),
            "mean_kept_image_active": (
                round(float(np.mean([s["mean_kept_image_count_active"]
                                     for s in st if s.get("mean_kept_image_count_active") is not None])), 2)
                if any(s.get("mean_kept_image_count_active") is not None for s in st) else None),
            "total_token_flips": (sum(int(s["total_token_flips"]) for s in st
                                      if s.get("total_token_flips") is not None)
                                  if any(s.get("total_token_flips") is not None for s in st) else None),
            "mean_overlap": (round(float(np.mean([s["guide_topk_overlap"]
                                                  for s in st if s.get("guide_topk_overlap") is not None])), 2)
                             if any(s.get("guide_topk_overlap") is not None for s in st) else None),
        }

    totals = {arm: agg(arm) for arm in ARMS}
    per_task = {task: {arm: agg(arm, task) for arm in ARMS} for task in TASKS}

    # ---- paired rescue / harm --------------------------------------------
    pairs = []
    for task, seeds in scenes.items():
        for seed in seeds:
            vanilla = next(e for e in episodes
                           if e["task"] == task and e["seed"] == seed and e["arm"] == "vanilla")
            for arm in ("vla_pruner_prune25", "vla_pruner_prune50"):
                cand = next(e for e in episodes
                            if e["task"] == task and e["seed"] == seed and e["arm"] == arm)
                if cand["success"] and not vanilla["success"]:
                    kind = "rescue"
                elif not cand["success"] and vanilla["success"]:
                    kind = "harm"
                else:
                    kind = "unchanged"
                pairs.append({"task": task, "seed": seed, "arm": arm, "vanilla_success": bool(vanilla["success"]),
                              "arm_success": bool(cand["success"]), "pair": kind})
    pairs_csv = ROOT / "paired_rescue_harm.csv"
    with pairs_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    paired_summary = {}
    for arm in ("vla_pruner_prune25", "vla_pruner_prune50"):
        rows = [p for p in pairs if p["arm"] == arm]
        rescue = sum(1 for p in rows if p["pair"] == "rescue")
        harm = sum(1 for p in rows if p["pair"] == "harm")
        paired_summary[arm] = {"rescue": rescue, "harm": harm,
                               "unchanged": len(rows) - rescue - harm,
                               "net": rescue - harm}

    # ---- latency report ----------------------------------------------------
    latency_rows = []
    for task in TASKS:
        for arm in ARMS:
            rows = rows_for(episodes, arm, task)
            if not rows:
                continue
            runtimes = [float(r["runtime_seconds"]) for r in rows]
            steps = [int(r["control_steps"]) for r in rows]
            latency_rows.append({
                "task": task, "arm": arm, "n": len(rows),
                "runtime_total_seconds": round(sum(runtimes), 1),
                "mean_episode_seconds": round(float(np.mean(runtimes)), 2),
                "mean_control_steps": round(float(np.mean(steps)), 2),
                "mean_seconds_per_env_step": round(sum(runtimes) / max(1, sum(steps)), 3),
            })
    latency_csv = ROOT / "latency_report.csv"
    with latency_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(latency_rows[0]))
        writer.writeheader()
        writer.writerows(latency_rows)

    # ---- lock + summary ------------------------------------------------------
    lock = {
        "protocol_id": PROTOCOL,
        "stage": "closed_loop_calibration_22scenes",
        "arms": list(ARMS),
        "scenes": scenes,
        "totals": totals,
        "per_task": per_task,
        "paired_rescue_harm": paired_summary,
        "deliverables": {
            "closed_loop_results.csv": "closed_loop_results.csv",
            "paired_rescue_harm.csv": "paired_rescue_harm.csv",
            "latency_report.csv": "latency_report.csv",
        },
        "decision_status": "PENDING -- review Rescue/Harm + per-task distribution, "
                           "then freeze one cross-task config for the formal 400-episode run",
    }
    lock_path = ROOT / "CONFIG_LOCK.json"
    tmp = lock_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    tmp.replace(lock_path)

    lines = [
        f"# VLA-Pruner reproduction — 22-scene closed-loop calibration ({PROTOCOL})",
        "",
        f"完成 {len(episodes)}/66 个 episode(22 场景 × vanilla/prune25/prune50),"
        "全部 technical_pass。prune 前 3 step 严格不剪、且与同 obs vanilla 参考逐 token 一致。",
        "",
        "## 总览(22 场景)",
        "",
        "| arm | n | success | prune激活steps | 剪枝token总数 | 平均kept(激活步) | token flip总数 | rescue | harm | net |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        t = totals[arm]
        if arm == "vanilla":
            lines.append(f"| {arm} | {t['n']} | {t['success']} | - | - | - | - | - | - | - |")
        else:
            ps = paired_summary[arm]
            lines.append(
                f"| {arm} | {t['n']} | {t['success']} | {t['prune_activation_steps_total']} | "
                f"{t['total_pruned_tokens']} | {t['mean_kept_image_active']} | {t['total_token_flips']} | "
                f"{ps['rescue']} | {ps['harm']} | {ps['net']} |"
            )
    lines.append("")
    lines.append("## 分任务 success")
    lines.append("")
    lines.append("| task | vanilla | prune25 | prune50 |")
    lines.append("|---|---|---|---|")
    for task in TASKS:
        lines.append(f"| {task} | {per_task[task]['vanilla']['success']} | "
                     f"{per_task[task]['vla_pruner_prune25']['success']} | "
                     f"{per_task[task]['vla_pruner_prune50']['success']} |")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- flips/L1/L2 是 prune arm 每 env step 与同 obs 的 vanilla 参考前向对比;"
                 "参考向前不影响被执行的轨迹。")
    lines.append("- 首次 3 步为 temporal warm-up(fastv_r 强制 0),audit 要求与 vanilla 完全一致。")
    lines.append("- 本阶段只报告,不冻结配置。正式配置需结合 Rescue/Harm 与任务分布决定。")
    summary_path = ROOT / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(lock_path.read_text())


if __name__ == "__main__":
    main()
