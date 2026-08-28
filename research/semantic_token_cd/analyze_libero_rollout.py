#!/usr/bin/env python3
"""LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1 — rollout analysis + gates.

Aggregates per-episode summaries (episodes/<task>/<arm>/episode_XXX.json) into
SR, Rescue/Harm, task breakdown, and the four rollout gates:

  A: ΔSR(primary) > +3pp and not single-task-driven
  B: Rescue/Harm > 1 (per-episode vanilla<->cd pairing)
  C: ≥5/10 tasks non-worse
  D: SR(semantic primary) > SR(random control)

Arms: vanilla / entity_cd_025 / entity_cd_050 (primary) / random_mask (control).

Run:
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/analyze_libero_rollout.py \
    --artifact artifacts/libero_object_semantic_entity_cd_phase0_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PRIMARY = "entity_cd_050"
AUX = "entity_cd_025"
CONTROL = "random_mask"
ARMS = ("vanilla", AUX, PRIMARY, CONTROL)
DELTA_PP = 0.03
TASKS_NON_WORSE = 5


def _load(art: Path) -> dict[str, dict[str, dict[int, dict]]]:
    """{(task, episode): {arm: summary}} for episodes with all 4 arms present."""
    data: dict[str, dict[str, dict[int, dict]]] = {}
    for arm_dir in (art / "episodes").glob("*/"):
        task = arm_dir.name
        for arm in ARMS:
            for f in (arm_dir / arm).glob("episode_*.json"):
                ep = int(f.stem.split("_")[1])
                data.setdefault(task, {}).setdefault(ep, {})[arm] = json.loads(f.read_text())
    # keep only complete quadruples
    out = {t: {ep: d for ep, d in eps.items() if all(a in d for a in ARMS)}
           for t, eps in data.items()}
    return {t: eps for t, eps in out.items() if eps}


def sr(summaries: list[dict]) -> float:
    n = len(summaries)
    return sum(1 for d in summaries if d["success"]) / n if n else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    a = p.parse_args()
    art = a.artifact.resolve()
    data = _load(art)
    tasks = sorted(data)

    # ---- per-arm SR ----
    sr_overall = {}
    sr_task = {t: {} for t in tasks}
    for arm in ARMS:
        sr_overall[arm] = sr([d[arm] for t in tasks for ep, d in data[t].items()])
        for t in tasks:
            sr_task[t][arm] = sr([d[arm] for ep, d in data[t].items()])

    # ---- Rescue/Harm (vanilla <-> arm pairing) ----
    def rescue_harm(arm):
        rescue = harm = ff = ss = 0
        for t in tasks:
            for ep, d in data[t].items():
                v, c = d["vanilla"]["success"], d[arm]["success"]
                rescue += (not v) and c
                harm += v and (not c)
                ff += (not v) and (not c)
                ss += v and c
        ratio = (rescue / harm) if harm else float("inf")
        return {"arm": arm, "rescue": rescue, "harm": harm, "ratio": ratio,
                "fail_fail": ff, "succ_succ": ss, "n": rescue + harm + ff + ss}

    rh = {arm: rescue_harm(arm) for arm in (AUX, PRIMARY, CONTROL)}

    # ---- gates (on PRIMARY) ----
    delta = sr_overall[PRIMARY] - sr_overall["vanilla"]
    improving = [t for t in tasks if sr_task[t][PRIMARY] > sr_task[t]["vanilla"]]
    non_worse = [t for t in tasks if sr_task[t][PRIMARY] >= sr_task[t]["vanilla"]]
    gates = {
        "A": {"PASS": delta > DELTA_PP and len(improving) > 1,
              "delta": delta, "threshold": DELTA_PP,
              "improving_tasks": improving, "single_task_driven": len(improving) == 1},
        "B": {"PASS": rh[PRIMARY]["ratio"] > 1.0,
              "ratio": rh[PRIMARY]["ratio"], "rescue": rh[PRIMARY]["rescue"],
              "harm": rh[PRIMARY]["harm"]},
        "C": {"PASS": len(non_worse) >= TASKS_NON_WORSE,
              "non_worse_tasks": len(non_worse), "of": len(tasks),
              "threshold": TASKS_NON_WORSE},
        "D": {"PASS": sr_overall[PRIMARY] > sr_overall[CONTROL],
              "sr_primary": sr_overall[PRIMARY], "sr_random": sr_overall[CONTROL]},
    }
    all_pass = all(g["PASS"] for g in gates.values())
    verdict = "PASS_TO_ROLLOUT" if all_pass else "STOP_LIBERO_OBJECT_SEMANTIC_NO_GO"

    agg = {
        "experiment": "LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1",
        "phase": "rollout",
        "n_tasks": len(tasks),
        "n_episodes_per_arm": sum(len(d) for d in data.values()),
        "SR": sr_overall,
        "task_SR": sr_task,
        "rescue_harm": rh,
        "gates": gates,
        "verdict": verdict,
    }

    # ---- CSVs ----
    with (art / "rollout_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "episode"] + list(ARMS))
        w.writeheader()
        for t in tasks:
            for ep in sorted(data[t]):
                row = {"task": t, "episode": ep}
                for arm in ARMS:
                    row[arm] = int(data[t][ep][arm]["success"])
                w.writerow(row)

    with (art / "rescue_harm.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "rescue", "harm", "ratio", "fail_fail", "succ_succ", "n"])
        w.writeheader()
        for arm in (AUX, PRIMARY, CONTROL):
            w.writerow(rh[arm])

    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task"] + list(ARMS) + ["delta_primary", "delta_aux"])
        w.writeheader()
        for t in tasks:
            row = {"task": t}
            for arm in ARMS:
                row[arm] = round(sr_task[t][arm], 4)
            row["delta_primary"] = round(sr_task[t][PRIMARY] - sr_task[t]["vanilla"], 4)
            row["delta_aux"] = round(sr_task[t][AUX] - sr_task[t]["vanilla"], 4)
            w.writerow(row)

    with (art / "random_control.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "vanilla", PRIMARY, CONTROL, "semantic_minus_random"])
        w.writeheader()
        for t in tasks:
            w.writerow({"task": t, "vanilla": sr_task[t]["vanilla"],
                        PRIMARY: sr_task[t][PRIMARY], CONTROL: sr_task[t][CONTROL],
                        "semantic_minus_random": round(sr_task[t][PRIMARY] - sr_task[t][CONTROL], 4)})

    # ---- decision.json (rollout) ----
    (art / "decision_rollout.json").write_text(json.dumps({
        "experiment": "LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1",
        "phase": "rollout", "status": verdict, "date": "2026-08-24",
        "SR": sr_overall, "delta_SR_primary": delta,
        "rescue_harm_primary": rh[PRIMARY],
        "rescue_harm_aux": rh[AUX],
        "gates": gates, "closed_loop": "RUN",
        "core_question": "semantic entity counterfactual 是否在 LIBERO-Object 产生净闭环收益？",
    }, indent=2, sort_keys=True) + "\n")

    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
