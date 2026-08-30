"""SCR-CD Phase 1 analyzer — 5-arm, 9-task paired rollout.

Reads every episode summary under artifacts/semantic_recon_k8_m10_v1/episodes and
produces the exact metrics the spec requires:

  * per (task, seed-range, arm) success rates (the coarse table the user asked for)
  * paired DR = SR_arm - SR_vanilla, Rescue (0->1), Harm (1->0), Net
  * paired 95% CI (cluster bootstrap over tasks; seed-pair bootstrap per task)
  * exact McNemar (two-sided binomtest on discordant pairs)
  * 9-task macro DR, #tasks non-worse (DR >= 0), worst-task harm min_t DR_t,
    cross-task variance
  * grouped analysis: Direct-grasp / Articulated / Relational
  * E_t / q / cos diagnostics already recorded per-episode, summarized per (arm, task)

Judgment (locked by spec):
  Method GO iff  DR_macro >= +3pp AND >=6/9 tasks non-worse AND min_t DR_t > -10pp
                 AND semantic_recon_k8_m10 > random_recon_k8_m10 (macro DR).
  Science GO (direction replicated under the correct control, not a method win)
  iff  random_recon_k8_m10 > semantic_recon_k8_m10 (macro DR) — i.e. the deletion
  is generic, the semantic selector confers no advantage, CD itself is the driver.

Partial-data safe: only 5-arm-complete (task, seed) pairs enter the paired tables;
a coverage report flags anything short of the full 9x100x5 = 4500.

Run (offline, no CUDA needed):
  <venv>/bin/python research/semantic_token_cd/analyze_semantic_recon.py \
      --artifact artifacts/semantic_recon_k8_m10_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

PROTOCOL = "SCR_CD_SEMANTIC_RECON_K8_M10_V1"
ARMS = (
    "vanilla",
    "semantic_attn_k8_l8_15",
    "semantic_merge_k8_eta100",
    "semantic_recon_k8_m10",
    "random_recon_k8_m10",
)
CD_ARMS = ARMS[1:]
EXPECT_TASKS = 9
EXPECT_SEEDS = 100
EXPECT_EPISODES = 9 * 100 * 5

# Semantic grouping (standard RLAIF-style taxonomy over the 9 tasks).
GROUP = {
    "google_robot_pick_coke_can": "direct_grasp",
    "google_robot_open_drawer": "articulated",
    "google_robot_close_drawer": "articulated",
    "google_robot_place_apple_in_closed_top_drawer": "articulated",
    "google_robot_move_near": "relational",
    "widowx_carrot_on_plate": "relational",
    "widowx_put_eggplant_in_basket": "relational",
    "widowx_spoon_on_towel": "relational",
    "widowx_stack_cube": "relational",
}
GROUP_ORDER = ("direct_grasp", "articulated", "relational")


def load_records(artifact: Path):
    """Return records[(task, arm, seed)] = summary dict."""
    rec: dict[tuple[str, str, int], dict] = {}
    for tf in sorted((artifact / "episodes").glob("*")):
        if not tf.is_dir():
            continue
        task = tf.name
        for af in sorted(tf.glob("*")):
            if not af.is_dir():
                continue
            arm = af.name
            if arm not in ARMS:
                continue
            for sf in sorted(af.glob("episode_*_summary.json")):
                seed = int(sf.name.split("_")[1])
                try:
                    rec[(task, arm, seed)] = json.loads(sf.read_text())
                except json.JSONDecodeError:
                    continue
    return rec


def paired_rows(rec):
    """Rows for every (task, seed) with all 5 arms present + hash-verified."""
    by_pair: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for (task, arm, seed), r in rec.items():
        by_pair[(task, seed)][arm] = r
    rows, hash_fail = [], []
    for (task, seed), arms in sorted(by_pair.items()):
        if len(arms) != len(ARMS):
            continue
        hashes = {a: arms[a].get("canonical_snapshot_sha256") for a in ARMS}
        if len(set(hashes.values())) != 1:
            hash_fail.append((task, seed, hashes))
            continue
        rows.append({
            "task": task,
            "seed": seed,
            "group": GROUP[task],
            "arms": arms,
        })
    return rows, hash_fail


def cluster_boot_macro(rows, arm, n=4000, seed=731902):
    """95% CI on macro DR (mean over tasks of per-task paired DR), cluster = task."""
    rng = np.random.default_rng(seed)
    tasks = sorted({r["task"] for r in rows})
    by = {t: [r for r in rows if r["task"] == t] for t in tasks}
    vals = []
    for _ in range(n):
        chosen = rng.choice(tasks, len(tasks), replace=True)
        d = []
        for t in chosen:
            pool = by[t]
            sample = [pool[i] for i in rng.integers(0, len(pool), len(pool))]
            d.append(np.mean([
                int(r["arms"][arm]["success"]) - int(r["arms"]["vanilla"]["success"])
                for r in sample
            ]))
        vals.append(np.mean(d))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def seed_boot_ci(task_rows, arm, n=4000, seed=731902):
    """95% CI on per-task paired DR (resample seeds within the task)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = [task_rows[i] for i in rng.integers(0, len(task_rows), len(task_rows))]
        vals.append(np.mean([
            int(r["arms"][arm]["success"]) - int(r["arms"]["vanilla"]["success"])
            for r in sample
        ]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def per_arm_block(rec, rows, arm):
    v = np.array([r["arms"]["vanilla"]["success"] for r in rows], int)
    m = np.array([r["arms"][arm]["success"] for r in rows], int)
    d = m - v
    n01 = int(((v == 0) & (m == 1)).sum())   # rescued
    n10 = int(((v == 1) & (m == 0)).sum())   # harmed
    nb = int((d == 0).sum())
    discordant = n01 + n10
    mcnemar = float(binomtest(n01, discordant).pvalue) if discordant else 1.0

    by_task = defaultdict(list)
    for r, dv in zip(rows, d):
        by_task[r["task"]].append(dv)
    task_meta = {
        t: {
            "sr_vanilla": float(np.mean([int(r["arms"]["vanilla"]["success"])
                                         for r in rows if r["task"] == t])),
            "sr_arm": float(np.mean([int(r["arms"][arm]["success"])
                                     for r in rows if r["task"] == t])),
            "delta": float(np.mean([dv for dv in by_task[t]])),
            "rei": int(sum(1 for r in rows if r["task"] == t
                           and r["arms"]["vanilla"]["success"] == 0
                           and r["arms"][arm]["success"] == 1)),
            "harm": int(sum(1 for r in rows if r["task"] == t
                            and r["arms"]["vanilla"]["success"] == 1
                            and r["arms"][arm]["success"] == 0)),
            "ci95": list(seed_boot_ci([r for r in rows if r["task"] == t], arm)),
        }
        for t in sorted(by_task)
    }
    # worst-task harm = min over tasks of delta
    worst_delta = min(tm["delta"] for tm in task_meta.values()) if task_meta else None

    return {
        "sr_vanilla": float(v.mean()),
        "sr_arm": float(m.mean()),
        "delta": float(d.mean()),
        "delta_macro": float(np.mean([task_meta[t]["delta"] for t in task_meta])) if task_meta else 0.0,
        "rescued": n01,
        "harmed": n10,
        "net": n01 - n10,
        "net_rate_vs_discordant": float((n01 - n10) / discordant) if discordant else 0.0,
        "mcnemar_p": mcnemar,
        "ci95_macro": list(cluster_boot_macro(rows, arm)),
        "n_pairs": len(rows),
        "n_tasks_nonworse": int(sum(1 for tm in task_meta.values() if tm["delta"] >= 0)),
        "worst_task_delta": worst_delta,
        "cross_task_var": float(np.var([task_meta[t]["delta"] for t in task_meta])) if task_meta else 0.0,
        "by_task": task_meta,
    }


def group_block(rows, arm):
    out = {}
    for g in GROUP_ORDER:
        gr = [r for r in rows if r["group"] == g]
        if not gr:
            continue
        d = np.array([int(r["arms"][arm]["success"]) - int(r["arms"]["vanilla"]["success"])
                      for r in gr], float)
        out[g] = {
            "n_pairs": len(gr),
            "sr_vanilla": float(np.mean([int(r["arms"]["vanilla"]["success"]) for r in gr])),
            "sr_arm": float(np.mean([int(r["arms"][arm]["success"]) for r in gr])),
            "delta": float(d.mean()),
            "rescued": int(((d > 0).sum())),
            "harmed": int(((d < 0).sum())),
        }
    return out


def diagnostic_block(rec, rows, arm):
    """E_t / q / cos aggregated over episodes (already per-episode means)."""
    keys = ("mean_e_rel", "mean_q_ratio", "mean_cos_v_vhat", "mean_residual_norm", "mean_num_tokens")
    present = {k: [] for k in keys}
    for r in rows:
        s = r["arms"][arm]
        for k in keys:
            if k in s:
                present[k].append(s[k])
    return {
        k: {
            "mean": float(np.mean(present[k])),
            "median": float(np.median(present[k])),
            "p10": float(np.percentile(present[k], 10)),
            "p90": float(np.percentile(present[k], 90)),
        }
        for k in keys if present[k]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    a = p.parse_args()
    art = a.artifact.resolve()

    rec = load_records(art)
    rows, hash_fail = paired_rows(rec)

    n_episodes = len(rec)
    n_complete_pairs = len(rows)
    coverage = {
        "episodes_found": n_episodes,
        "expected_episodes": EXPECT_EPISODES,
        "five_arm_complete_pairs": n_complete_pairs,
        "expected_pairs": EXPECT_TASKS * EXPECT_SEEDS,
        "hash_failures": [{"task": t, "seed": s} for t, s, _ in hash_fail],
        "complete": n_episodes >= EXPECT_EPISODES,
    }

    summary = {"coverage": coverage, "comparisons": {}, "groups": {}, "diagnostics": {}}

    for arm in CD_ARMS:
        summary["comparisons"][arm] = per_arm_block(rec, rows, arm)
        summary["groups"][arm] = group_block(rows, arm)
        if arm in ("semantic_recon_k8_m10", "random_recon_k8_m10"):
            summary["diagnostics"][arm] = diagnostic_block(rec, rows, arm)

    # Macro (9-task) table, and the spec's GO decision.
    srecon = summary["comparisons"]["semantic_recon_k8_m10"]
    rrecon = summary["comparisons"]["random_recon_k8_m10"]
    dr_macro = srecon["delta_macro"]
    n_nonworse = srecon["n_tasks_nonworse"]
    min_delta = srecon["worst_task_delta"]
    recon_beats_random = dr_macro > rrecon["delta_macro"]
    random_beats_recon = rrecon["delta_macro"] > dr_macro

    if dr_macro >= 0.03 and n_nonworse >= 6 and min_delta > -0.10 and recon_beats_random:
        decision = "METHOD_GO"
    elif random_beats_recon and rrecon["delta_macro"] > dr_macro:
        decision = "SCIENCE_GO__CONTROL_BEATS_SELECTOR"
    elif dr_macro < 0.03:
        decision = "NO_METHOD_WIN__MAGNITUDE_BELOW_3PP"
    elif min_delta <= -0.10:
        decision = "NO_METHOD_WIN__WORST_TASK_HARM_EXCEEDS_10PP"
    elif not recon_beats_random:
        decision = "NO_METHOD_WIN__SELECTOR_NOT_BETTER_THAN_RANDOM_CONTROL"
    else:
        decision = "MIXED_OR_INCONCLUSIVE"

    analysis = {
        "protocol": PROTOCOL,
        "coverage": coverage,
        "comparisons": summary["comparisons"],
        "groups": summary["groups"],
        "diagnostics": summary["diagnostics"],
        "decision": {
            "decision": decision,
            "dr_macro": dr_macro,
            "n_tasks_nonworse": n_nonworse,
            "min_task_delta": min_delta,
            "semantic_recon_delta_macro": dr_macro,
            "random_recon_delta_macro": rrecon["delta_macro"],
            "semantic_beats_random": recon_beats_random,
            "criteria": {
                "dr_macro_ge_3pp": dr_macro >= 0.03,
                "n_tasks_nonworse_ge_6": n_nonworse >= 6,
                "min_task_delta_gt_-10pp": min_delta > -0.10,
                "semantic_gt_random": recon_beats_random,
            },
        },
    }

    out = art
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    (out / "decision.json").write_text(json.dumps(
        {"decision": decision, "method_go": decision == "METHOD_GO",
         "dr_macro": dr_macro, "n_tasks_nonworse": n_nonworse,
         "min_task_delta": min_delta}, indent=2, sort_keys=True) + "\n")

    # paired_results.csv: seed x arm success (for downstream / spot-checking).
    fieldnames = ["task", "seed", "group"] + [f"{a}_success" for a in ARMS] + [f"{a}_delta" for a in CD_ARMS]
    with (out / "paired_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {"task": r["task"], "seed": r["seed"], "group": r["group"]}
            for a in ARMS:
                row[f"{a}_success"] = int(r["arms"][a]["success"])
            for a in CD_ARMS:
                row[f"{a}_delta"] = int(r["arms"][a]["success"]) - int(r["arms"]["vanilla"]["success"])
            w.writerow(row)

    # Human-readable coarse table (task / seed-range / arm / SR).
    print(f"\n=== SCR-CD Phase 1  ({n_episodes}/{EXPECT_EPISODES} episodes, "
          f"{n_complete_pairs}/{EXPECT_TASKS * EXPECT_SEEDS} complete 5-arm pairs) ===")
    print(f"decision: {decision}\n")
    seed_lo = min((r["seed"] for r in rows), default=0)
    seed_hi = max((r["seed"] for r in rows), default=0)
    print(f"seed range: {seed_lo}-{seed_hi} | 5 arms: {', '.join(ARMS)}\n")
    for t in sorted({r["task"] for r in rows}):
        tr = [r for r in rows if r["task"] == t]
        line = f"{t:48s} (n={len(tr)}, {GROUP[t]})"
        for a in ARMS:
            sr = np.mean([int(r["arms"][a]["success"]) for r in tr])
            line += f"  {a[:18]:>18s}={sr:5.1%}"
        print(line)
    print()
    for a in CD_ARMS:
        c = summary["comparisons"][a]
        print(f"[{a}] DR={c['delta']:+.3f} (macro {c['delta_macro']:+.3f}, "
              f"CI95 {c['ci95_macro'][0]:+.3f}..{c['ci95_macro'][1]:+.3f})  "
              f"rescue={c['rescued']} harm={c['harmed']} net={c['net']:+d}  "
              f"mcnemar_p={c['mcnemar_p']:.3g}  nonworse={c['n_tasks_nonworse']}/9  "
              f"min_DR={c['worst_task_delta']:+.3f}")
    print(f"\nmethod-GO criteria: DR_macro >= +3pp? {dr_macro >= 0.03} (got {dr_macro:+.3f}) | "
          f">=6 tasks non-worse? {n_nonworse >= 6} (got {n_nonworse}/9) | "
          f"min DR > -10pp? {min_delta > -0.10} (got {min_delta:+.3f}) | "
          f"SemanticRecon > RandomRecon? {recon_beats_random}")
    print(f"decision: {decision}\n")


if __name__ == "__main__":
    main()