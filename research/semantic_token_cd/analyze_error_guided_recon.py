"""Error-Guided Recon analyzer — 4-arm, 3-task paired rollout.

Reads every episode summary under artifacts/error_guided_recon_v1/episodes and
answers the spec's two questions:

  (1) Does Error-Guided Recon beat plain FULL recon (and vanilla)?
      -> error_guided_recon_k8_m10 vs semantic_recon_k8_m10  and  vs vanilla.
  (2) Is the gain from *selective* weakening-by-error, or just an *overall
      lighter* intervention?
      -> error_guided_recon_k8_m10 vs matched_strength_recon_k8_m10  (head-to-head).

For each CD arm we report paired dSR vs vanilla (Rescue/Harm/Net), exact McNemar
(two-sided binomtest on discordant pairs), and 95% CI (cluster bootstrap over
tasks). For the primary question (2) we additionally report a paired head-to-head
McNemar between error_guided and matched_strength. Diagnostics (e_i, s_i, mean_s,
perturbation norm D, residual norm) are summarized per (arm, task).

Judgment (locked by spec):
  ERROR_GUIDED_VALID iff  error_guided dSR_macro > 0 AND
      error_guided > matched_strength on macro dSR (selectivity is doing work).
  If error_guided ~ matched_strength, the winner is the *lighter-touch* story,
  not the reconstruction-error story -> STRENGTH_NOT_SELECTIVITY.

Partial-data safe: only 4-arm-complete (task, seed) pairs enter paired tables.

Run (offline, no CUDA):
  <venv>/bin/python research/semantic_token_cd/analyze_error_guided_recon.py \
      --artifact artifacts/error_guided_recon_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

PROTOCOL = "SCR_CD_ERROR_GUIDED_RECON_V1"
ARMS = (
    "vanilla",
    "semantic_recon_k8_m10",
    "error_guided_recon_k8_m10",
    "matched_strength_recon_k8_m10",
)
CD_ARMS = ARMS[1:]
ERROR_GUIDED = "error_guided_recon_k8_m10"
MATCHED = "matched_strength_recon_k8_m10"
FULL = "semantic_recon_k8_m10"
EXPECT_TASKS = 3
EXPECT_SEEDS = 100
EXPECT_EPISODES = 3 * 100 * 4

GROUP = {
    "google_robot_pick_coke_can": "direct_grasp",
    "google_robot_close_drawer": "articulated",
    "google_robot_move_near": "relational",
}


def load_records(artifact: Path):
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
        rows.append({"task": task, "seed": seed, "group": GROUP[task], "arms": arms})
    return rows, hash_fail


def cluster_boot_macro(rows, arm, n=4000, seed=731902):
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
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = [task_rows[i] for i in rng.integers(0, len(task_rows), len(task_rows))]
        vals.append(np.mean([
            int(r["arms"][arm]["success"]) - int(r["arms"]["vanilla"]["success"])
            for r in sample
        ]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def per_arm_block(rows, arm):
    v = np.array([r["arms"]["vanilla"]["success"] for r in rows], int)
    m = np.array([r["arms"][arm]["success"] for r in rows], int)
    d = m - v
    n01 = int(((v == 0) & (m == 1)).sum())   # rescued
    n10 = int(((v == 1) & (m == 0)).sum())   # harmed
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
            "delta": float(np.mean(by_task[t])),
            "rescued": int(sum(1 for r in rows if r["task"] == t
                               and r["arms"]["vanilla"]["success"] == 0
                               and r["arms"][arm]["success"] == 1)),
            "harmed": int(sum(1 for r in rows if r["task"] == t
                              and r["arms"]["vanilla"]["success"] == 1
                              and r["arms"][arm]["success"] == 0)),
            "ci95": list(seed_boot_ci([r for r in rows if r["task"] == t], arm)),
        }
        for t in sorted(by_task)
    }
    return {
        "sr_vanilla": float(v.mean()),
        "sr_arm": float(m.mean()),
        "delta": float(d.mean()),
        "delta_macro": float(np.mean([task_meta[t]["delta"] for t in task_meta])) if task_meta else 0.0,
        "rescued": n01,
        "harmed": n10,
        "net": n01 - n10,
        "mcnemar_p": mcnemar,
        "ci95_macro": list(cluster_boot_macro(rows, arm)),
        "n_pairs": len(rows),
        "n_tasks_nonworse": int(sum(1 for tm in task_meta.values() if tm["delta"] >= 0)),
        "by_task": task_meta,
    }


def head_to_head(rows, arm_a, arm_b):
    """Paired McNemar between two non-vanilla arms (discordant success pairs)."""
    a = np.array([r["arms"][arm_a]["success"] for r in rows], int)
    b = np.array([r["arms"][arm_b]["success"] for r in rows], int)
    n_ab = int(((a == 1) & (b == 0)).sum())   # A succeeds, B fails
    n_ba = int(((a == 0) & (b == 1)).sum())   # B succeeds, A fails
    discordant = n_ab + n_ba
    p = float(binomtest(n_ab, discordant).pvalue) if discordant else 1.0
    by_task = {}
    for t in sorted({r["task"] for r in rows}):
        tr = [r for r in rows if r["task"] == t]
        ta = np.array([r["arms"][arm_a]["success"] for r in tr], int)
        tb = np.array([r["arms"][arm_b]["success"] for r in tr], int)
        by_task[t] = {
            "sr_a": float(ta.mean()),
            "sr_b": float(tb.mean()),
            "a_beats_b": int(((ta == 1) & (tb == 0)).sum()),
            "b_beats_a": int(((ta == 0) & (tb == 1)).sum()),
        }
    return {
        "arm_a": arm_a, "arm_b": arm_b,
        "sr_a": float(a.mean()), "sr_b": float(b.mean()),
        "delta_sr": float(a.mean() - b.mean()),
        "a_beats_b": n_ab, "b_beats_a": n_ba,
        "net": n_ab - n_ba,
        "mcnemar_p": p,
        "n_pairs": len(rows),
        "by_task": by_task,
    }


def diagnostic_block(rows, arm):
    keys = (
        "mean_e_rel", "mean_q_ratio", "mean_cos_v_vhat",
        "mean_mean_s", "mean_perturb_norm_D", "mean_residual_norm", "mean_num_tokens",
    )
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
        "four_arm_complete_pairs": n_complete_pairs,
        "expected_pairs": EXPECT_TASKS * EXPECT_SEEDS,
        "hash_failures": [{"task": t, "seed": s} for t, s, _ in hash_fail],
        "complete": n_episodes >= EXPECT_EPISODES,
    }

    comparisons = {arm: per_arm_block(rows, arm) for arm in CD_ARMS}
    h2h_eg_vs_matched = head_to_head(rows, ERROR_GUIDED, MATCHED)
    h2h_eg_vs_full = head_to_head(rows, ERROR_GUIDED, FULL)
    diagnostics = {arm: diagnostic_block(rows, arm) for arm in CD_ARMS}

    eg = comparisons[ERROR_GUIDED]
    ms = comparisons[MATCHED]
    full = comparisons[FULL]

    eg_dr = eg["delta_macro"]
    ms_dr = ms["delta_macro"]
    full_dr = full["delta_macro"]
    selectivity_wins = eg_dr > ms_dr
    eg_positive = eg_dr > 0.0
    eg_beats_full = eg_dr > full_dr

    if eg_positive and selectivity_wins:
        decision = "ERROR_GUIDED_VALID__SELECTIVITY_HELPS"
    elif eg_positive and not selectivity_wins:
        decision = "LIGHTER_TOUCH_NOT_SELECTIVITY"
    elif not eg_positive and eg_beats_full:
        decision = "EG_NOT_POSITIVE_BUT_BEATS_FULL"
    else:
        decision = "ERROR_GUIDED_NO_GO__FULL_RECON_STILL_BEST"

    analysis = {
        "protocol": PROTOCOL,
        "coverage": coverage,
        "comparisons": comparisons,
        "head_to_head": {
            "error_guided_vs_matched_strength": h2h_eg_vs_matched,
            "error_guided_vs_full": h2h_eg_vs_full,
        },
        "diagnostics": diagnostics,
        "decision": {
            "decision": decision,
            "error_guided_delta_macro": eg_dr,
            "matched_strength_delta_macro": ms_dr,
            "full_recon_delta_macro": full_dr,
            "error_guided_positive": eg_positive,
            "selectivity_wins": selectivity_wins,
            "error_guided_beats_full": eg_beats_full,
            "h2h_eg_vs_matched_mcnemar_p": h2h_eg_vs_matched["mcnemar_p"],
            "h2h_eg_vs_matched_net": h2h_eg_vs_matched["net"],
            "criteria": {
                "eg_dr_gt_0": eg_positive,
                "eg_gt_matched": selectivity_wins,
            },
        },
    }

    (art / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    (art / "decision.json").write_text(json.dumps({
        "decision": decision,
        "error_guided_delta_macro": eg_dr,
        "matched_strength_delta_macro": ms_dr,
        "full_recon_delta_macro": full_dr,
        "h2h_eg_vs_matched_mcnemar_p": h2h_eg_vs_matched["mcnemar_p"],
    }, indent=2, sort_keys=True) + "\n")

    fieldnames = ["task", "seed", "group"] + [f"{a}_success" for a in ARMS] + [f"{a}_delta" for a in CD_ARMS]
    with (art / "paired_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {"task": r["task"], "seed": r["seed"], "group": r["group"]}
            for a in ARMS:
                row[f"{a}_success"] = int(r["arms"][a]["success"])
            for a in CD_ARMS:
                row[f"{a}_delta"] = int(r["arms"][a]["success"]) - int(r["arms"]["vanilla"]["success"])
            w.writerow(row)

    print(f"\n=== Error-Guided Recon  ({n_episodes}/{EXPECT_EPISODES} episodes, "
          f"{n_complete_pairs}/{EXPECT_TASKS * EXPECT_SEEDS} complete 4-arm pairs) ===")
    print(f"decision: {decision}\n")
    for t in sorted({r["task"] for r in rows}):
        tr = [r for r in rows if r["task"] == t]
        line = f"{t:48s} (n={len(tr)}, {GROUP[t]})"
        for a in ARMS:
            sr = np.mean([int(r["arms"][a]["success"]) for r in tr])
            line += f"  {a[:18]:>18s}={sr:5.1%}"
        print(line)
    print()
    for a in CD_ARMS:
        c = comparisons[a]
        print(f"[{a}] dSR={c['delta']:+.3f} (macro {c['delta_macro']:+.3f}, "
              f"CI95 {c['ci95_macro'][0]:+.3f}..{c['ci95_macro'][1]:+.3f})  "
              f"rescue={c['rescued']} harm={c['harmed']} net={c['net']:+d}  "
              f"mcnemar_p={c['mcnemar_p']:.3g}  nonworse={c['n_tasks_nonworse']}/3")
    print()
    print(f"[head-to-head] {ERROR_GUIDED} vs {MATCHED}: "
          f"dSR={h2h_eg_vs_matched['delta_sr']:+.3f}  a_beats_b={h2h_eg_vs_matched['a_beats_b']} "
          f"b_beats_a={h2h_eg_vs_matched['b_beats_a']}  mcnemar_p={h2h_eg_vs_matched['mcnemar_p']:.3g}")
    print(f"[head-to-head] {ERROR_GUIDED} vs {FULL}: "
          f"dSR={h2h_eg_vs_full['delta_sr']:+.3f}  a_beats_b={h2h_eg_vs_full['a_beats_b']} "
          f"b_beats_a={h2h_eg_vs_full['b_beats_a']}  mcnemar_p={h2h_eg_vs_full['mcnemar_p']:.3g}")
    print()
    for a in CD_ARMS:
        d = diagnostics[a]
        if "mean_mean_s" in d and "mean_perturb_norm_D" in d:
            print(f"[{a}] mean_s={d['mean_mean_s']['mean']:.4f}  "
                  f"D={d['mean_perturb_norm_D']['mean']:.4f}  "
                  f"e_rel={d['mean_e_rel']['mean']:.4f}  "
                  f"residual={d['mean_residual_norm']['mean']:.4f}")
    print(f"\nselectivity (eg>matched)? {selectivity_wins} | eg positive? {eg_positive} | "
          f"eg beats full? {eg_beats_full}")
    print(f"decision: {decision}\n")


if __name__ == "__main__":
    main()
