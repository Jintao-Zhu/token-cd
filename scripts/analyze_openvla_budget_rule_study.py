#!/usr/bin/env python3
"""Final analysis for the OpenVLA budget-rule study (Matched vs Attention-TopP vs Fixed, lambda=1.0).

Reference : artifacts/openvla_lambda1_extension_200_299_v1   (matched lambda=1.0, 900 episodes)
Study     : artifacts/openvla_budget_rule_9task_200_299_v1   (11 new arms x 900 episodes)

Reports
  1. coverage / integrity per arm (with technical_pass and pairing-hash agreement)
  2. overall table: success/total, rate, paired delta vs Matched (rescue/harm), exact McNemar p,
     Holm-adjusted p over the 11 comparisons, and the realized budget statistics
     (mean / median / min / max of the per-episode mean m_t)
  3. per-task x arm success counts (the Supplementary table)
  4. success vs realized mean budget (the Figure-6c data)
  5. trend tests within each family (Cochran-Armitage on the ordered TopP thresholds and the
     ordered fixed budgets)

Usage: python3 scripts/analyze_openvla_budget_rule_study.py [--write]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

REFERENCE = Path("artifacts/openvla_lambda1_extension_200_299_v1")
STUDY = Path("artifacts/openvla_budget_rule_9task_200_299_v1")
REF_ARM = "lambda_100"
TASKS = [
    "google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can",
    "google_robot_move_near", "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
]
TOPP = [("top_p075_l100", 0.75), ("top_p080_l100", 0.80), ("top_p085_l100", 0.85), ("top_p090_l100", 0.90)]
FIXED = [("fixed016_l100", 16), ("fixed024_l100", 24), ("fixed032_l100", 32), ("fixed040_l100", 40),
         ("fixed048_l100", 48), ("fixed056_l100", 56), ("fixed064_l100", 64)]
ARMS = [a for a, _ in TOPP] + [a for a, _ in FIXED]
SHORT = lambda t: t.replace("google_robot_", "").replace("widowx_", "")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        val = min(1.0, pvals[i] * (m - rank))
        running = max(running, val)
        adj[i] = running
    return adj


def cochran_armitage(groups: list[tuple[float, int, int]]) -> tuple[float, float]:
    """groups = [(score, n, successes), ...] -> (z, two-sided p)."""
    N = sum(n for _, n, _ in groups)
    if N == 0:
        return 0.0, 1.0
    X = sum(x for _, _, x in groups)
    pbar = X / N
    if pbar in (0.0, 1.0):
        return 0.0, 1.0
    sum_nt = sum(n * t for t, n, _ in groups)
    sum_nt2 = sum(n * t * t for t, n, _ in groups)
    T = sum(t * (x - n * pbar) for t, n, x in groups)
    var = pbar * (1 - pbar) * (sum_nt2 - sum_nt ** 2 / N)
    if var <= 0:
        return 0.0, 1.0
    z = T / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def load(root: Path, arm: str) -> dict[tuple[str, int], dict]:
    out = {}
    for f in glob.glob(str(root / "episodes" / "*" / arm / "episode_*_summary.json")):
        d = json.load(open(f))
        out[(d["task"], int(d["seed"]))] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--study", default=str(STUDY))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out-prefix", default="docs/OPENVLA_BUDGET_RULE_9TASK_FINAL")
    args = ap.parse_args()

    ref = load(Path(args.reference), REF_ARM)
    study = {a: load(Path(args.study), a) for a in ARMS}
    keys = sorted(set(ref) & set().union(*[set(v) for v in study.values()])) if study else []

    L: list[str] = []
    add = L.append
    add("# OpenVLA budget-rule study — Matched vs Attention-TopP vs Fixed budget (λ=1.0)")
    add("")
    add(f"*reference:* `{args.reference}` (matched, 900 episodes, seeds 200–299)  ·  "
        f"*study:* `{args.study}`")
    add("")
    out = {"reference": args.reference, "study": args.study, "arms": {}}

    # ---------------- 1. coverage ----------------
    add("## 1. Coverage / integrity")
    add("")
    add("| arm | episodes | technical_pass | state-hash agreement | mean m_t | median m_t | min | max |")
    add("|---|---|---|---|---|---|---|---|")
    stats = {}
    for arm in ARMS:
        rows = study[arm]
        pass_n = sum(1 for d in rows.values() if d.get("technical_pass"))
        agree = sum(1 for k, d in rows.items() if k in ref
                    and d.get("initial_state_sha256") == ref[k].get("initial_state_sha256")
                    and d.get("initial_rgb_sha256") == ref[k].get("initial_rgb_sha256"))
        mts = [d.get("mean_selected_token_count") or 0 for d in rows.values()]
        stats[arm] = {
            "n": len(rows), "technical_pass": pass_n, "hash_agreement": agree,
            "mean_m_t": statistics.mean(mts) if mts else None,
            "median_m_t": statistics.median(mts) if mts else None,
            "min_m_t": min(mts) if mts else None, "max_m_t": max(mts) if mts else None,
        }
        s = stats[arm]
        add(f"| {arm} | {s['n']}/900 | {pass_n} | {agree}/{s['n']} | "
            f"{s['mean_m_t']:.1f} | {s['median_m_t']:.1f} | {s['min_m_t']:.0f} | {s['max_m_t']:.0f} |")

    # ---------------- 2. overall ----------------
    add("")
    add("## 2. Overall, paired against Matched on the same (task, seed) cells")
    add("")
    ref_succ = sum(1 for d in ref.values() if d.get("success"))
    add(f"Matched reference: **{ref_succ}/{len(ref)}** = {100 * ref_succ / len(ref):.1f}%")
    add("")
    add("| arm | success/total | rate | Δ net | rescue/harm | McNemar p | Holm p | mean m_t |")
    add("|---|---|---|---|---|---|---|---|")
    pvals, arms_used, rows_out = [], [], {}
    for arm in ARMS:
        b = c = n = 0
        sc = 0
        for k in ref:
            d = study[arm].get(k)
            if not d or not d.get("success") in (True, False):
                continue
            n += 1
            rs, ds = bool(ref[k].get("success")), bool(d.get("success"))
            sc += ds
            if ds and not rs:
                b += 1
            elif rs and not ds:
                c += 1
        p = mcnemar_exact(b, c)
        pvals.append(p)
        arms_used.append(arm)
        rows_out[arm] = {"n": n, "success": sc, "rate": sc / n if n else 0,
                         "rescue": b, "harm": c, "net": b - c, "p": p,
                         "mean_m_t": stats[arm]["mean_m_t"]}
    adj = holm(pvals)
    for arm, pa in zip(arms_used, adj):
        r = rows_out[arm]
        add(f"| {arm} | {r['success']}/{r['n']} | {100 * r['rate']:.1f}% | {r['net']:+d} | "
            f"{r['rescue']}/{r['harm']} | {r['p']:.4f} | {pa:.4f} | {r['mean_m_t']:.1f} |")
    out["overall"] = {a: {**rows_out[a], "holm_p": pa} for a, pa in zip(arms_used, adj)}

    # ---------------- 3. per task ----------------
    add("")
    add("## 3. Per task (success / total, same-seed cells)")
    add("")
    add("| task | Matched | " + " | ".join(a.replace("_l100", "") for a in ARMS) + " |")
    add("|---" * (len(ARMS) + 2) + "|")
    for t in TASKS:
        cells = []
        rn = rs = 0
        for k, d in ref.items():
            if k[0] == t:
                rn += 1
                rs += bool(d.get("success"))
        cells.append(f"{rs}/{rn}")
        for arm in ARMS:
            n = s = 0
            for k in ref:
                if k[0] != t:
                    continue
                d = study[arm].get(k)
                if not d:
                    continue
                n += 1
                s += bool(d.get("success"))
            cells.append(f"{s}/{n}")
        add(f"| {SHORT(t)} | " + " | ".join(cells) + " |")

    # ---------------- 4. success vs realized budget ----------------
    add("")
    add("## 4. Success vs realized mean budget (Figure 6c data)")
    add("")
    add("| arm | mean m_t | success rate |")
    add("|---|---|---|")
    add(f"| matched | {statistics.mean([d.get('mean_selected_token_count') or 0 for d in ref.values()]):.1f} | "
        f"{100 * ref_succ / len(ref):.1f}% |")
    for arm in ARMS:
        r = rows_out[arm]
        add(f"| {arm} | {r['mean_m_t']:.1f} | {100 * r['rate']:.1f}% |")

    # ---------------- 5. trend tests ----------------
    add("")
    add("## 5. Trend within each family (Cochran-Armitage)")
    add("")
    add("| family | ordering | z | p |")
    add("|---|---|---|---|")
    fam_out = {}
    for name, fam in (("Attention-TopP", TOPP), ("Fixed budget", FIXED)):
        groups = []
        for arm, score in fam:
            r = rows_out[arm]
            groups.append((float(score), r["n"], r["success"]))
        z, p = cochran_armitage(groups)
        fam_out[name] = {"z": z, "p": p}
        add(f"| {name} | {fam[0][0].split('_')[1]} → {fam[-1][0].split('_')[1]} | {z:+.3f} | {p:.4f} |")
    out["family_trends"] = fam_out
    out["budget_stats"] = stats

    text = "\n".join(L) + "\n"
    print(text)
    if args.write:
        Path(args.out_prefix + ".md").write_text(text)
        Path(args.out_prefix + ".json").write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.out_prefix}.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
