"""Final paired statistics for the pi0-FAST L6-Matched run.

Deliverables, all keyed on the physics-level pairing (initial_agent_state_sha256) rather than
the RGB hash, because the SIMPLER renderer is sporadically non-deterministic (see
KNOWN_LIMITATIONS.md section 1):

  * per-task success counts and rates for the five arms
  * paired DeltaSR against vanilla
  * Rescue (vanilla fail, guided success) and Harm (vanilla success, guided fail), Net Success
  * exact McNemar test per lambda, Holm-corrected across the four lambdas
  * 95% CI for the paired rate difference (task-stratified bootstrap, so tasks of different
    difficulty do not distort the interval)
  * FAST stream validity and the diagnostic summary (mean m_t, logit JS proxy, action change)

Only episodes with status=complete are counted; infra failures are reported separately rather
than silently dropped.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

RUN = Path("/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_l6_matched_9task_100seed")
ARMS = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075", "l6_matched_100",
        "l6_matched_125", "l6_matched_150"]
SHORT = {"vanilla": "Vanilla", "l6_matched_025": "lambda=0.25", "l6_matched_050": "lambda=0.50",
         "l6_matched_075": "lambda=0.75", "l6_matched_100": "lambda=1.00",
         "l6_matched_125": "lambda=1.25", "l6_matched_150": "lambda=1.50"}
BOOT = 2000
SEED = 12345


def load():
    rows = []
    for f in RUN.glob("episodes/*/*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        rows.append(d)
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: b and c are the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 2 * P(X <= k) with X ~ Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals: list[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """Holm-Bonferroni step-down. Returns (name, p, adjusted_p) in the original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i][1])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i][1]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return [(pvals[i][0], pvals[i][1], adj[i]) for i in range(m)]


def stratified_bootstrap_ci(pairs, lam: str, n_boot: int = BOOT, seed: int = SEED):
    """Paired DeltaSR CI, resampling seeds within each task so task mix stays fixed."""
    by_task = collections.defaultdict(list)
    for (task, seed), arms in pairs.items():
        if "vanilla" in arms and lam in arms:
            by_task[task].append((int(bool(arms["vanilla"])), int(bool(arms[lam]))))
    tasks = sorted(by_task)
    if not tasks:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        tot_v = tot_g = tot_n = 0
        for t in tasks:
            sample = rng.choices(by_task[t], k=len(by_task[t]))
            tot_v += sum(s[0] for s in sample)
            tot_g += sum(s[1] for s in sample)
            tot_n += len(sample)
        if tot_n:
            diffs.append(100.0 * (tot_g - tot_v) / tot_n)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    point = 100.0 * (
        sum(s[1] for t in tasks for s in by_task[t]) - sum(s[0] for t in tasks for s in by_task[t])
    ) / sum(len(by_task[t]) for t in tasks)
    return point, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RUN / "final_statistics.json")
    args = ap.parse_args()

    rows = load()
    complete = [d for d in rows if d.get("status") == "complete"]
    incomplete = [d for d in rows if d.get("status") != "complete"]
    tasks = sorted({d["task"] for d in complete})

    # ---- per task x arm ----
    table = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for d in complete:
        c = table[d["task"]][d["arm"]]
        c[0] += 1
        c[1] += int(bool(d.get("success")))

    print(f"episodes: complete={len(complete)} incomplete={len(incomplete)} tasks={len(tasks)}")
    hdr = f"{'Task':<44}" + "".join(f"{SHORT[a]:>14}" for a in ARMS)
    print(hdr)
    print("-" * len(hdr))
    tot = collections.defaultdict(lambda: [0, 0])
    for t in tasks:
        line = f"{t.replace('google_robot_',''):<44}"
        for a in ARMS:
            n, s = table[t][a]
            tot[a][0] += n
            tot[a][1] += s
            line += f"{(f'{s}/{n} {100*s/n:.0f}%' if n else '—'):>14}"
        print(line)
    print("-" * len(hdr))
    line = f"{'OVERALL':<44}"
    for a in ARMS:
        n, s = tot[a]
        line += f"{(f'{s}/{n} {100*s/n:.1f}%' if n else '—'):>14}"
    print(line)

    # ---- paired analysis ----
    # With more than two arms, not every (task, seed) has every arm yet, so each arm is
    # paired only over the seeds where BOTH it and vanilla are present.
    pairs = collections.defaultdict(dict)
    for d in complete:
        pairs[(d["task"], d["seed"])][d["arm"]] = bool(d.get("success"))
    all_arms_pairs = {k: v for k, v in pairs.items() if len(v) == len(ARMS)}

    results = {"episodes_complete": len(complete), "episodes_incomplete": len(incomplete),
               "tasks": tasks, "arms": ARMS,
               "pairs_with_all_arms": len(all_arms_pairs),
               "per_task": {t: {a: {"n": table[t][a][0], "success": table[t][a][1],
                                    "rate": (table[t][a][1] / table[t][a][0]) if table[t][a][0] else None}
                                for a in ARMS} for t in tasks},
               "overall": {a: {"n": tot[a][0], "success": tot[a][1],
                               "rate": tot[a][1] / tot[a][0] if tot[a][0] else None} for a in ARMS},
               "paired": {}}

    print("\n=== paired vs vanilla (seeds where both arms are present) ===")
    pvals = []
    for a in ARMS[1:]:
        arm_pairs = {k: v for k, v in pairs.items() if "vanilla" in v and a in v}
        if not arm_pairs:
            print(f"  {SHORT[a]:>14}: no paired data yet")
            continue
        b = c = both = neither = 0
        for k, v in arm_pairs.items():
            vo, go = v["vanilla"], v[a]
            if not vo and go:
                b += 1
            elif vo and not go:
                c += 1
            elif vo and go:
                both += 1
            else:
                neither += 1
        n = b + c + both + neither
        guided_rate = 100.0 * (b + both) / n if n else None
        vanilla_rate = 100.0 * (c + both) / n if n else None
        delta = (guided_rate - vanilla_rate) if n else None
        p = mcnemar_exact(b, c)
        ci = stratified_bootstrap_ci(arm_pairs, a)
        pvals.append((a, p))
        results["paired"][a] = {
            "n_pairs": n, "rescue": b, "harm": c, "net_success": b - c,
            "both_success": both, "both_fail": neither,
            "vanilla_rate": vanilla_rate,
            "guided_rate": guided_rate,
            "delta_sr_pp": delta,
            "mcnemar_exact_p": p,
            "ci95_pp": {"point": ci[0], "lo": ci[1], "hi": ci[2]} if ci else None,
        }
        print(f"  {SHORT[a]:>14}: n={n:4d}  rescue={b:3d} harm={c:3d} net={b-c:+3d}  "
              f"ΔSR={delta:+6.2f}pp  "
              f"CI95=[{ci[1]:+.2f},{ci[2]:+.2f}]  McNemar p={p:.4g}")

    adj = holm(pvals)
    print("\n=== Holm-corrected across the four lambdas ===")
    for name, p, pa in adj:
        results["paired"][name]["holm_adjusted_p"] = pa
        print(f"  {SHORT[name]:>14}: p={p:.4g}  p_holm={pa:.4g}  "
              f"{'significant at 0.05' if pa < 0.05 else 'not significant'}")

    # ---- diagnostics ----
    diag = collections.defaultdict(lambda: {"m_t": [], "residual": [], "flips": 0, "chunks": 0,
                                            "valid": 0, "agrees": 0, "deltas": collections.Counter()})
    for d in complete:
        if d["arm"] == "vanilla":
            continue
        for ch in d.get("chunks", []):
            t = ch.get("trace", {})
            g = diag[d["task"]]
            g["chunks"] += 1
            if t.get("matched_m_t"):
                g["m_t"].append(t["matched_m_t"])
            if t.get("first_logit_residual_norm"):
                g["residual"].append(t["first_logit_residual_norm"])
            if t.get("first_clean_top_token") != t.get("first_negative_top_token"):
                g["flips"] += 1
            if t.get("fast_action_stream_valid"):
                g["valid"] += 1
            if t.get("openpi_official_decoder_agrees"):
                g["agrees"] += 1
            g["deltas"][t.get("decode_target_delta")] += 1
    results["diagnostics"] = {}
    print("\n=== diagnostics (guided arms) ===")
    print(f"  {'Task':<44}{'mean m_t':>10}{'resid':>9}{'flip%':>8}{'valid%':>8}")
    for t in tasks:
        g = diag.get(t)
        if not g or not g["chunks"]:
            continue
        mean_m = sum(g["m_t"]) / len(g["m_t"]) if g["m_t"] else None
        mean_r = sum(g["residual"]) / len(g["residual"]) if g["residual"] else None
        results["diagnostics"][t] = {
            "chunks": g["chunks"], "mean_m_t": mean_m, "mean_logit_residual": mean_r,
            "first_token_flip_rate": g["flips"] / g["chunks"],
            "stream_valid_rate": g["valid"] / g["chunks"],
            "decoder_agrees_rate": g["agrees"] / g["chunks"],
            "decode_delta_histogram": {str(k): v for k, v in g["deltas"].items()},
        }
        print(f"  {t.replace('google_robot_',''):<44}{mean_m:10.1f}{mean_r:9.1f}"
              f"{100*g['flips']/g['chunks']:7.1f}%{100*g['valid']/g['chunks']:7.1f}%")

    allvalid = sum(v["valid"] for v in diag.values())
    allchunks = sum(v["chunks"] for v in diag.values())
    allagree = sum(v["agrees"] for v in diag.values())
    results["fast_stream_valid_rate"] = allvalid / allchunks if allchunks else None
    results["decoder_agrees_rate"] = allagree / allchunks if allchunks else None
    print(f"\n  overall FAST stream valid: {allvalid}/{allchunks} "
          f"({100*allvalid/max(allchunks,1):.3f}%)   decoder agrees: {allagree}/{allchunks}")

    args.out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
