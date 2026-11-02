"""Analysis for the Pi0 guidance-form search (PI0_GUIDANCE_FORM_5TASK_V1).

Pre-declared analysis choices
-----------------------------
* Primary endpoint: paired episode success, arm versus the in-run `pi0_vanilla`,
  with pairing on (task, seed) cells that share an identical initial-state and
  initial-RGB hash.
* Test: two-sided exact McNemar (binomial on the discordant pairs).  No normal
  approximation, no independent-arm bootstrap.
* Multiplicity: Holm over the 7 guided arms tested against vanilla.  That family
  is fixed here, before reading the outcomes.
* Interval: percentile bootstrap over resampled (task, seed) cells, so the same
  cell index is used for both arms.  Independent-arm resampling would be wrong.
* Non-significance is reported as "not established", never as equivalence.
* Per-task results are always shown; a pooled number alone hides the ceiling
  effects that motivated dropping pick_coke_can (vanilla 296/300 there).

Outputs: artifacts/pi0_guidance_form_5task_100seed_v1/ANALYSIS.{json,md}
"""

from __future__ import annotations

import glob
import json
import pathlib
from collections import defaultdict
from math import comb

import numpy as np

REPO = pathlib.Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = REPO / "artifacts" / "pi0_guidance_form_5task_100seed_v1"
REF = REPO / "artifacts" / "pi0_shr_5task_0_299_v1"

GUIDED_ARMS = [
    "pi0_dense_l050",
    "pi0_dense_l150",
    "pi0_dense_l300",
    "pi0_top2_l050",
    "pi0_top2_l150",
    "pi0_last3_l050",
    "pi0_relgate015_l050",
]
B = 20000
RNG = np.random.default_rng(0)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[name] = running
    return out


def load(arm_root: pathlib.Path, arm: str) -> dict:
    """(task, seed) -> (success, state_hash, rgb_hash)"""
    out = {}
    for f in glob.glob(str(arm_root / "*" / arm / "*_summary.json")):
        d = json.loads(pathlib.Path(f).read_text())
        out[(d["task"], int(d["seed"]))] = (
            bool(d["success"]),
            d.get("initial_state_sha256"),
            d.get("initial_rgb_sha256"),
        )
    return out


def main() -> None:
    art_dir = ART / "episodes"
    tasks = sorted(p.name for p in art_dir.iterdir() if p.is_dir())
    arms = ["pi0_vanilla"] + GUIDED_ARMS

    raw = {arm: load(art_dir, arm) for arm in arms}
    vanilla = raw["pi0_vanilla"]

    report: dict = {"protocol": "PI0_GUIDANCE_FORM_5TASK_V1", "tasks": tasks, "arms": {}}

    # ---- integrity: pairing hashes must agree across arms -------------------
    pairing_ok, pairing_bad = 0, []
    for arm in GUIDED_ARMS:
        for key, (_, sh, rh) in raw[arm].items():
            if key not in vanilla:
                continue
            _, vsh, vrh = vanilla[key]
            if sh == vsh and rh == vrh:
                pairing_ok += 1
            else:
                pairing_bad.append((arm, key))
    report["pairing"] = {
        "cells_checked": pairing_ok + len(pairing_bad),
        "cells_with_identical_state_and_rgb": pairing_ok,
        "mismatches": pairing_bad[:20],
        "n_mismatches": len(pairing_bad),
    }

    # ---- per-arm totals -----------------------------------------------------
    for arm in arms:
        succ = sum(1 for v in raw[arm].values() if v[0])
        report["arms"][arm] = {
            "episodes": len(raw[arm]),
            "successes": succ,
            "rate": succ / max(len(raw[arm]), 1),
        }

    # ---- paired comparisons vs vanilla -------------------------------------
    cells = sorted(set(vanilla) & set().union(*[set(raw[a]) for a in GUIDED_ARMS]))
    task_index = {t: i for i, t in enumerate(tasks)}
    idx = np.array([task_index[k[0]] for k in cells])
    v = np.array([vanilla[k][0] for k in cells], dtype=float)

    pvals: dict[str, float] = {}
    for arm in GUIDED_ARMS:
        a = np.array([raw[arm][k][0] for k in cells], dtype=float)
        rescue = int(((a == 1) & (v == 0)).sum())
        harm = int(((a == 0) & (v == 1)).sum())
        net = rescue - harm
        p = mcnemar_exact(rescue, harm)

        # paired bootstrap over shared cells, stratified reshuffling within nothing:
        # resample cell indices once and reuse for both arms
        d = a - v
        boots = np.empty(B)
        n = len(d)
        for i in range(B):
            pick = RNG.integers(0, n, n)
            boots[i] = d[pick].mean() * 100.0
        lo, hi = np.percentile(boots, [2.5, 97.5])

        per_task = {}
        for t in tasks:
            m = idx == task_index[t]
            if not m.any():
                continue
            at, vt = a[m], v[m]
            per_task[t] = {
                "n": int(m.sum()),
                "arm_success": int(at.sum()),
                "vanilla_success": int(vt.sum()),
                "rescue": int(((at == 1) & (vt == 0)).sum()),
                "harm": int(((at == 0) & (vt == 1)).sum()),
                "net": int(((at == 1) & (vt == 0)).sum() - ((at == 0) & (vt == 1)).sum()),
                "rate_diff_pp": float((at.mean() - vt.mean()) * 100.0),
            }

        report["arms"][arm].update({
            "paired_cells": len(cells),
            "rescue": rescue,
            "harm": harm,
            "net": net,
            "rate_diff_pp": float((a.mean() - v.mean()) * 100.0),
            "mcnemar_exact_p": p,
            "paired_rate_diff_ci95_pp": [float(lo), float(hi)],
            "per_task": per_task,
            "task_macro_rate_diff_pp": float(np.mean([x["rate_diff_pp"] for x in per_task.values()])),
        })
        pvals[arm] = p

    adj = holm(pvals)
    for arm in GUIDED_ARMS:
        report["arms"][arm]["holm_adjusted_p"] = adj[arm]
        report["arms"][arm]["holm_family_size"] = len(pvals)

    (ART / "ANALYSIS.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    # ---- markdown ----------------------------------------------------------
    L = []
    L.append("# Pi0 guidance-form search (PI0_GUIDANCE_FORM_5TASK_V1)\n")
    L.append(f"4 tasks x 100 seeds x 8 arms. Pairing checked on "
             f"{report['pairing']['cells_checked']} cells: "
             f"{report['pairing']['cells_with_identical_state_and_rgb']} share both the "
             f"initial-state and initial-RGB hash.\n")
    L.append("## Pooled (4 tasks, 400 paired cells per arm)\n")
    L.append("| arm | success | rate | vs vanilla (pp) | 95% paired CI | rescue/harm | exact p | Holm p |")
    L.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for arm in arms:
        r = report["arms"][arm]
        if arm == "pi0_vanilla":
            L.append(f"| `{arm}` | {r['successes']}/{r['episodes']} | {100*r['rate']:.2f}% | — | — | — | — | — |")
            continue
        ci = r["paired_rate_diff_ci95_pp"]
        L.append(
            f"| `{arm}` | {r['successes']}/{r['episodes']} | {100*r['rate']:.2f}% | "
            f"{r['rate_diff_pp']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
            f"{r['rescue']}/{r['harm']} | {r['mcnemar_exact_p']:.4f} | {r['holm_adjusted_p']:.4f} |"
        )
    L.append("\nHolm family = the 7 guided arms versus the in-run vanilla.\n")
    L.append("## Per task (net = rescue - harm)\n")
    header = "| arm | " + " | ".join(t.replace("google_robot_", "g_").replace("widowx_", "w_") for t in tasks) + " | macro pp |"
    L.append(header)
    L.append("|---|" + "---:|" * (len(tasks) + 1))
    for arm in GUIDED_ARMS:
        r = report["arms"][arm]
        cells_md = []
        for t in tasks:
            pt = r["per_task"].get(t)
            cells_md.append("—" if not pt else f"{pt['net']:+d} ({pt['rate_diff_pp']:+.1f})")
        L.append(f"| `{arm}` | " + " | ".join(cells_md) + f" | {r['task_macro_rate_diff_pp']:+.2f} |")
    L.append("\nPer-task cells are 100 paired episodes each; individual task contrasts are "
             "underpowered and shown descriptively.\n")
    (ART / "ANALYSIS.md").write_text("\n".join(L) + "\n")

    print("\n".join(L))


if __name__ == "__main__":
    main()
