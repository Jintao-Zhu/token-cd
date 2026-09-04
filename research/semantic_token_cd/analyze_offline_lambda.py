"""Offline lambda / SC-SHR harm analysis on the local paired rollouts.

Known-success points are only lambda in {0, 0.5}: lambda=0 = vanilla arm,
lambda=0.5 = SHR arm (and SC-SHR). Intermediate lambdas are analyzed at the
first control step in token/action space (flip count, bin jump) because success
for those lambdas would require environment rollout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data/docker/dev_zjt/data/code/artifacts")
V1 = ROOT / "spatial_harmonic_recon_v1" / "episodes"
SC = ROOT / "sc_shr_local_v1" / "episodes"
TASKS = [
    "google_robot_close_drawer",
    "google_robot_open_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
]
LAMBDAS = (0.0, 0.25, 0.5, 0.75)
OUT = ROOT / "offline_lambda_analysis_v1"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def cdfinal(pos_row, neg_row, lam):
    final = np.argmax(pos_row, axis=-1).astype(np.int64)
    final[:6] = np.argmax((1 + lam) * pos_row[:6] - lam * neg_row[:6], axis=-1)
    return final


def flips(clean, final):
    idx = np.nonzero(final != clean)[0]
    bins = [int(abs(int(final[i]) - int(clean[i]))) for i in idx]
    return (idx.tolist(), float(np.mean(bins)) if bins else 0.0,
            float(np.max(bins)) if bins else 0.0)


def main() -> None:
    rows = []
    for task in TASKS:
        for sc_file in (SC / task / "sc_shr_harmonic").glob("episode_*_summary.json"):
            seed = int(sc_file.name.split("_")[1])
            sc = json.loads(sc_file.read_text())
            v = json.loads((V1 / task / "vanilla" / f"episode_{seed:03d}_summary.json").read_text())
            s = json.loads((V1 / task / "spatial_harmonic_recon"
                            / f"episode_{seed:03d}_summary.json").read_text())
            rec = {
                "task": task, "seed": seed,
                "vanilla_ok": bool(v["success"]), "shr_ok": bool(s["success"]),
                "sc_ok": bool(sc["success"]),
            }
            # SHR first-step trace features
            st = (s.get("selector_trace") or [])
            if st:
                t0 = st[0]
                rec["shr_g"] = int(t0.get("num_tokens", np.nan))
                rec["shr_D"] = fnum(t0.get("shr_perturb_norm_D"))
                rec["shr_cos"] = fnum((t0.get("shr_cos") or {}).get("median"))
                rec["n_entities"] = len(t0.get("selected_entities") or [])
            # SC first-step trace features
            cst = (sc.get("selector_trace") or [])
            if cst:
                c0 = cst[0]
                rec["sc_before"] = int(c0.get("sc_before_tokens", np.nan))
                rec["sc_after"] = int(c0.get("sc_after_tokens", np.nan))
                rec["sc_kept"] = c0.get("sc_kept_component_sizes", [])
                rec["sc_components"] = c0.get("sc_component_sizes", [])
            # SHR arrays: first-step lambda response
            arr = np.load(V1 / task / "spatial_harmonic_recon"
                          / f"episode_{seed:03d}_arrays.npz")
            pos = np.asarray(arr["positive_logits"][0], dtype=np.float32)
            neg = np.asarray(arr["negative_logits"][0], dtype=np.float32)
            clean = np.argmax(pos, axis=-1).astype(np.int64)
            for lam in LAMBDAS:
                idx, bmean, bmax = flips(clean, cdfinal(pos, neg, lam))
                rec[f"flip_{lam}"] = len(idx)
                rec[f"binmax_{lam}"] = bmax
                rec[f"bins_{lam}"] = sorted(int(i) for i in idx)
            rows.append(rec)

    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (OUT / "tables" / "per_seed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # --- Table 1: known lambda preference {0, 0.5} from real rollouts ---
    lines = ["# Offline lambda analysis (local v1 + SC-SHR)"]
    lines.append("")
    lines.append("## 1. Known per-episode lambda outcome (real rollouts)")
    lines.append("")
    lines.append("`pref=0.5`: SHR(0.5) ok; `pref=0`: only vanilla(0) ok; `both`: both ok; `neither`: both fail.")
    lines.append("")
    lines.append("| task | n | only lam=0 | only lam=0.5 | both ok | neither |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        c = {"0": 0, "0.5": 0, "both": 0, "neither": 0}
        for r in rs:
            if r["vanilla_ok"] and r["shr_ok"]:
                c["both"] += 1
            elif (not r["vanilla_ok"]) and r["shr_ok"]:
                c["0.5"] += 1
            elif r["vanilla_ok"] and (not r["shr_ok"]):
                c["0"] += 1
            else:
                c["neither"] += 1
        lines.append(f"| {task.replace('google_robot_','')} | {len(rs)} | {c['0']} | {c['0.5']} | {c['both']} | {c['neither']} |")

    # --- Table 2: offline first-step action response per task ---
    lines.append("")
    lines.append("## 2. First-step action response to lambda (offline, token level)")
    lines.append("")
    lines.append("Mean flip dims and mean max bin jump at step 0 over the SHR stored states.")
    lines.append("")
    lines.append("| task | n | lambda | mean flip dims | mean max bin jump |")
    lines.append("|---|---:|---:|---:|---:|")
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        for lam in LAMBDAS:
            fl = np.mean([r[f"flip_{lam}"] for r in rs])
            bm = np.mean([r[f"binmax_{lam}"] for r in rs])
            lines.append(f"| {task.replace('google_robot_','')} | {len(rs)} | {lam} | {fl:.2f} | {bm:.1f} |")

    # --- Table 3: SC-SHR harm cases on move_near ---
    lines.append("")
    lines.append("## 3. SC-SHR harm (SHR ok, SC fail) on move_near")
    lines.append("")
    harm = [r for r in rows if r["task"] == "google_robot_move_near"
            and r["shr_ok"] and not r["sc_ok"]]
    ok = [r for r in rows if r["task"] == "google_robot_move_near" and r["sc_ok"]]
    lines.append("Per-seed rows: SHR G size, SC before/after, kept components, flip dims at step0.")
    lines.append("")
    lines.append("| seed | SHR G | SC before | SC after | kept comps | flips @0.25 | flips @0.5 | flips @0.75 |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for r in sorted(harm, key=lambda x: x["seed"]):
        lines.append(
            f"| {r['seed']} | {r['shr_g']:.0f} | {r['sc_before']:.0f} | {r['sc_after']:.0f} "
            f"| {r.get('sc_kept')} | {r['flip_0.25']} | {r['flip_0.5']} | {r['flip_0.75']} |")
    def mean(rs, k):
        vals = [r.get(k) for r in rs]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return float(np.mean(vals)) if vals else float("nan")
    lines.append("")
    lines.append("Aggregates (move_near):")
    lines.append("")
    lines.append("| group | n | SHR G | SC before | SC after | kept |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| harm | {len(harm)} | {mean(harm,'shr_g'):.1f} | {mean(harm,'sc_before'):.1f} "
                 f"| {mean(harm,'sc_after'):.1f} | {np.mean([len(r.get('sc_kept',[])) for r in harm]):.2f} |")
    lines.append(f"| sc ok | {len(ok)} | {mean(ok,'shr_g'):.1f} | {mean(ok,'sc_before'):.1f} "
                 f"| {mean(ok,'sc_after'):.1f} | {np.mean([len(r.get('sc_kept',[])) for r in ok]):.2f} |")

    # --- Table 4: correlate known lambda preference with step0 G / D ---
    lines.append("")
    lines.append("## 4. Known lambda preference vs first-step features (mean)")
    lines.append("")
    lines.append("| task | group | n | SHR G | D | cos med |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        for label, cond in (
            ("pref=0", lambda r: r["vanilla_ok"] and not r["shr_ok"]),
            ("pref=0.5", lambda r: (not r["vanilla_ok"]) and r["shr_ok"]),
            ("both", lambda r: r["vanilla_ok"] and r["shr_ok"]),
        ):
            sub = [r for r in rs if cond(r)]
            if not sub:
                continue
            lines.append(f"| {task.replace('google_robot_','')} | {label} | {len(sub)} "
                         f"| {mean(sub,'shr_g'):.1f} | {mean(sub,'shr_D'):.3f} | {mean(sub,'shr_cos'):.3f} |")

    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
