"""State-coupling ablation analysis: Matched vs Matched-Shuffle (frozen permuted tape)."""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")


def load(base: Path, arm: str | None = None):
    """Load episodes. If arm is given, only that arm directory is read
    (base is expected to be the --arm-name/../<arm> root or contain it)."""
    out = {}
    pattern = f"*/{arm}/episode_*_summary.json" if arm else "episode_*_summary.json"
    for f in base.glob(pattern) if arm else base.rglob(pattern):
        d = json.loads(f.read_text())
        out[(d["task"], int(d["seed"]))] = d
    return out


def mcnemar(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--matched-artifact", type=Path, required=True)
    ap.add_argument("--out-prefix", default="STATE_COUPLING")
    args = ap.parse_args()
    root = args.artifact.resolve()
    ref = load(args.matched_artifact.resolve() / "episodes", arm="l11_matched")

    sh = {}
    for f in (root / "episodes").rglob("episode_*_summary.json"):
        d = json.loads(f.read_text())
        sh[(d["task"], int(d["seed"]))] = d

    keys = sorted(set(ref) & set(sh))
    L = ["# Matched 状态耦合消融（episode 内打乱预算 tape）", "",
         f"可比配对 episode 数: **{len(keys)}**", ""]
    L += ["| 任务 | n | Matched | Shuffle | Harm | Rescue | 净 | McNemar p |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    tot = {"n": 0, "m": 0, "s": 0, "h": 0, "r": 0}
    per_task = {}
    for t in TASKS:
        ks = [k for k in keys if k[0] == t]
        if not ks:
            continue
        m = sum(1 for k in ks if ref[k]["success"]); s = sum(1 for k in ks if sh[k]["success"])
        h = sum(1 for k in ks if ref[k]["success"] and not sh[k]["success"])
        r = sum(1 for k in ks if (not ref[k]["success"]) and sh[k]["success"])
        p = mcnemar(h, r)
        per_task[t] = {"n": len(ks), "matched": m, "shuffle": s, "harm": h, "rescue": r, "p": p}
        tot["n"] += len(ks); tot["m"] += m; tot["s"] += s; tot["h"] += h; tot["r"] += r
        L.append(f"| {t.replace('google_robot_','')} | {len(ks)} | {m}/{len(ks)} ({m/len(ks)*100:.1f}%) | "
                 f"{s}/{len(ks)} ({s/len(ks)*100:.1f}%) | {h} | {r} | {h-r:+d} | {p:.4f} |")
    if tot["n"]:
        p_all = mcnemar(tot["h"], tot["r"])
        L.append(f"| **总体** | **{tot['n']}** | **{tot['m']}/{tot['n']} ({tot['m']/tot['n']*100:.1f}%)** | "
                 f"**{tot['s']}/{tot['n']} ({tot['s']/tot['n']*100:.1f}%)** | **{tot['h']}** | **{tot['r']}** | "
                 f"**{tot['h']-tot['r']:+d}** | **{p_all:.4f}** |")

    # manipulation check + mis-assignment diagnostic
    deltas, tape_mean, cur_mean = [], [], []
    for k in keys:
        tr = sh[k]["selector_trace"]
        a = [y["actual_selected_count"] for y in tr if y.get("actual_selected_count") is not None]
        b = [y["current_state_matched_m"] for y in tr if y.get("current_state_matched_m") is not None]
        if a and b:
            deltas.append(float(np.mean(np.abs(np.array(a) - np.array(b)))))
            tape_mean.append(float(np.mean(a))); cur_mean.append(float(np.mean(b)))
    L += ["", "## 操纵检查", ""]
    if deltas:
        L += [f"- 每条 episode 的 |tape − 当前状态| 平均 = **{np.mean(deltas):.2f}**，范围 [{np.min(deltas):.1f}, {np.max(deltas):.1f}]",
              f"- tape 的 m 均值 = {np.mean(tape_mean):.2f}；当前状态自己算的 m 均值 = {np.mean(cur_mean):.2f}", ""]
    if keys:
        L += [f"- 超长步数（需扩展池）: {sum(1 for k in keys if sh[k].get('num_steps_from_extension_pool',0)>0)} 条 episode", ""]

    (root / f"{args.out_prefix}_REPORT.md").write_text("\n".join(L) + "\n")
    (root / f"{args.out_prefix}_RESULTS.json").write_text(json.dumps(
        {"n_pairs": tot["n"], "per_task": per_task, "overall": tot,
         "mean_mismatch": (float(np.mean(deltas)) if deltas else None)}, indent=1) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
