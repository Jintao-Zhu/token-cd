"""Paired analysis for correct-query vs wrong-query/random-cluster budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
SHORT = {t: t.replace("google_robot_", "") for t in TASKS}
ARMS = ("wrong_entity", "random_cluster")


def load(root: Path, task: str, arm: str, seed: int):
    p = root / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def paired(root: Path, matched_root: Path, task: str, arm: str):
    rows = []
    for seed in range(100, 200):
        a = load(matched_root, task, "l11_matched", seed)
        b = load(root, task, arm, seed)
        if a is None or b is None:
            continue
        rows.append({
            "seed": seed,
            "matched_success": bool(a["success"]), "arm_success": bool(b["success"]),
            "matched_m": float(a.get("mean_selected_token_count", 0.0)),
            "arm_m": float(b.get("mean_selected_token_count", 0.0)),
        })
    if not rows:
        return None
    harm = sum(r["matched_success"] and not r["arm_success"] for r in rows)
    rescue = sum(r["arm_success"] and not r["matched_success"] for r in rows)
    p = stats.binomtest(min(harm, rescue), harm + rescue, 0.5).pvalue if harm + rescue else 1.0
    mm = np.asarray([r["matched_m"] for r in rows], float)
    am = np.asarray([r["arm_m"] for r in rows], float)
    rho, _ = stats.spearmanr(mm, am) if len(set(mm)) > 1 and len(set(am)) > 1 else (float("nan"), float("nan"))
    return {
        "n": len(rows), "matched": sum(r["matched_success"] for r in rows),
        "arm": sum(r["arm_success"] for r in rows), "harm": int(harm), "rescue": int(rescue),
        "net": int(harm - rescue), "p": float(p),
        "mean_matched_m": float(mm.mean()), "mean_arm_m": float(am.mean()),
        "std_arm_m": float(am.std()), "median_arm_m": float(np.median(am)),
        "p10_arm_m": float(np.percentile(am, 10)), "p90_arm_m": float(np.percentile(am, 90)),
        "mean_abs_m_diff": float(np.mean(np.abs(mm - am))), "rho_seed_m": float(rho),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--matched-artifact", type=Path, required=True)
    args = ap.parse_args()
    root = args.artifact.resolve(); matched = args.matched_artifact.resolve()
    out = {"per_task": {}, "overall": {}}
    lines = ["# Correct-query vs wrong-query/random-cluster", ""]
    for arm in ARMS:
        lines += [f"## {arm}", "", "| Task | n | Matched | Arm | Harm | Rescue | Net | p | matched m | arm m | mean abs diff | rho |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        pooled = []
        for task in TASKS:
            st = paired(root, matched, task, arm)
            if st is None: continue
            out["per_task"].setdefault(arm, {})[task] = st
            lines.append(f"| {SHORT[task]} | {st['n']} | {st['matched']} | {st['arm']} | {st['harm']} | {st['rescue']} | {st['net']:+d} | {st['p']:.4f} | {st['mean_matched_m']:.2f} | {st['mean_arm_m']:.2f} | {st['mean_abs_m_diff']:.2f} | {st['rho_seed_m']:.3f} |")
            pooled.extend([(st['matched'], st['arm'], st['harm'], st['rescue'], st['mean_matched_m']*st['n'], st['mean_arm_m']*st['n'])])
        # pooled directly from summaries
        allrows=[]
        for task in TASKS:
            for seed in range(100,200):
                a=load(matched,task,'l11_matched',seed); b=load(root,task,arm,seed)
                if a and b: allrows.append((bool(a['success']),bool(b['success']),float(a.get('mean_selected_token_count',0)),float(b.get('mean_selected_token_count',0))))
        if allrows:
            h=sum(x and not y for x,y,_,_ in allrows); r=sum(y and not x for x,y,_,_ in allrows)
            p=stats.binomtest(min(h,r),h+r,0.5).pvalue if h+r else 1.0
            ms=np.array([x for _,_,x,_ in allrows]); aa=np.array([y for _,_,_,y in allrows])
            rho=stats.spearmanr(ms,aa).statistic if len(set(ms))>1 and len(set(aa))>1 else float('nan')
            stat={"n":len(allrows),"matched":sum(x for x,_,_,_ in allrows),"arm":sum(y for _,y,_,_ in allrows),"harm":int(h),"rescue":int(r),"net":int(h-r),"p":float(p),"mean_matched_m":float(ms.mean()),"mean_arm_m":float(aa.mean()),"std_arm_m":float(aa.std()),"mean_abs_m_diff":float(np.mean(np.abs(ms-aa))),"rho_seed_m":float(rho)}
            out["overall"][arm]=stat
            lines.append(f"| **OVERALL** | {stat['n']} | {stat['matched']} | {stat['arm']} | {stat['harm']} | {stat['rescue']} | {stat['net']:+d} | {stat['p']:.4f} | {stat['mean_matched_m']:.2f} | {stat['mean_arm_m']:.2f} | {stat['mean_abs_m_diff']:.2f} | {stat['rho_seed_m']:.3f} |")
        lines.append("")
    (root / "QUERY_PROVENANCE_RESULTS.json").write_text(json.dumps(out, indent=2) + "\n")
    (root / "QUERY_PROVENANCE_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
