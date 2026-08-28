#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("single_g_positive", "marginal_g_positive", "nearest_g_positive", "delta_d_raw_positive", "delta_d_applied_positive")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def state_rows(records, candidate, flow_step=None):
    output=[]
    for record in records:
        rows=[row for row in record["rows"] if row["candidate"]==candidate and (flow_step is None or row["flow_step"]==flow_step)]
        output.append({"task_id":record["state"]["task_id"],"state_id":record["state"]["state_id"],**{key:float(np.mean([row[key] for row in rows])) for key in METRICS},"marginal_cosine":float(np.mean([row["marginal_cosine"] for row in rows])),"nearest_cosine":float(np.mean([row["nearest_cosine"] for row in rows])),"posterior_ess":float(np.mean([row["posterior_ess"] for row in rows])),"posterior_max_weight":float(np.mean([row["posterior_max_weight"] for row in rows]))})
    return output


def bootstrap(rows, key, replicates=10000, seed=20260816):
    by_task=defaultdict(list)
    for row in rows: by_task[row["task_id"]].append(row[key])
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(replicates): draws.append(np.mean(np.concatenate([rng.choice(by_task[t],len(by_task[t]),replace=True) for t in sorted(by_task)])))
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def summarize(rows, candidate, scope):
    out={"candidate":candidate,"scope":scope,"states":len(rows)}
    for key in METRICS:
        out[key]=float(np.mean([row[key] for row in rows]));out[f"{key}_ci95"]=bootstrap(rows,key)
    for key in ("marginal_cosine","nearest_cosine","posterior_ess","posterior_max_weight"):out[key]=float(np.mean([row[key] for row in rows]))
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--artifact",type=Path,required=True);args=parser.parse_args();artifact=args.artifact.resolve()
    records=[json.loads(path.read_text()) for path in sorted((artifact/"states").glob("*.json"))]
    if len(records)!=250 or any(not row["integrity"]["finite"] or row["integrity"]["rows"]!=60 for row in records):raise RuntimeError(f"incomplete/invalid: {len(records)}")
    candidates=("weak_a_8l","weak_b_4l")
    overall=[summarize(state_rows(records,c),c,"overall") for c in candidates];write_csv(artifact/"overall.csv",overall)
    task=[];timestep=[]
    for c in candidates:
        states=state_rows(records,c)
        for t in range(10):task.append(summarize([r for r in states if r["task_id"]==t],c,f"task_{t}"))
        for step in range(10):timestep.append(summarize(state_rows(records,c,step),c,f"flow_step_{step}"))
    write_csv(artifact/"per_task.csv",task);write_csv(artifact/"per_timestep.csv",timestep)
    decisions=[]
    for row in overall:
        single=row["single_g_positive"]
        manifold=max(row["marginal_g_positive"],row["delta_d_applied_positive"])
        if manifold-single>=.10 and manifold>=.60:decision="SINGLE_DEMO_TARGET_MISSES_ACTION_MULTIMODALITY"
        elif row["marginal_g_positive"]<=.50 and row["delta_d_applied_positive"]<=.50:decision="ACTION_MULTIMODALITY_DOES_NOT_EXPLAIN_AG_FAILURE"
        else:decision="INCONCLUSIVE_ACTION_MANIFOLD_DIAGNOSIS"
        decisions.append({"candidate":row["candidate"],"decision":decision})
    final="SINGLE_DEMO_TARGET_MISSES_ACTION_MULTIMODALITY" if any(x["decision"]=="SINGLE_DEMO_TARGET_MISSES_ACTION_MULTIMODALITY" for x in decisions) else "ACTION_MULTIMODALITY_DOES_NOT_EXPLAIN_AG_FAILURE" if all(x["decision"]=="ACTION_MULTIMODALITY_DOES_NOT_EXPLAIN_AG_FAILURE" for x in decisions) else "INCONCLUSIVE_ACTION_MANIFOLD_DIAGNOSIS"
    result={"status":"ACTION_MANIFOLD_DIAGNOSIS_COMPLETE","states":250,"k":16,"overall":overall,"candidate_decisions":decisions,"decision":final}
    (artifact/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");(artifact/"status/current.json").write_text(json.dumps({"stage":"COMPLETE","decision":final},indent=2)+"\n");print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
