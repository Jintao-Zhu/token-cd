#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCOPES=("full","front10","tail40","full_translation","full_rotation","full_gripper","front10_translation","front10_rotation","front10_gripper")


def bootstrap_task_stratified(state_rows,key,statistic,seed=20260816,replicates=10000):
    by_task=defaultdict(list)
    for row in state_rows:by_task[row["task_id"]].append(row[key])
    rng=np.random.default_rng(seed);draws=np.empty(replicates)
    for index in range(replicates):draws[index]=statistic(np.concatenate([rng.choice(np.asarray(by_task[task],dtype=float),len(by_task[task]),replace=True) for task in range(10)]))
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def summarize_state_values(rows):
    """Summarize repeated noise/timestep measurements without inflating N."""
    return {
        "median_alpha": float(np.median([row["alpha"] for row in rows])),
        "p_alpha_gt_1": float(np.mean([row["alpha"] > 1 for row in rows])),
        "median_orthogonal_ratio": float(np.median([row["orth"] for row in rows])),
        "mean_error_cosine": float(np.mean([row["cos"] for row in rows])),
    }


def aggregate_by_state(records,candidate,scope,flow_step=None,task_id=None):
    output=[]
    for record in records:
        if task_id is not None and record["state"]["task_id"] != task_id:
            continue
        points=[row for row in record["points"] if row["candidate"]==candidate and f"{scope}_alpha" in row and (flow_step is None or row["flow_step"]==flow_step)]
        if not points:
            continue
        repeated=[{"alpha":row[f"{scope}_alpha"],"orth":row[f"{scope}_orthogonal_ratio"],"cos":row[f"{scope}_error_cosine"]} for row in points]
        output.append({"task_id":record["state"]["task_id"],"state_id":record["state"]["state_id"],**summarize_state_values(repeated)})
    return output


def state_aggregate(records,candidate,scope):
    output=[]
    for record in records:
        rows=[row for row in record["points"] if row["candidate"]==candidate and f"{scope}_alpha" in row]
        output.append({"task_id":record["state"]["task_id"],"state_id":record["state"]["state_id"],"alpha_values":np.asarray([row[f"{scope}_alpha"] for row in rows]),"orthogonal_values":np.asarray([row[f"{scope}_orthogonal_ratio"] for row in rows]),"cosine_values":np.asarray([row[f"{scope}_error_cosine"] for row in rows])})
    return output


def summarize_scope(records,candidate,scope):
    states=state_aggregate(records,candidate,scope);flat_alpha=np.concatenate([row["alpha_values"] for row in states]);flat_orth=np.concatenate([row["orthogonal_values"] for row in states]);flat_cos=np.concatenate([row["cosine_values"] for row in states])
    alpha_state=[{"task_id":row["task_id"],"alpha":np.median(row["alpha_values"]),"p":np.mean(row["alpha_values"]>1),"orth":np.median(row["orthogonal_values"]),"cos":np.mean(row["cosine_values"])} for row in states]
    p_rows=[{"task_id":row["task_id"],"value":row["p"]} for row in alpha_state]
    orth_rows=[{"task_id":row["task_id"],"value":row["orth"]} for row in alpha_state]
    return {"candidate":candidate,"scope":scope,"state_count":len(states),"point_count":len(flat_alpha),"median_alpha":float(np.median(flat_alpha)),"mean_alpha":float(np.mean(flat_alpha)),"p_alpha_gt_1":float(np.mean(flat_alpha>1)),"median_orthogonal_ratio":float(np.median(flat_orth)),"mean_orthogonal_ratio":float(np.mean(flat_orth)),"median_error_cosine":float(np.median(flat_cos)),"mean_error_cosine":float(np.mean(flat_cos)),"state_median_alpha_ci95":bootstrap_task_stratified([{"task_id":row["task_id"],"value":row["alpha"]} for row in alpha_state],"value",np.median),"state_mean_p_alpha_gt_1":float(np.mean([row["p"] for row in alpha_state])),"state_mean_p_alpha_gt_1_ci95":bootstrap_task_stratified(p_rows,"value",np.mean),"state_median_orthogonal_ratio":float(np.median([row["orth"] for row in alpha_state])),"state_median_orthogonal_ratio_ci95":bootstrap_task_stratified(orth_rows,"value",np.median)}


def write_csv(path,rows):
    with path.open("w",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--artifact",type=Path,required=True);args=parser.parse_args();artifact=args.artifact.resolve();records=[json.loads(path.read_text()) for path in sorted((artifact/"states").glob("task*.json"))]
    if len(records)!=250 or any(not row["integrity"]["finite"] for row in records):raise RuntimeError(f"incomplete/invalid: {len(records)}")
    candidates=("weak_a_8l","weak_b_4l");scope_rows=[summarize_scope(records,candidate,scope) for candidate in candidates for scope in SCOPES];write_csv(artifact/"scope_summary.csv",scope_rows)
    task_rows=[];timestep_rows=[]
    for candidate in candidates:
        for task in range(10):
            for scope in ("full","front10"):
                states=aggregate_by_state(records,candidate,scope,task_id=task)
                task_rows.append({"candidate":candidate,"task_id":task,"scope":scope,"states":len(states),"median_alpha":np.median([row["median_alpha"] for row in states]),"p_alpha_gt_1":np.mean([row["p_alpha_gt_1"] for row in states]),"median_orthogonal_ratio":np.median([row["median_orthogonal_ratio"] for row in states]),"mean_error_cosine":np.mean([row["mean_error_cosine"] for row in states])})
        for step in range(10):
            for scope in ("full","front10"):
                states=aggregate_by_state(records,candidate,scope,flow_step=step)
                timestep_rows.append({"candidate":candidate,"flow_step":step,"timestep":1-.1*step,"scope":scope,"states":len(states),"median_alpha":np.median([row["median_alpha"] for row in states]),"p_alpha_gt_1":np.mean([row["p_alpha_gt_1"] for row in states]),"median_orthogonal_ratio":np.median([row["median_orthogonal_ratio"] for row in states]),"mean_error_cosine":np.mean([row["mean_error_cosine"] for row in states])})
    write_csv(artifact/"per_task.csv",task_rows);write_csv(artifact/"per_timestep.csv",timestep_rows)
    position_rows=[]
    for candidate in candidates:
        for position in range(50):
            values=[row for record in records for row in record["positions"] if row["candidate"]==candidate and row["position"]==position]
            if values:position_rows.append({"candidate":candidate,"position":position,"states":len(values),"median_alpha":np.median([row["median_alpha"] for row in values]),"mean_alpha":np.mean([row["mean_alpha"] for row in values]),"p_alpha_gt_1":np.mean([row["p_alpha_gt_1"] for row in values]),"median_orthogonal_ratio":np.median([row["median_orthogonal_ratio"] for row in values]),"mean_error_cosine":np.mean([row["mean_error_cosine"] for row in values])})
    write_csv(artifact/"per_chunk_position.csv",position_rows)
    figure,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for candidate,label in (("weak_a_8l","8L"),("weak_b_4l","4L")):
        rows=[row for row in position_rows if row["candidate"]==candidate];axes[0].plot([row["position"] for row in rows],[row["median_alpha"] for row in rows],label=label);axes[1].plot([row["position"] for row in rows],[row["median_orthogonal_ratio"] for row in rows],label=label)
    axes[0].axhline(1,color="gray",linestyle="--");axes[0].set_ylabel("Median alpha");axes[1].set_ylabel("Median orthogonal ratio");axes[1].set_xlabel("Action chunk position");axes[0].legend();axes[1].legend();figure.tight_layout();figure.savefig(artifact/"figures/error_decomposition_by_position.png",dpi=180);plt.close(figure)
    primary={row["candidate"]:{key:row[key] for key in ("median_alpha","mean_alpha","p_alpha_gt_1","median_orthogonal_ratio","mean_orthogonal_ratio","median_error_cosine","mean_error_cosine","state_median_alpha_ci95")} for row in scope_rows if row["scope"]=="full"}
    front10={row["candidate"]:{key:row[key] for key in ("median_alpha","mean_alpha","p_alpha_gt_1","median_orthogonal_ratio","mean_orthogonal_ratio","median_error_cosine","mean_error_cosine","state_median_alpha_ci95")} for row in scope_rows if row["scope"]=="front10"}
    report={"status":"ERROR_DECOMPOSITION_COMPLETE","states":250,"primary_full_chunk":primary,"executed_front10":front10,"interpretation_guard":"This diagnoses error geometry relative to single-demo u only; it does not resolve action-manifold target validity."};(artifact/"result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");(artifact/"status/current.json").write_text(json.dumps({"stage":"COMPLETE","status":"ERROR_DECOMPOSITION_COMPLETE"},indent=2)+"\n");print(json.dumps(report,sort_keys=True))


if __name__=="__main__":main()
