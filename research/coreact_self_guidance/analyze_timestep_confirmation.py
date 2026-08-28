from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest


ARMS=("A_vanilla","N0_pure_negative","W05_shrink","W15_extrapolate","W20_extrapolate","REF_toward_top8")
SHIFT_ARMS=("N0_pure_negative","W05_shrink","W15_extrapolate","W20_extrapolate")


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def comparison(rows,arm,seed):
    a=np.asarray([int(x[arm]["success"]) for x in rows]); b=np.asarray([int(x["A_vanilla"]["success"]) for x in rows]); d=a-b; rng=np.random.default_rng(seed); boot=[float(np.mean(d[rng.integers(0,len(d),len(d))])) for _ in range(2000)]; x=int(((a==1)&(b==0)).sum()); y=int(((a==0)&(b==1)).sum()); n=x+y
    return {"comparison":f"{arm}_minus_A_vanilla","point_estimate":float(d.mean()),"paired_bootstrap_95_ci":np.quantile(boot,[.025,.975]).tolist(),"arm_only_success":x,"vanilla_only_success":y,"discordant":n,"exact_mcnemar_p":float(binomtest(x,n,.5).pvalue) if n else 1.0}


def holm(comps):
    order=sorted(comps,key=lambda k:comps[k]["exact_mcnemar_p"]); running=0.0
    for i,k in enumerate(order): running=max(running,min(1.0,(len(order)-i)*comps[k]["exact_mcnemar_p"])); comps[k]["holm_adjusted_p"]=running


def contains_zero(c): return c["paired_bootstrap_95_ci"][0] <= 0 <= c["paired_bootstrap_95_ci"][1]


def classify(comps):
    n0=comps["N0_pure_negative"]; w05=comps["W05_shrink"]; w15=comps["W15_extrapolate"]; w20=comps["W20_extrapolate"]; best=max((w15,w20),key=lambda x:x["point_estimate"])
    if n0["paired_bootstrap_95_ci"][1] < 0 and best["paired_bootstrap_95_ci"][0] > 0: return "STRONG_CONTRASTIVE"
    if n0["point_estimate"] < 0 and best["point_estimate"] > 0: return "DIRECTIONAL_CONTRASTIVE"
    if contains_zero(n0) and w05["paired_bootstrap_95_ci"][0] > 0 and w15["paired_bootstrap_95_ci"][0] <= 0 and w20["paired_bootstrap_95_ci"][0] <= 0: return "STRONG_SHRINKAGE"
    if contains_zero(n0) and w05["point_estimate"] > max(w15["point_estimate"],w20["point_estimate"],0): return "DIRECTIONAL_SHRINKAGE"
    if all(contains_zero(comps[x]) and abs(comps[x]["point_estimate"]) < .10 for x in SHIFT_ARMS): return "NULL"
    return "MIXED"


def cosine(a,b):
    a=a.flatten().double(); b=b.flatten().double(); denom=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return float(torch.dot(a,b)/denom) if float(denom)>0 else None


def main():
    p=argparse.ArgumentParser(); p.add_argument("--artifact",type=Path,required=True); args=p.parse_args(); art=args.artifact.resolve(); protocol=__import__("yaml").safe_load((art/"protocol.lock.yaml").read_text()); task_ids=[int(x["id"]) for x in protocol["tasks"]]; specs=[json.loads(x) for x in (art/"episode_manifest.jsonl").read_text().splitlines()]; files=list((art/"episodes").glob("*.json")); expected={x["episode_id"] for x in specs}
    if len(specs)!=1200 or len(files)!=1200 or {x.stem for x in files}!=expected: raise RuntimeError("analysis requires exact 1200/1200")
    grouped=defaultdict(dict); failures=[]; sidecar_count=0
    for f in files:
        r=json.loads(f.read_text()); grouped[r["pair_id"]][r["arm"]]=r
        if r["status"]!="complete" or not r["all_actions_finite"]: failures.append([r["episode_id"],"status/finite"])
        if r["arm"]!="A_vanilla":
            side=art/r["correction_vector_path"]
            if not side.exists() or sha256(side)!=r["correction_vector_sha256"]: failures.append([r["episode_id"],"correction sidecar/hash"]); continue
            data=torch.load(side,weights_only=True,map_location="cpu"); raw=data["raw_clean_minus_branch"]; applied=data["actual_applied"]; sidecar_count+=1
            if list(raw.shape)!=r["correction_vector_shape"] or raw.shape[1:]!=(10,1,50,7) or raw.shape!=applied.shape or not torch.isfinite(raw).all() or not torch.isfinite(applied).all(): failures.append([r["episode_id"],"correction shape/finite"])
    pairs=[]
    for pair,arms in sorted(grouped.items()):
        if set(arms)!=set(ARMS): failures.append([pair,"arms"]); continue
        ref=arms["A_vanilla"]; keys=("suite","task_id","init_state_id","reset_seed","action_noise_seed","selection_seed","language")
        if any(arms[a][k]!=ref[k] for a in ARMS for k in keys): failures.append([pair,"identity"]); continue
        if len({arms[a]["initial_sim_state_sha256"] for a in ARMS})!=1 or len({arms[a]["initial_prepared_input_sha256"] for a in ARMS})!=1: failures.append([pair,"state/input"]); continue
        m=min(arms[a]["replans"] for a in ARMS); noise=ref["first_noise_sha256_by_replan"][:m]
        if any(arms[a]["first_noise_sha256_by_replan"][:m]!=noise for a in ARMS): failures.append([pair,"noise"]); continue
        pairs.append({"pair_id":pair,**arms})
    if len(pairs)!=200 or sidecar_count!=1000 or failures:
        (art/"analysis_integrity_failure.json").write_text(json.dumps({"pairs":len(pairs),"sidecars":sidecar_count,"failures":failures},indent=2)+"\n"); raise RuntimeError("analysis integrity failure")
    results={}; flat=[]; cosine_rows=[]
    for task in task_ids:
        rows=[x for x in pairs if int(x["A_vanilla"]["task_id"])==task]; comps={arm:comparison(rows,arm,20260810+task*100+i) for i,arm in enumerate((*SHIFT_ARMS,"REF_toward_top8"))}; holm({a:comps[a] for a in SHIFT_ARMS}); rates={arm:float(np.mean([x[arm]["success"] for x in rows])) for arm in ARMS}; regime=classify(comps); results[str(task)]={"states":len(rows),"success_rates":rates,"comparisons":comps,"regime":regime,"monotonic_support":comps["W20_extrapolate"]["point_estimate"]>=comps["W15_extrapolate"]["point_estimate"]}
        for row in rows:
            out={"pair_id":row["pair_id"],"task_id":task,"init_state_id":row["A_vanilla"]["init_state_id"]}
            for arm in ARMS: out[f"{arm}_success"]=int(row[arm]["success"])
            flat.append(out)
            ref=torch.load(art/row["REF_toward_top8"]["correction_vector_path"],weights_only=True,map_location="cpu")["raw_clean_minus_branch"][0]
            for arm in SHIFT_ARMS:
                shifted=torch.load(art/row[arm]["correction_vector_path"],weights_only=True,map_location="cpu")["raw_clean_minus_branch"][0]
                for step in range(10): cosine_rows.append({"pair_id":row["pair_id"],"task_id":task,"arm":arm,"flow_step":step,"first_replan_ref_cosine":cosine(ref[step],shifted[step]),"shift_consecutive_step_cosine":cosine(shifted[step-1],shifted[step]) if step else None})
    with (art/"paired_results.csv").open("x",newline="") as f: w=csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    with (art/"correction_cosines.csv").open("x",newline="") as f: w=csv.DictWriter(f,fieldnames=list(cosine_rows[0])); w.writeheader(); w.writerows(cosine_rows)
    with (art/"missingness.csv").open("x",newline="") as f: csv.DictWriter(f,fieldnames=("episode_id","reason")).writeheader()
    cosine_summary={}
    for task in task_ids:
        cosine_summary[str(task)]={}
        for arm in SHIFT_ARMS:
            vals=[x["first_replan_ref_cosine"] for x in cosine_rows if x["task_id"]==task and x["arm"]==arm and x["first_replan_ref_cosine"] is not None]
            cosine_summary[str(task)][arm]={"median":float(np.median(vals)),"mean":float(np.mean(vals)),"n":len(vals)}
    summary={"experiment_name":protocol["experiment_name"],"stage":protocol["stage"],"completion":{"episodes":1200,"paired_units":200,"correction_sidecars":1000,"missing":0,"duplicates":0},"locked_shift":protocol["locked_shift"],"task_results":results,"first_replan_ref_vs_shift_cosine":cosine_summary,"regime_counts":dict(__import__("collections").Counter(x["regime"] for x in results.values())),"confirmation_scope":protocol["confirmation_scope"],"decision":"REGIME_REPLICATION_COMPLETE","integrity_pass":True}
    (art/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); (art/"decision.json").write_text(json.dumps({"decision":summary["decision"],"integrity_pass":True,"regime_counts":summary["regime_counts"],"confirmation_scope":protocol["confirmation_scope"]},indent=2)+"\n"); (art/"report.md").write_text("# Timestep Self-Guidance New-Task Regime Replication\n\nMechanism-heldout task replication; benchmark tasks may be historically burned.\n\n```json\n"+json.dumps(results,indent=2,sort_keys=True)+"\n```\n\nCorrection cosine summary:\n\n```json\n"+json.dumps(cosine_summary,indent=2,sort_keys=True)+"\n```\n",encoding="utf-8")
    audit={str(x.relative_to(art)):sha256(x) for x in sorted(art.rglob("*")) if x.is_file() and x.name!="sha256_audit.json"}; (art/"sha256_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); (art/"status/analysis.complete").touch(exist_ok=False); print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":main()
