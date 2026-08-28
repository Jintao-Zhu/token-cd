#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,spearmanr

CONDITIONS=("vanilla","mask_strong_positive","mask_strong_negative","mask_near_zero")
def boot(deltas,seed=918273,n=2000):
 rng=np.random.default_rng(seed);d=np.asarray(deltas,float);vals=[float(np.mean(d[rng.integers(0,len(d),len(d))])) for _ in range(n)]
 return {"replicates":n,"mean":float(np.mean(d)),"ci95":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();rows={}
 for path in (o/"episodes").glob("*/episode.json"):
  r=json.loads(path.read_text());rows[(r["pair_id"],r["condition"])]=r
 pairs=sorted({k[0] for k in rows});missing=[]
 for pair in pairs:
  if any((pair,c) not in rows for c in CONDITIONS):missing.append(pair)
 if len(pairs)!=50 or missing:raise RuntimeError(f"paired completeness failed {len(pairs)} missing={missing}")
 integrity_failures=[]
 for pair in pairs:
  records=[rows[(pair,c)] for c in CONDITIONS]
  if len({r["reset_seed"] for r in records})!=1 or len({r["action_noise_seed"] for r in records})!=1:integrity_failures.append(f"{pair}: paired seed mismatch")
  for r in records:
   if r["status"]!="complete" or len((o/r["step_log"]).read_text().splitlines())!=r["control_steps"]:integrity_failures.append(f"{r['episode_id']}: incomplete step log")
   if r["condition"]!="vanilla":
    for t in r["selection_traces"]:
     if len(t["selected_indices"])!=8 or sorted(t["selected_indices"])!=sorted(t["changed_indices"]) or len(t["candidate_indices"])!=32 or not np.isfinite(t["candidate_utilities"]).all():integrity_failures.append(f"{r['episode_id']}: invalid selection trace")
 integrity={"pass":not integrity_failures,"episodes":len(rows),"complete_pairs":len(pairs),"failures":integrity_failures}
 (o/"post_integrity_report.json").write_text(json.dumps(integrity,indent=2,sort_keys=True)+"\n")
 if integrity_failures:raise RuntimeError(str(integrity_failures[:5]))
 result=[];summary={}
 for c in CONDITIONS:
  y=np.asarray([int(rows[(pair,c)]["success"]) for pair in pairs]);summary[c]={"n":50,"successes":int(y.sum()),"success_rate":float(y.mean())}
 for c in CONDITIONS[1:]:
  v=np.asarray([rows[(pair,"vanilla")]["success"] for pair in pairs],int);m=np.asarray([rows[(pair,c)]["success"] for pair in pairs],int);delta=m-v
  n10=int(((v==1)&(m==0)).sum());n01=int(((v==0)&(m==1)).sum());summary[c]["minus_vanilla"]={"point":float(delta.mean()),"bootstrap":boot(delta),"vanilla_only_success":n10,"condition_only_success":n01,"mcnemar_exact_two_sided":float(binomtest(n01,n10+n01).pvalue) if n10+n01 else 1.0}
 for pair in pairs:
  base=rows[(pair,"vanilla")]
  for c in CONDITIONS[1:]:
   trace=rows[(pair,c)].get("selection_traces",[]); utility=None
   if trace:
    t=trace[0];candidate=dict(zip(t["candidate_indices"],t["candidate_utilities"]));utility=float(np.mean([candidate[i] for i in t["selected_indices"]]))
   result.append({"pair_id":pair,"condition":c,"vanilla_success":int(base["success"]),"mask_success":int(rows[(pair,c)]["success"]),"causal_harm":int(base["success"])-int(rows[(pair,c)]["success"]),"group_mean_utility":utility})
 for c in CONDITIONS[1:]:
  x=[r["group_mean_utility"] for r in result if r["condition"]==c and r["group_mean_utility"] is not None];y=[r["causal_harm"] for r in result if r["condition"]==c and r["group_mean_utility"] is not None];summary[c]["utility_vs_binary_causal_harm_spearman"]={"rho":float(spearmanr(x,y).statistic),"pvalue":float(spearmanr(x,y).pvalue),"n":len(x)}
 all_x=[r["group_mean_utility"] for r in result if r["group_mean_utility"] is not None];all_y=[r["causal_harm"] for r in result if r["group_mean_utility"] is not None];summary["all_groups_utility_vs_causal_harm_spearman"]={"rho":float(spearmanr(all_x,all_y).statistic),"pvalue":float(spearmanr(all_x,all_y).pvalue),"n":len(all_x)}
 negative=np.asarray([rows[(pair,"mask_strong_negative")]["success"] for pair in pairs],int);positive=np.asarray([rows[(pair,"mask_strong_positive")]["success"] for pair in pairs],int);summary["mask_strong_negative_minus_positive"]={"point":float(np.mean(negative-positive)),"bootstrap":boot(negative-positive)}
 with (o/"paired_results.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
 summary["complete_pairs"]=len(pairs)
 pos=summary["mask_strong_positive"]["minus_vanilla"]["bootstrap"]["ci95"][1]<0
 neg=summary["mask_strong_negative_minus_positive"]["bootstrap"]["ci95"][0]>0
 decision="CAUSAL_SIGN_VALIDATED" if pos and neg else "VALUE_PROBE_NOT_CAUSALLY_VALIDATED"
 (o/"analysis.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");(o/"decision.json").write_text(json.dumps({"decision":decision,"positive_mask_worse_than_vanilla":pos,"negative_mask_better_than_positive":neg,"guidance_permitted":False},indent=2,sort_keys=True)+"\n")
 print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__":main()
