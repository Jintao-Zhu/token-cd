#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
CONDS=("vanilla","attention_top8","random8","bottom8")
def cluster_boot(rows,condition,n=2000,seed=731902):
 rng=np.random.default_rng(seed);tasks=sorted({r["task_id"] for r in rows});vals=[]
 by={(r["task_id"],r["init_state_id"]):r for r in rows}
 for _ in range(n):
  chosen=rng.choice(tasks,len(tasks),replace=True);d=[]
  for t in chosen:
   ids=sorted(i for tt,i in by if tt==t);sample=rng.choice(ids,len(ids),replace=True)
   d.extend(int(by[(t,int(i))][condition]["success"])-int(by[(t,int(i))]["vanilla"]["success"]) for i in sample)
  vals.append(np.mean(d))
 return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def jaccard(traces):
 sets=[set(t["selected_indices"]) for t in traces]
 if len(sets)<2:return None
 return float(np.mean([len(a&b)/len(a|b) for a,b in zip(sets[:-1],sets[1:])]))
def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();manifest=[json.loads(x) for x in (o/"episode_manifest.jsonl").read_text().splitlines()];records={};fail=[]
 for s in manifest:
  path=o/"episodes"/s["episode_id"]/"episode.json"
  if not path.exists():fail.append(f"missing {s['episode_id']}");continue
  r=json.loads(path.read_text());records[(s["pair_id"],s["condition"])]=r
  steps=(o/r["step_log"]).read_text().splitlines() if (o/r["step_log"]).exists() else []
  if r["status"]!="complete" or len(steps)!=r["control_steps"]:fail.append(f"invalid {s['episode_id']}")
  expected=min(r["replans"],3) if s["condition"]!="vanilla" else 0
  if len(r["selection_traces"])!=expected:fail.append(f"trace count {s['episode_id']}")
  for t in r["selection_traces"]:
   if len(t["selected_indices"])!=8 or sorted(t["selected_indices"])!=sorted(t["changed_indices"]) or not t["all_output_finite"]:fail.append(f"trace invalid {s['episode_id']}")
 pairs=sorted({s["pair_id"] for s in manifest});grouped=[]
 for pair in pairs:
  rs={c:records.get((pair,c)) for c in CONDS}
  if any(x is None for x in rs.values()):fail.append(f"incomplete pair {pair}");continue
  if len({x["reset_seed"] for x in rs.values()})!=1 or len({x["action_noise_seed"] for x in rs.values()})!=1:fail.append(f"seed mismatch {pair}")
  grouped.append({"pair_id":pair,"task_id":rs["vanilla"]["task_id"],"init_state_id":rs["vanilla"]["init_state_id"],**rs})
 integrity={"pass":not fail and len(records)==600 and len(grouped)==150,"episodes":len(records),"pairs":len(grouped),"failures":fail};(o/"post_integrity_report.json").write_text(json.dumps(integrity,indent=2,sort_keys=True)+"\n")
 if not integrity["pass"]:raise SystemExit(1)
 summary={"rates":{},"comparisons":{},"mechanism":{}}
 for c in CONDS:
  y=np.array([r[c]["success"] for r in grouped],int);summary["rates"][c]={"successes":int(y.sum()),"n":len(y),"rate":float(y.mean()),"by_task":{str(t):float(np.mean([r[c]["success"] for r in grouped if r["task_id"]==t])) for t in (0,1,2)},"mean_steps":float(np.mean([r[c]["control_steps"] for r in grouped])),"mean_action_tv":float(np.mean([r[c]["action_total_variation"] for r in grouped])),"mean_chunk_discontinuity":float(np.mean([r[c]["chunk_boundary_discontinuity"] for r in grouped])),"mean_eef_displacement":float(np.mean([r[c]["eef_displacement"] for r in grouped]))}
 v=np.array([r["vanilla"]["success"] for r in grouped],int)
 for c in CONDS[1:]:
  m=np.array([r[c]["success"] for r in grouped],int);d=m-v;n01=int(((v==0)&(m==1)).sum());n10=int(((v==1)&(m==0)).sum());task_diff={str(t):float(np.mean([int(r[c]["success"])-int(r["vanilla"]["success"]) for r in grouped if r["task_id"]==t])) for t in (0,1,2)};summary["comparisons"][c]={"minus_vanilla":float(d.mean()),"cluster_bootstrap_ci95":cluster_boot(grouped,c),"rescued":n01,"harmed":n10,"net_rescued":n01-n10,"mcnemar_p":float(binomtest(n01,n01+n10).pvalue) if n01+n10 else 1.,"task_differences":task_diff};summary["mechanism"][c]={"mean_mask_clean_action_norm":float(np.mean([x for r in grouped for x in r[c]["mask_action_delta_norms"]])),"mean_token_jaccard":float(np.mean([x for r in grouped if (x:=jaccard(r[c]["selection_traces"])) is not None]))}
 comps=summary["comparisons"];allpos=all(comps[c]["minus_vanilla"]>0 and sum(x>0 for x in comps[c]["task_differences"].values())>=2 for c in CONDS[1:]);strong=allpos and all(comps[c]["cluster_bootstrap_ci95"][0]>0 for c in CONDS[1:]);top=comps["attention_top8"]["cluster_bootstrap_ci95"][0]>0 and comps["random8"]["minus_vanilla"]<=0 and comps["bottom8"]["minus_vanilla"]<=0;near=all(abs(comps[c]["minus_vanilla"])<=.05 and comps[c]["cluster_bootstrap_ci95"][0]<=0<=comps[c]["cluster_bootstrap_ci95"][1] for c in CONDS[1:]);decision="GENERIC_MASKING_STRONG" if strong else "GENERIC_MASKING_DIRECTIONAL_ONLY" if allpos else "TOP_SELECTOR_SPECIFIC" if top else "TASK8_SIGNAL_NOT_REPLICATED" if near else "MIXED_OR_INCONCLUSIVE";summary["decision"]=decision;(o/"analysis.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");(o/"decision.json").write_text(json.dumps({"decision":decision,"confirmation_claim_allowed":False},indent=2)+"\n")
 out=[]
 for r in grouped:out.append({"pair_id":r["pair_id"],"task_id":r["task_id"],**{f"{c}_success":int(r[c]["success"]) for c in CONDS}})
 with (o/"paired_results.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
 print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__":main()
