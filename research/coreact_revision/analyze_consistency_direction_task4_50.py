#!/usr/bin/env python3
"""Analyze the locked 50-state four-condition task-4 direction experiment."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
CONDITIONS=("vanilla","attention_toward","consistency_mask_only","consistency_toward")
COMPARISONS=(("consistency_toward","vanilla"),("consistency_toward","attention_toward"),("consistency_toward","consistency_mask_only"),("attention_toward","vanilla"),("consistency_mask_only","vanilla"))
def compare(by_pair,left,right,seed):
 values=np.asarray([int(pair[left]["success"])-int(pair[right]["success"]) for pair in by_pair.values()],dtype=float);lo=sum(pair[left]["success"] and not pair[right]["success"] for pair in by_pair.values());ro=sum(pair[right]["success"] and not pair[left]["success"] for pair in by_pair.values());n=lo+ro;rng=np.random.default_rng(seed);boot=np.asarray([values[rng.integers(0,len(values),len(values))].mean() for _ in range(2000)])
 return {"comparison":f"{left}_minus_{right}","point_estimate":float(values.mean()),"paired_bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],f"{left}_only_success":int(lo),f"{right}_only_success":int(ro),"discordant":int(n),"mcnemar_exact_p":float(binomtest(lo,n,.5).pvalue) if n else 1.0}
def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();art=a.artifact.resolve();manifest=[json.loads(x) for x in (art/"episode_manifest.jsonl").read_text().splitlines() if x.strip()]
 if len(manifest)!=200 or len({r["episode_id"] for r in manifest})!=200:raise RuntimeError("expected 200 unique manifest episodes")
 records=[];failures=[]
 for spec in manifest:
  path=art/"episodes"/spec["episode_id"]/"episode.json"
  if not path.exists():raise RuntimeError(f"missing {path}")
  row=json.loads(path.read_text());records.append(row)
  for key in ("pair_id","condition","init_state_id","reset_seed","action_noise_seed","language"):
   if row[key]!=spec[key]:failures.append([spec["episode_id"],"identity",key])
  for trace in row.get("replan_traces",[]):
   if sorted(trace["selected_indices"])!=sorted(trace["changed_indices"]):failures.append([spec["episode_id"],trace["replan"],"selected_changed"])
   if not trace["protected_tokens_untouched"] or not trace["all_output_finite"]:failures.append([spec["episode_id"],trace["replan"],"protected_or_finite"])
 by_pair={}
 for row in records:by_pair.setdefault(row["pair_id"],{})[row["condition"]]=row
 if len(by_pair)!=50 or any(set(pair)!=set(CONDITIONS) for pair in by_pair.values()):raise RuntimeError("expected 50 complete four-condition pairs")
 rows=[]
 for pair_id,pair in sorted(by_pair.items()):
  output={"pair_id":pair_id,"init_state_id":pair["vanilla"]["init_state_id"]}
  for condition in CONDITIONS:output[f"{condition}_success"]=int(pair[condition]["success"]);output[f"{condition}_steps"]=pair[condition]["control_steps"]
  rows.append(output)
 with (art/"paired_results.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rates={c:{"successes":sum(int(p[c]["success"]) for p in by_pair.values()),"episodes":50} for c in CONDITIONS}
 for value in rates.values():value["success_rate"]=value["successes"]/50
 comparisons=[compare(by_pair,l,r,8675311+i) for i,(l,r) in enumerate(COMPARISONS)]
 selection={}
 for condition in ("consistency_mask_only","consistency_toward"):
  counts=[trace["selected_count"] for row in records if row["condition"]==condition for trace in row["replan_traces"]];selection[condition]={"replans":len(counts),"mean_selected":float(np.mean(counts)),"median_selected":float(np.median(counts)),"abstention_fraction":float(np.mean(np.asarray(counts)==0)),"full_8_fraction":float(np.mean(np.asarray(counts)==8))}
 secondary={c:{"mean_steps":float(np.mean([r["control_steps"] for r in records if r["condition"]==c])),"mean_action_tv":float(np.mean([r["action_total_variation"] for r in records if r["condition"]==c])),"median_latency_s_per_replan":float(np.median([r["policy_seconds_median_per_replan"] for r in records if r["condition"]==c]))} for c in CONDITIONS}
 primary=comparisons[0];status="FAILED_INTEGRITY" if failures else ("CONSISTENCY_TOWARD_POSITIVE_DEVELOPMENT_SIGNAL" if primary["paired_bootstrap_95_ci"][0]>0 else "CONSISTENCY_TOWARD_HARMFUL" if primary["paired_bootstrap_95_ci"][1]<0 else "NO_CLEAR_CONSISTENCY_TOWARD_GAIN")
 result={"episodes":len(records),"paired_states":len(by_pair),"integrity_pass":not failures,"integrity_failures":failures,"success_rates":rates,"comparisons":comparisons,"selection_behavior":selection,"secondary":secondary,"decision":status,"scope":"task-4 development; prior outcomes informed direction test; not confirmation"};(art/"analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");(art/"decision.json").write_text(json.dumps({"status":status,"confirmation_claim_allowed":False},indent=2,sort_keys=True)+"\n")
 lines=["# Task-4 Consistency Direction Experiment","",f"Integrity: **{'PASS' if not failures else 'FAIL'}**; 200/200 episodes, 50/50 paired states.","","| Condition | Successes | Rate |","|---|---:|---:|"]+[f"| {c} | {rates[c]['successes']}/50 | {rates[c]['success_rate']:.3f} |" for c in CONDITIONS]+["","## Paired comparisons",""]+[f"- `{x['comparison']}`: {x['point_estimate']:+.3f}, 95% CI [{x['paired_bootstrap_95_ci'][0]:+.3f}, {x['paired_bootstrap_95_ci'][1]:+.3f}], McNemar p={x['mcnemar_exact_p']:.4g}." for x in comparisons]+["",f"Decision: **{status}**. This is development, not independent confirmation."];(art/"report.md").write_text("\n".join(lines)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
