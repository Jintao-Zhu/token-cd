#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

BRANCHES=("W1_skip_last_1","W2_skip_last_2");WINDOWS={"last_2":([8,9],[0,1]),"last_3":([7,8,9],[0,1,2]),"last_4":([6,7,8,9],[0,1,2,3])};RATES=("marginal_g_positive","nearest_g_positive","delta_d_raw_positive","delta_d_applied_positive","clipped");CONT=("marginal_cosine","nearest_cosine","weak_to_strong_manifold_distance_ratio","weak_to_strong_marginal_error_ratio","direction_relative_norm","clip_scale","posterior_ess")

def write(path,rows):
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def states(records,branch,steps):
 out=[]
 for rec in records:
  rs=[r for r in rec["rows"] if r["candidate"]==branch and r["flow_step"] in steps]
  out.append({"task_id":rec["state"]["task_id"],"state_id":rec["state"]["state_id"],**{k:float(np.mean([r[k] for r in rs])) for k in RATES+CONT}})
 return out

def boot(rows,key,seed=20260817,n=10000):
 by=defaultdict(list)
 for r in rows:by[r["task_id"]].append(r[key])
 rng=np.random.default_rng(seed);draw=np.empty(n)
 for i in range(n):draw[i]=np.mean(np.concatenate([rng.choice(by[t],len(by[t]),replace=True) for t in sorted(by)]))
 return [float(np.quantile(draw,.025)),float(np.quantile(draw,.975))]

def summary(rows,branch,scope):
 x={"candidate":branch,"scope":scope,"states":len(rows)}
 for k in RATES:x[k]=float(np.mean([r[k] for r in rows]));x[k+"_ci95"]=boot(rows,k)
 for k in CONT:x[k]=float(np.mean([r[k] for r in rows]));x["median_"+k]=float(np.median([r[k] for r in rows]))
 return x

def parent_comparison(parent,steps):
 records=[json.loads(p.read_text()) for p in sorted((parent/"states").glob("*.json"))];out=[]
 for candidate,label in (("weak_a_8l","8L_independent"),("weak_b_4l","4L_independent")):
  full=[];low=[]
  for rec in records:
   a=[r for r in rec["rows"] if r["candidate"]==candidate];b=[r for r in a if r["flow_step"] in steps]
   full.append(np.mean([r["marginal_g_positive"] for r in a]));low.append({k:np.mean([r[k] for r in b]) for k in ("marginal_g_positive","nearest_g_positive","delta_d_applied_positive")})
  out.append({"weak_type":label,"full_marginal":np.mean(full),"low_marginal":np.mean([r["marginal_g_positive"] for r in low]),"low_nearest":np.mean([r["nearest_g_positive"] for r in low]),"low_manifold_gain":np.mean([r["delta_d_applied_positive"] for r in low])})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();art=a.artifact.resolve();records=[json.loads(x.read_text()) for x in sorted((art/"selection").glob("*.json"))]
 if len(records)!=250 or any(not r["integrity"]["finite"] or r["integrity"]["rows"]!=60 or r["integrity"]["all_ones_strong_parity_max_abs"]>=1e-6 for r in records):raise RuntimeError(f"invalid selection {len(records)}")
 full=[];time=[];window=[];tasks=[];gate=[]
 for b in BRANCHES:
  full_row=summary(states(records,b,range(10)),b,"full");full.append(full_row)
  for step in range(10):time.append(summary(states(records,b,[step]),b,f"flow_step_{step}"))
  for name,(lo,hi) in WINDOWS.items():
   low=summary(states(records,b,lo),b,name+"_low");high=summary(states(records,b,hi),b,name+"_high_placebo");window.extend([low,high]);task_good=0
   low_states=states(records,b,lo)
   for task in range(10):
    row=summary([x for x in low_states if x["task_id"]==task],b,name+f"_task_{task}");tasks.append(row);task_good+=row["marginal_g_positive"]>.5
   checks={"delta_d_ge_55":low["delta_d_applied_positive"]>=.55,"marginal_ge_60":low["marginal_g_positive"]>=.60,"marginal_ci_lower_gt_50":low["marginal_g_positive_ci95"][0]>.50,"tasks_gt_50_ge_7":task_good>=7,"low_minus_full_ge_10pp":low["marginal_g_positive"]-full_row["marginal_g_positive"]>=.10,"low_minus_high_ge_10pp":low["marginal_g_positive"]-high["marginal_g_positive"]>=.10,"low_delta_d_gt_high":low["delta_d_applied_positive"]>high["delta_d_applied_positive"]}
   gate.append({"candidate":b,"window":name,"low_steps":lo,"high_steps":hi,"tasks_marginal_gt_half":task_good,"low_marginal":low["marginal_g_positive"],"low_nearest":low["nearest_g_positive"],"low_delta_d":low["delta_d_applied_positive"],"high_marginal":high["marginal_g_positive"],"high_delta_d":high["delta_d_applied_positive"],"full_marginal":full_row["marginal_g_positive"],"checks":checks,"pass":all(checks.values())})
 write(art/"full_summary.csv",full);write(art/"per_timestep.csv",time);write(art/"window_summary.csv",window);write(art/"per_task_window.csv",tasks)
 passing=[x for x in gate if x["pass"]]
 selected=None
 if passing:
  best=max(x["low_delta_d"] for x in passing);near=[x for x in passing if best-x["low_delta_d"]<.03];near.sort(key=lambda x:((0 if x["candidate"]=="W1_skip_last_1" else 1),len(x["low_steps"]),-x["low_delta_d"]));selected=near[0];(art/"selected_self_weak.lock.json").write_text(json.dumps(selected,indent=2,sort_keys=True)+"\n");decision="SELECTION_PASS_CONFIRMATION_REQUIRED"
 else:decision="SLG_SELF_WEAK_LOW_NOISE_COMPATIBILITY_NO_GO"
 comparison=[]
 for name,(lo,_) in WINDOWS.items():
  for row in parent_comparison(Path(json.loads((art/"protocol.json").read_text())["selection"]["manifold_parent"]),lo):comparison.append({"window":name,**row})
  for b in BRANCHES:
   f=next(r for r in full if r["candidate"]==b);l=next(r for r in window if r["candidate"]==b and r["scope"]==name+"_low");comparison.append({"window":name,"weak_type":b,"full_marginal":f["marginal_g_positive"],"low_marginal":l["marginal_g_positive"],"low_nearest":l["nearest_g_positive"],"low_manifold_gain":l["delta_d_applied_positive"]})
 write(art/"weak_type_comparison.csv",comparison);result={"status":"SELF_WEAK_SELECTION_COMPLETE","decision":decision,"integrity":{"states":250,"rows_per_state":60,"strong_parity_max_abs":max(r["integrity"]["all_ones_strong_parity_max_abs"] for r in records),"normalization_max_abs":max(r["integrity"]["normalization_max_abs"] for r in records)},"scheduler":{"step_0":"t=1.0 highest noise","step_9":"t=0.1 lowest noise"},"full":full,"windows":window,"gates":gate,"selected":selected};(art/"selection_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");(art/"status/current.json").write_text(json.dumps({"stage":"SELECTION_COMPLETE","decision":decision},indent=2)+"\n");print(json.dumps(result,sort_keys=True))

if __name__=="__main__":main()
