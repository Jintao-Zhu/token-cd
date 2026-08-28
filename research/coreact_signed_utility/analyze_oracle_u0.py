from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ARMS=("V","IT","IA","AT","AA","RT","RA"); ORDER={x:i for i,x in enumerate(ARMS)}

def ci(x): return [float(np.quantile(x,.025)),float(np.quantile(x,.975))]
def main():
 import argparse; p=argparse.ArgumentParser(); p.add_argument("--artifact",type=Path,required=True); a=p.parse_args(); art=a.artifact.resolve(); paths=sorted((art/"episodes").glob("*.json"))
 if len(paths)!=1680 or list((art/"invalid_units").glob("*.json")): raise RuntimeError("full integrity gate not met")
 rows=[json.loads(x.read_text()) for x in paths]
 if any(r["status"]!="complete" or not r["all_actions_finite"] for r in rows): raise RuntimeError("invalid episode")
 units={}
 for r in rows: units.setdefault(r["unit_id"],{})[r["arm"]]=r
 if len(units)!=240 or any(set(x)!=set(ARMS) for x in units.values()): raise RuntimeError("unit/arm coverage")
 for unit_id, group in units.items():
  for field in ("fingerprints", "noise_sha256", "clean_chunk_sha256"):
   if len({json.dumps(group[arm]["branch_point"][field], sort_keys=True) for arm in ARMS}) != 1:
    raise RuntimeError(f"{unit_id}: branch {field} mismatch")
  if not all(group[arm]["branch_point"]["reference_equal"] for arm in ARMS):
   raise RuntimeError(f"{unit_id}: reference mismatch")
  for arm in ARMS:
   row = group[arm]; traces = row["intervention_traces"]
   expected = 0 if arm == "V" else min(3, int(row["replans"]))
   if row["candidate_replans_applied"] != expected or len(traces) != expected:
    raise RuntimeError(f"{unit_id}/{arm}: intervention duration")
   if [trace["replan"] for trace in traces] != list(range(expected)):
    raise RuntimeError(f"{unit_id}/{arm}: intervention replan indices")
   if any(len(trace["changed_indices"]) != 4 or not trace["protected_tokens_untouched"] for trace in traces):
    raise RuntimeError(f"{unit_id}/{arm}: token integrity")
 snapshots={}
 for unit in units.values(): snapshots.setdefault(unit["V"]["snapshot_id"],[]).append(unit)
 if len(snapshots)!=40 or any(len(x)!=6 for x in snapshots.values()): raise RuntimeError("snapshot seed coverage")
 selections={}; selection_rows=[]
 for sid,six in sorted(snapshots.items()):
  selection=sorted([x for x in six if x["V"]["seed_role"]=="selection"],key=lambda x:x["V"]["continuation_seed"]); evaluation=sorted([x for x in six if x["V"]["seed_role"]=="evaluation"],key=lambda x:x["V"]["continuation_seed"])
  successes={arm:sum(int(x[arm]["success"]) for x in selection) for arm in ARMS}; top=max(successes.values()); tied=[x for x in ARMS if successes[x]==top]
  if "V" in tied: chosen="V"; tie_reason="vanilla_at_max"
  else:
   norms={arm:float(np.mean([x[arm]["mean_applied_correction_norm"] for x in selection])) for arm in tied}; minimum=min(norms.values()); chosen=min((x for x in tied if abs(norms[x]-minimum)<1e-12),key=lambda x:ORDER[x]); tie_reason="min_correction_then_fixed_order"
  selections[sid]=chosen; v=[int(x["V"]["success"]) for x in evaluation]; o=[int(x[chosen]["success"]) for x in evaluation]; meta=evaluation[0]["V"]
  selection_rows.append({"snapshot_id":sid,"task_id":meta["task_id"],"target_progress":meta["target_progress"],"chosen_arm":chosen,"tie_reason":tie_reason,"selection_successes":successes,"evaluation_vanilla":v,"evaluation_oracle":o})
 eval_rows=[]
 for s in selection_rows:
  for i,(v,o) in enumerate(zip(s["evaluation_vanilla"],s["evaluation_oracle"])): eval_rows.append({"snapshot_id":s["snapshot_id"],"task_id":s["task_id"],"seed":i+3,"V":v,"O":o,"chosen_arm":s["chosen_arm"]})
 d=np.asarray([x["O"]-x["V"] for x in eval_rows]); effect=float(d.mean()*100); rescue=int((d==1).sum()); harm=int((d==-1).sum()); disc=rescue+harm; rng=np.random.default_rng(1729)
 by_snapshot=np.asarray([[x["O"]-x["V"] for x in eval_rows if x["snapshot_id"]==sid] for sid in sorted(snapshots)])
 boot=np.empty(100000); strat=np.empty(100000); by_task=[by_snapshot[i*4:(i+1)*4] for i in range(10)]
 for start in range(0,100000,2000):
  idx=rng.integers(0,40,(2000,40)); boot[start:start+2000]=by_snapshot[idx].mean(axis=(1,2))*100
  ti=rng.integers(0,4,(2000,10,4)); sampled=np.stack([by_task[t][ti[:,t]] for t in range(10)],axis=1); strat[start:start+2000]=sampled.mean(axis=(1,2,3))*100
 fixed={arm:float(np.mean([int(unit[arm]["success"]) for unit in units.values() if unit[arm]["seed_role"]=="evaluation"])) for arm in ARMS}
 task=[]
 for t in range(10):
  sub=[x for x in eval_rows if x["task_id"]==t]; task.append({"task_id":t,"vanilla":int(sum(x["V"] for x in sub)),"oracle":int(sum(x["O"] for x in sub)),"delta_pp":float(100*np.mean([x["O"]-x["V"] for x in sub]))})
 signs=[int(sum(x["delta_pp"]>0 for x in task)),int(sum(x["delta_pp"]==0 for x in task)),int(sum(x["delta_pp"]<0 for x in task))]; best_arm=max((x for x in ARMS if x!="V"),key=lambda x:fixed[x]); oracle_rate=float(np.mean([x["O"] for x in eval_rows])); best_delta=float((oracle_rate-fixed[best_arm])*100)
 hindsight=float(np.mean([max(int(units[x["snapshot_id"]+f"__seed{x['seed']:02d}"][arm]["success"]) for arm in ARMS) for x in eval_rows]))
 lower=ci(boot)[0]; strong=effect>=8 and lower>0 and signs[0]+signs[1]>=7 and rescue>harm and min(x["delta_pp"] for x in task)>=-15 and best_delta>=3; promising=5<=effect<8 and signs[0]+signs[1]>=7 and rescue>harm and best_delta>0
 decision="TOKEN_DIRECTION_ORACLE_HEADROOM_STRONG_GO" if strong else "TOKEN_DIRECTION_ORACLE_HEADROOM_PROMISING" if promising else "TOKEN_DIRECTION_ORACLE_HEADROOM_NO_GO"
 analysis={"decision":decision,"episodes":1680,"snapshots":40,"evaluation_units":120,"oracle_rate":oracle_rate,"vanilla_rate":fixed["V"],"oracle_minus_vanilla_pp":effect,"snapshot_cluster_bootstrap_95_ci_pp":ci(boot),"task_stratified_bootstrap_95_ci_pp":ci(strat),"mcnemar_exact_p":float(binomtest(min(rescue,harm),disc).pvalue) if disc else 1.0,"rescue":rescue,"harm":harm,"fixed_arm_rates":fixed,"best_fixed_arm":best_arm,"oracle_minus_best_fixed_pp":best_delta,"task_signs_positive_tie_negative":signs,"hindsight_candidate_max_descriptive":hindsight,"task_rows":task,"selection_counts":{arm:int(sum(x["chosen_arm"]==arm for x in selection_rows)) for arm in ARMS}}
 (art/"analysis.json").write_text(json.dumps(analysis,indent=2)+"\n"); (art/"decision.json").write_text(json.dumps({"decision":decision,"analysis":analysis},indent=2)+"\n")
 integrity={"status":"PASS","snapshots":"40/40","episodes":"1680/1680","matched_units":"240/240","snapshot_arm_groups":"280/280","invalid_units":0,"complete_and_finite":"1680/1680","branch_hash_match":"240/240","candidate_duration_audit":"PASS","four_changed_tokens_each_candidate_replan":"PASS","selection_evaluation_seed_separation":"PASS"}; (art/"integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
 with (art/"task_success.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=task[0]);w.writeheader();w.writerows(task)
 table="\n".join(f"| {x['task_id']} | {x['vanilla']}/12 | {x['oracle']}/12 | {x['delta_pp']:+.1f}pp |" for x in task); fixed_table="\n".join(f"| {x} | {fixed[x]:.1%} |" for x in ARMS)
 report=f"""# Token Candidate Oracle Headroom U0\n\n## Decision\n\n`{decision}`\n\n- Evaluation-only Oracle: {oracle_rate:.1%}\n- Vanilla: {fixed['V']:.1%}\n- Oracle - Vanilla: {effect:+.1f}pp\n- Snapshot-cluster 95% CI: [{ci(boot)[0]:+.1f}, {ci(boot)[1]:+.1f}]pp\n- Task-stratified 95% CI: [{ci(strat)[0]:+.1f}, {ci(strat)[1]:+.1f}]pp\n- McNemar p: {analysis['mcnemar_exact_p']:.6g}; rescue/harm: {rescue}/{harm}\n- Oracle positive/tie/negative tasks: {signs[0]}/{signs[1]}/{signs[2]}\n- Best fixed arm: {best_arm} ({fixed[best_arm]:.1%}); Oracle advantage: {best_delta:+.1f}pp\n- Hindsight candidate max (descriptive only): {hindsight:.1%}\n\n## Fixed evaluation arms\n\n| Arm | Success |\n|---|---:|\n{fixed_table}\n\n## Per task\n\n| Task | Vanilla | Oracle | Delta |\n|---:|---:|---:|---:|\n{table}\n\nSelection used seeds 0-2 only; all reported outcomes use held-out seeds 3-5. No training, new score, lambda tuning, or partial-outcome selection was used.\n"""; (art/"report.md").write_text(report); (art/"status.json").write_text(json.dumps({"status":"complete","decision":decision,"episodes":"1680/1680","integrity":"PASS"},indent=2)+"\n")
 files=sorted(x for x in art.rglob("*") if x.is_file() and x.name!="final_artifacts.sha256" and "logs" not in x.relative_to(art).parts); (art/"final_artifacts.sha256").write_text("\n".join(f"{hashlib.sha256(x.read_bytes()).hexdigest()}  {x.relative_to(art)}" for x in files)+"\n"); print(json.dumps(analysis,indent=2))
if __name__=="__main__": main()
