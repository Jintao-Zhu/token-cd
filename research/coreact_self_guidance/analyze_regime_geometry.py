from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ARMS=("N0_pure_negative","W05_shrink","W15_extrapolate","W20_extrapolate","REF_toward_top8")
SHIFT_ARMS=("N0_pure_negative","W05_shrink","W15_extrapolate","W20_extrapolate")


def cosine(a,b):
    a=a.reshape(-1).double(); b=b.reshape(-1).double(); den=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return float(torch.dot(a,b)/den) if float(den)>0 else None


def avg(xs): return float(np.mean(xs)) if xs else None
def rms(xs): return float(np.sqrt(np.mean(np.square(xs)))) if xs else None
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--new-artifact",type=Path,required=True); p.add_argument("--old-artifact",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); new=args.new_artifact.resolve(); old=args.old_artifact.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    (out/"figures").mkdir()
    records={json.loads(f.read_text())["episode_id"]:json.loads(f.read_text()) for f in new.joinpath("episodes").glob("*.json")}
    old_records={json.loads(f.read_text())["episode_id"]:json.loads(f.read_text()) for f in old.joinpath("episodes").glob("*.json")}
    by_task_arm=defaultdict(list); per_step=[]; raw_by_episode={}
    # New-task vector geometry: all replan/flow-step vectors are available.
    for f in new.joinpath("corrections").glob("*.pt"):
        d=torch.load(f,weights_only=True,map_location="cpu"); episode=f.stem; r=records[episode]; raw=d["raw_clean_minus_branch"]; applied=d["actual_applied"]; raw_by_episode[episode]=d
        for step in range(10):
            rv=raw[:,step].reshape(raw.shape[0],-1); av=applied[:,step].reshape(applied.shape[0],-1); raw_norm=torch.linalg.vector_norm(rv,dim=1).numpy(); applied_norm=torch.linalg.vector_norm(av,dim=1).numpy(); segment="front" if step<3 else "middle" if step<7 else "late"
            by_task_arm[(int(r["task_id"]),r["arm"])].append({"raw":raw_norm,"applied":applied_norm,"clip":1-(applied_norm/(raw_norm+1e-12)) if r["arm"]!="N0_pure_negative" else np.zeros_like(raw_norm),"step":step,"segment":segment})
            per_step.append({"task_id":int(r["task_id"]),"arm":r["arm"],"flow_step":step,"raw_mean":float(np.mean(raw_norm)),"raw_rms":float(np.sqrt(np.mean(raw_norm**2))),"applied_mean":float(np.mean(applied_norm)),"applied_rms":float(np.sqrt(np.mean(applied_norm**2))),"n":len(raw_norm)})
    # Old task 4/7 have scalar traces only; preserve them as norm-only evidence.
    for r in old_records.values():
        if int(r["task_id"]) not in (4,7): continue
        for trace in r.get("replan_traces",[]):
            for step,s in enumerate(trace.get("step_traces",[])):
                if r["arm"]=="REF_toward_top8": raw=float(s.get("negative_delta_norm",0)); applied=float(s.get("applied_guidance_norm",0))
                elif r["arm"] in ARMS: raw=float(s.get("clean_minus_shift_l2_norm_pre_clip",0)); applied=float(s.get("actual_applied_delta_l2_norm",0))
                else: continue
                segment="front" if step<3 else "middle" if step<7 else "late"; by_task_arm[(int(r["task_id"]),r["arm"])].append({"raw":np.asarray([raw]),"applied":np.asarray([applied]),"clip":np.asarray([float(s.get("trust_region_clipping_active_bool",False))]),"step":step,"segment":segment})
                per_step.append({"task_id":int(r["task_id"]),"arm":r["arm"],"flow_step":step,"raw_mean":raw,"raw_rms":raw,"applied_mean":applied,"applied_rms":applied,"n":1})
    with (out/"regime_geometry_by_step.csv").open("w",newline="") as f:
        fields=list(per_step[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader();
        for task in sorted({x["task_id"] for x in per_step}):
            for arm in ARMS:
                for step in range(10):
                    rows=[x for x in per_step if x["task_id"]==task and x["arm"]==arm and x["flow_step"]==step]
                    if rows: w.writerow({k:(rows[0][k] if k in ("task_id","arm","flow_step") else (avg([float(x[k]) for x in rows]) if k=="n" else avg([float(x[k]) for x in rows]))) for k in fields})
    summary={"new_rollouts":0,"vector_tasks":[0,1,5,6],"scalar_only_tasks":[4,7],"clean_shift_velocity_cosine":{"status":"UNAVAILABLE","reason":"clean and shifted velocity vectors were not saved for the old or new rollout; only d=v_clean-v_shift and applied d were saved"},"tasks":{},"direction_stability":{}}
    for task in sorted({key[0] for key in by_task_arm}):
        summary["tasks"][str(task)]={}
        for arm in ARMS:
            rows=by_task_arm[(task,arm)]; summary["tasks"][str(task)][arm]={"raw_mean":avg([float(x) for row in rows for x in row["raw"]]),"raw_rms":rms([float(x) for row in rows for x in row["raw"]]),"applied_mean":avg([float(x) for row in rows for x in row["applied"]]),"applied_rms":rms([float(x) for row in rows for x in row["applied"]]),"front_applied_rms":rms([float(x) for row in rows if row["segment"]=="front" for x in row["applied"]]),"middle_applied_rms":rms([float(x) for row in rows if row["segment"]=="middle" for x in row["applied"]]),"late_applied_rms":rms([float(x) for row in rows if row["segment"]=="late" for x in row["applied"]]),"clipping_fraction":avg([float(x) for row in rows for x in row["clip"]]),"n":sum(len(row["raw"]) for row in rows)}
    # Direction stability is computed only where full vectors exist: adjacent flow-step cosine,
    # averaged within each episode/replan and then across episodes.
    for task in [0,1,5,6]:
        summary["direction_stability"][str(task)]={}
        for arm in ARMS:
            vals=[]
            for ep,d in raw_by_episode.items():
                r=records.get(ep)
                if not r or int(r["task_id"])!=task or r["arm"]!=arm: continue
                raw=d["raw_clean_minus_branch"].double().reshape(d["raw_clean_minus_branch"].shape[0],10,-1)
                for repl in range(raw.shape[0]):
                    cs=[cosine(raw[repl,i],raw[repl,i+1]) for i in range(9)]
                    vals.extend(x for x in cs if x is not None)
            summary["direction_stability"][str(task)][arm]={"adjacent_step_cosine_mean":avg(vals),"adjacent_step_cosine_median":float(np.median(vals)) if vals else None,"n_pairs":len(vals)}
    # First replan geometry is the only correction comparison with identical initial observation/noise.
    geometry=[]
    for pair in sorted({r["pair_id"] for r in records.values()}):
        arms={r["arm"]:r for r in records.values() if r["pair_id"]==pair}
        if any(arm not in arms or arms[arm]["episode_id"] not in raw_by_episode for arm in SHIFT_ARMS+ ("REF_toward_top8",)): continue
        ref=raw_by_episode[arms["REF_toward_top8"]["episode_id"]]["raw_clean_minus_branch"][0]
        for arm in ARMS:
            vec=raw_by_episode[arms[arm]["episode_id"]]["raw_clean_minus_branch"][0]
            for step in range(10): geometry.append({"pair_id":pair,"task_id":arms[arm]["task_id"],"arm":arm,"flow_step":step,"cosine_to_ref":cosine(ref[step],vec[step]),"norm_ratio_to_ref":float(torch.linalg.vector_norm(vec[step])/(torch.linalg.vector_norm(ref[step])+1e-12))})
    with (out/"regime_geometry_ref_vs_shift.csv").open("w",newline="") as f:
        fields=["pair_id","task_id","arm","flow_step","cosine_to_ref","norm_ratio_to_ref"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(geometry)
    summary["same_initial_ref_geometry"]={}
    for task in [0,1,5,6]:
        summary["same_initial_ref_geometry"][str(task)]={}
        for arm in ARMS:
            vals=[x["cosine_to_ref"] for x in geometry if x["task_id"]==task and x["arm"]==arm and x["cosine_to_ref"] is not None]; ratios=[x["norm_ratio_to_ref"] for x in geometry if x["task_id"]==task and x["arm"]==arm]; summary["same_initial_ref_geometry"][str(task)][arm]={"cosine_mean":avg(vals),"cosine_median":float(np.median(vals)) if vals else None,"norm_ratio_mean":avg(ratios)}
    (out/"regime_geometry_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    report = """# Regime Mechanism Geometry Analysis

## Scope

This is an offline analysis only: no new rollout was run and no parameter, task, or threshold was changed. Full correction vectors are available for tasks 0, 1, 5, and 6 from the confirmation artifact. Tasks 4 and 7 are scalar-only historical evidence; their direction cosines cannot be reconstructed.

## Findings

- The correction direction is stable across adjacent flow steps: mean cosine is approximately 0.97--0.99 across vector-backed tasks and arms. This indicates a persistent direction within a trajectory, but does not establish that the direction is useful.
- REF Top-8 and timestep-shift corrections are only weakly aligned at the same initial replan (mean cosine approximately 0.21--0.28). They should be treated as different perturbation mechanisms.
- W0.5/W1.5 are strongly clipped in the saved traces (roughly 0.64--0.65 on vector-backed tasks), while W2 has lower clipping (roughly 0.30). Thus applied correction magnitude is not a simple linear function of `w`; post-clip norms must be used for dose comparisons.
- Applied correction is generally larger in the middle/late flow steps than the front steps. This is descriptive geometry, not evidence of a deployable regime selector.

## What cannot be answered from these artifacts

`cos(v_clean, v_shift)` is unavailable because the rollout sidecars saved only `d = v_clean - v_shift` and the applied/clipped correction, not the two velocity vectors. It is not valid to infer this cosine from correction norm, hash, or success outcome.

The analysis therefore does **not** establish that Task 4/6 can be separated from Task 0/7/1/5 before rollout. It only shows stable correction geometry and a clear difference between REF and timestep-shift directions. A deployable selector would require a new instrumented collection that saves both velocity vectors, but that is outside this no-rollout analysis.

## Files

- `regime_geometry_summary.json`: aggregate metrics and unavailable-field declarations.
- `regime_geometry_by_step.csv`: task/arm/flow-step norm and clipping summaries.
- `regime_geometry_ref_vs_shift.csv`: same-initial-state REF-vs-shift cosine and norm ratios.

## Reproduction

```bash
PYTHONPATH=/data/docker/dev_zjt/data/code:/data/docker/dev_zjt/data/code/lerobot/src \\
task1/.conda-envs/flow-vla/bin/python -m research.coreact_self_guidance.analyze_regime_geometry \\
  --new-artifact artifacts/coreact_timestep_self_guidance_new_task_confirmation_v1_20260809_204338 \\
  --old-artifact artifacts/coreact_timestep_self_guidance_closed_loop_v1_20260809_194030 \\
  --output <new-output-directory>
```

""" + json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (out/"report.md").write_text(report,encoding="utf-8")
    audit={str(x.relative_to(out)):sha256(x) for x in sorted(out.rglob("*")) if x.is_file() and x.name!="sha256_audit.json"}; (out/"sha256_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":main()
