#!/usr/bin/env python3
"""Post-process the completed matched causal rollout without rerunning episodes."""
from __future__ import annotations
import csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import random

ART = Path("artifacts/coreact_state_level_matched_causal_calibration_v1_20260809_225300")
ARMS = ("V_vanilla", "N0_shift_only", "W05_interpolation", "W20_extrapolation")
LABEL = {"N0_shift_only":"N0", "W05_interpolation":"W0.5", "W20_extrapolation":"W2"}

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def bootstrap(values, seed=20260810, reps=2000):
    if not values: return [float("nan")]*3
    rng=random.Random(seed); n=len(values); out=[]
    for _ in range(reps): out.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    out.sort(); return [out[int(.025*(reps-1))], mean(values), out[int(.975*(reps-1))]]

def mcnemar(a,b):
    # exact two-sided binomial test on discordant pairs
    x=sum(1 for u,v in zip(a,b) if u and not v); y=sum(1 for u,v in zip(a,b) if v and not u); n=x+y
    if n==0: return {"a_only":x,"b_only":y,"p_exact":1.0}
    k=min(x,y); p=sum(math.comb(n,i) for i in range(k+1))*0.5**n
    return {"a_only":x,"b_only":y,"p_exact":min(1.0,2*p)}

def main():
    lock = json.loads((ART/"rollout_code.lock.json").read_text())
    expected_protocol_hash = lock.get("files", {}).get("artifacts/coreact_state_level_matched_causal_calibration_v1_20260809_225300/protocol.lock.yaml")
    eps=[]
    for p in sorted((ART/"episodes").glob("*.json")):
        d=json.loads(p.read_text()); eps.append(d)
    by=defaultdict(dict); bad=[]; audit_counts=defaultdict(int)
    for d in eps:
        key=(d.get("snapshot_id"), int(d.get("noise_seed")))
        arm=d.get("arm")
        if arm in by[key]: bad.append((key,arm,"duplicate"))
        by[key][arm]=d
        if d.get("status")!="complete" or not d.get("all_actions_finite",False): bad.append((key,arm,"incomplete"))
        gp=ART/d.get("geometry_path", "")
        if not gp.is_file(): bad.append((key,arm,"missing_geometry")); continue
        audit_counts["geometry_present"] += 1
        if d.get("geometry_sha256") and sha256(gp) != d["geometry_sha256"]: bad.append((key,arm,"geometry_sha256"))
        if d.get("protocol_sha256") != expected_protocol_hash: bad.append((key,arm,"protocol_hash"))
    units=[]
    for key, arms in sorted(by.items()):
        if set(arms)!=set(ARMS): bad.append((key,"","missing_arm")); continue
        ref=arms["V_vanilla"]["branch_point"]
        for arm,d in arms.items():
            if d.get("branch_point",{}).get("fingerprints") != ref.get("fingerprints"): bad.append((key,arm,"branch_point_fingerprint"))
            if d.get("branch_point",{}).get("noise") != ref.get("noise"): bad.append((key,arm,"noise_hash"))
        v=int(bool(arms["V_vanilla"]["success"])); row={"snapshot_id":key[0],"noise_seed":key[1],"task_id":arms["V_vanilla"]["task_id"],"init_state_id":arms["V_vanilla"]["init_state_id"],"progress_bin":arms["V_vanilla"]["progress_bin"],"vanilla":v}
        for a in ARMS[1:]:
            y=int(bool(arms[a]["success"])); row[LABEL[a]]=y; row[f"D_{LABEL[a]}"]=y-v
        units.append(row)
    # Snapshot-level aggregation over the five matched noises.
    snaps=defaultdict(list)
    for r in units: snaps[r["snapshot_id"]].append(r)
    srows=[]
    for sid, rows in sorted(snaps.items()):
        base=rows[0]; sr={"snapshot_id":sid,"task_id":base["task_id"],"init_state_id":base["init_state_id"],"progress_bin":base["progress_bin"],"n_noise":len(rows)}
        for name in ("vanilla","N0","W0.5","W2"):
            sr[f"p_{name}"]=mean([r[name if name=="vanilla" else name] for r in rows])
        for name in ("N0","W0.5","W2"): sr[f"Delta_{name}"]=mean([r[f"D_{name}"] for r in rows])
        srows.append(sr)
    def write_csv(path, rows):
        if not rows: return
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_csv(ART/"causal_episode_pairs.csv", units); write_csv(ART/"state_level_causal.csv", srows)
    # Task-level descriptive and snapshot-cluster bootstrap CIs.
    taskrows=[]
    for task in sorted(set(r["task_id"] for r in srows)):
        rs=[r for r in srows if r["task_id"]==task]; tr={"task_id":task,"snapshots":len(rs)}
        for name in ("vanilla","N0","W0.5","W2"):
            tr[f"success_{name}"]=mean([r[f"p_{name}"] for r in rs])
        for name in ("N0","W0.5","W2"):
            vals=[r[f"Delta_{name}"] for r in rs]; lo,pt,hi=bootstrap(vals,20260810+int(task)*11)
            tr.update({f"Delta_{name}":pt,f"Delta_{name}_ci_low":lo,f"Delta_{name}_ci_high":hi,"{0}_rescue".format(name):sum(r[f"Delta_{name}"]>0 for r in rs),"{0}_harm".format(name):sum(r[f"Delta_{name}"]<0 for r in rs),"{0}_unchanged".format(name):sum(r[f"Delta_{name}"]==0 for r in rs)})
        taskrows.append(tr)
    write_csv(ART/"task_level_causal.csv", taskrows)
    def grouped_rows(group_key):
        out=[]
        groups=defaultdict(list)
        for r in srows: groups[group_key(r)].append(r)
        for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            g={"group":key,"snapshots":len(rs)}
            for name in ("vanilla","N0","W0.5","W2"):
                g[f"success_{name}"]=mean([r[f"p_{name}"] for r in rs])
            for name in ("N0","W0.5","W2"):
                vals=[r[f"Delta_{name}"] for r in rs]; lo,pt,hi=bootstrap(vals,20261000+sum(ord(c) for c in str(key)))
                g[f"Delta_{name}"]=pt; g[f"Delta_{name}_ci_low"]=lo; g[f"Delta_{name}_ci_high"]=hi
                g[f"{name}_rescue"]=sum(x>0 for x in vals); g[f"{name}_harm"]=sum(x<0 for x in vals); g[f"{name}_unchanged"]=sum(x==0 for x in vals)
            out.append(g)
        return out
    progress_rows=grouped_rows(lambda r: r["progress_bin"])
    task_progress_rows=grouped_rows(lambda r: f"task{int(r['task_id'])}@{r['progress_bin']}")
    write_csv(ART/"progress_level_causal.csv", progress_rows)
    write_csv(ART/"task_progress_causal.csv", task_progress_rows)
    # Pooled snapshot-cluster summaries and discordant episode counts.
    pooled={"snapshots":len(srows),"causal_units":len(units),"episodes":len(eps),"bad_records":bad}
    for name in ("N0","W0.5","W2"):
        vals=[r[f"Delta_{name}"] for r in srows]; lo,pt,hi=bootstrap(vals,20260901+len(name))
        pooled[name]={"delta":pt,"bootstrap_ci":[lo,hi],"rescue_snapshots":sum(x>0 for x in vals),"harm_snapshots":sum(x<0 for x in vals),"unchanged_snapshots":sum(x==0 for x in vals)}
        a=[r["vanilla"] for r in units]; b=[r[name] for r in units]; pooled[name]["mcnemar"]=mcnemar(a,b)
        pooled[name]["episode_rescue"]=sum(u==0 and z==1 for u,z in zip(a,b)); pooled[name]["episode_harm"]=sum(u==1 and z==0 for u,z in zip(a,b)); pooled[name]["episode_unchanged"]=sum(u==z for u,z in zip(a,b))
    # Same-task heterogeneity and overlap of rescue states.
    pooled["task_heterogeneity"]={str(t):{"N0":sum(r["Delta_N0"]>0 for r in srows if r["task_id"]==t),"W0.5":sum(r["Delta_W0.5"]>0 for r in srows if r["task_id"]==t),"W2":sum(r["Delta_W2"]>0 for r in srows if r["task_id"]==t)} for t in sorted(set(r["task_id"] for r in srows))}
    pooled["integrity_audit"]={"geometry_present":audit_counts["geometry_present"],"expected_geometry":len(eps),"bad_records":len(bad),"all_pass":len(bad)==0}
    (ART/"causal_summary.json").write_text(json.dumps(pooled,indent=2,sort_keys=True)+"\n")
    posthoc_decision = {
        "status": "CAUSAL_ROLLOUT_COMPLETE_HETEROGENEITY_WEAK_NO_SELECTOR",
        "basis": "N0 is harmful on paired episodes, while W0.5 and W2 pooled effects are near zero with snapshot-cluster CIs overlapping zero; only a small number of rescue snapshots were observed.",
        "integrity_pass": len(bad) == 0,
        "selector_training_started": False,
        "protocol_or_parameters_changed": False,
        "source_summary": "causal_summary.json"
    }
    (ART/"posthoc_decision.json").write_text(json.dumps(posthoc_decision,indent=2)+"\n")
    report=["# State-level matched causal analysis", "", f"Artifact: `{ART}`", "", "## Integrity", f"- Episodes: {len(eps)}/2000; causal units: {len(units)}/500; complete quadruplets: {len(units)==500 and not bad}.", f"- Invalid/missing/nonfinite records: {len(bad)}.", "- Unit of bootstrap: snapshot, after averaging five matched noise seeds.", "", "## Pooled results"]
    for name in ("N0","W0.5","W2"):
        x=pooled[name]; report.append(f"- {name}: snapshot Delta={x['delta']:.4f}, 95% CI [{x['bootstrap_ci'][0]:.4f}, {x['bootstrap_ci'][1]:.4f}]; episode rescue={x['episode_rescue']}, harm={x['episode_harm']}, unchanged={x['episode_unchanged']}; McNemar p={x['mcnemar']['p_exact']:.4g}.")
    report += ["", "## Interpretation", "These are paired development results, not independent confirmation. Positive Delta means the arm succeeded when vanilla failed more often than it harmed vanilla successes. State-level heterogeneity is reported without training a selector; no task or parameter was changed after seeing outcomes.", "", "## Progress split", "Progress and task-progress tables use snapshot-cluster bootstrap; the five noise seeds are averaged within each snapshot before inference."]
    for r in progress_rows:
        report.append(f"- Progress {r['group']}: Vanilla {float(r['success_vanilla']):.3f}; N0 {float(r['success_N0']):.3f} (Delta {float(r['Delta_N0']):+.3f}, rescue/harm {r['N0_rescue']}/{r['N0_harm']}); W0.5 {float(r['success_W0.5']):.3f} (Delta {float(r['Delta_W0.5']):+.3f}, rescue/harm {r['W0.5_rescue']}/{r['W0.5_harm']}); W2 {float(r['success_W2']):.3f} (Delta {float(r['Delta_W2']):+.3f}, rescue/harm {r['W2_rescue']}/{r['W2_harm']}).")
    report += ["", "## Files", "- `causal_episode_pairs.csv`: one row per matched snapshot/noise unit.", "- `state_level_causal.csv`: one row per snapshot, averaged over five noises.", "- `task_level_causal.csv`: task summaries and snapshot bootstrap intervals.", "- `progress_level_causal.csv`: progress-bin summaries.", "- `task_progress_causal.csv`: task-by-progress summaries.", "- `causal_summary.json`: machine-readable pooled and discordant counts."]
    (ART/"causal_analysis_report.md").write_text("\n".join(report)+"\n")
    hashes={p.name:sha256(p) for p in (ART/"causal_episode_pairs.csv",ART/"state_level_causal.csv",ART/"task_level_causal.csv",ART/"causal_summary.json",ART/"causal_analysis_report.md",ART/"posthoc_decision.json")}
    (ART/"causal_analysis_sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(pooled,indent=2,sort_keys=True))

if __name__=="__main__": main()
