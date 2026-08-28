#!/usr/bin/env python3
"""Paired analysis for the Task-4 guidance-sign diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

from research.coreact_closed_loop.audit_qualification import video_frames
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json


def compare(rows,first,second):
    a=np.asarray([r[f"{first}_success"] for r in rows],float); b=np.asarray([r[f"{second}_success"] for r in rows],float); d=a-b
    rng=np.random.default_rng(84_004_004); boot=[float(np.mean(d[rng.integers(0,len(d),len(d))])) for _ in range(2000)]
    ao=int(np.sum((a==1)&(b==0))); bo=int(np.sum((a==0)&(b==1))); n=ao+bo
    return {"comparison":f"{first}_minus_{second}","success_rate_difference":float(np.mean(d)),"paired_state_bootstrap_95_ci":np.quantile(boot,[.025,.975]).tolist(),
        "exact_mcnemar_p_value":float(binomtest(ao,n,.5).pvalue) if n else 1.0,f"{first}_only_success":ao,f"{second}_only_success":bo,"discordant_total":n}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--artifact",type=Path,required=True); args=p.parse_args(); artifact=args.artifact.resolve()
    protocol=yaml.safe_load((artifact/"protocol.lock.yaml").read_text()); top_k=int(protocol["selection"]["top_k"])
    mask_name=f"top{top_k}_mask_only"; away_name=f"coreact_away_top{top_k}"; toward_name=f"coreact_toward_top{top_k}"
    conditions=("vanilla",mask_name,away_name,toward_name)
    comparisons=((toward_name,away_name),(away_name,"vanilla"),(toward_name,"vanilla"),(mask_name,"vanilla"),(toward_name,mask_name))
    specs=read_jsonl(artifact/"episode_manifest.jsonl")
    amendment_path=artifact/"episode_manifest.amendment.jsonl"
    if amendment_path.exists():specs.extend(read_jsonl(amendment_path))
    amendment=json.loads((artifact/"integrity_amendment.json").read_text()) if (artifact/"integrity_amendment.json").exists() else {}
    superseded_pair_ids=set(amendment.get("superseded_pair_ids",[]))
    ids=[r["episode_id"] for r in specs]; failures=[]; dirs={x.name for x in (artifact/"episodes").iterdir() if x.is_dir()}
    if dirs!=set(ids):failures.append({"episode_id":"aggregate","reason":"directory set mismatch","missing":sorted(set(ids)-dirs),"extra":sorted(dirs-set(ids))})
    records={}
    for s in specs:
        f=artifact/"episodes"/s["episode_id"]/"episode.json"
        if not f.exists():failures.append({"episode_id":s["episode_id"],"reason":"missing"});continue
        r=json.loads(f.read_text());records[s["episode_id"]]=r
        if r["episode_id"]!=s["episode_id"] or r["status"]!="complete" or not r["all_sampler_outputs_finite"] or r["nonfinite_action_count"]:failures.append({"episode_id":s["episode_id"],"reason":"identity/status/finite"})
        if video_frames(artifact/r["video"]) is None or len(read_jsonl(artifact/r["step_log"]))!=r["control_steps"]:failures.append({"episode_id":s["episode_id"],"reason":"video/steps"})
    grouped=defaultdict(dict)
    for r in records.values():
        if r["pair_id"] not in superseded_pair_ids:grouped[r["pair_id"]][r["condition"]]=r
    pairs=[]
    for pair,cs in sorted(grouped.items()):
        if set(cs)!=set(conditions):failures.append({"episode_id":pair,"reason":"incomplete pair"});continue
        ref=cs["vanilla"]; keys=("suite","task_id","init_state_id","group_count","reset_seed","action_noise_seed","selection_seed","language")
        if any(cs[c][k]!=ref[k] for c in conditions for k in keys):failures.append({"episode_id":pair,"reason":"pair identity"});continue
        if len({cs[c]["initial_sim_state_sha256"] for c in conditions})!=1 or len({cs[c]["initial_prepared_input_sha256"] for c in conditions})!=1:failures.append({"episode_id":pair,"reason":"state/prepared identity"});continue
        m=min(cs[c]["replans"] for c in conditions)
        if any(cs[c]["noise_sha256_by_replan"][:m]!=ref["noise_sha256_by_replan"][:m] for c in conditions):failures.append({"episode_id":pair,"reason":"noise identity"});continue
        away=cs[away_name]["replan_traces"][0];toward=cs[toward_name]["replan_traces"][0];mask=cs[mask_name]["replan_traces"][0]
        if away["selected_indices"]!=toward["selected_indices"] or away["selected_indices"]!=mask["selected_indices"] or away["prefix_sha256"]!=toward["prefix_sha256"] or away["negative_prefix_sha256"]!=toward["negative_prefix_sha256"] or away["negative_prefix_sha256"]!=mask["masked_prefix_sha256"]:
            failures.append({"episode_id":pair,"reason":"top8/prefix identity"});continue
        row={"pair_id":pair,"init_state_id":ref["init_state_id"]}
        for c in conditions:
            r=cs[c];row.update({f"{c}_success":int(r["success"]),f"{c}_control_steps":r["control_steps"],f"{c}_action_total_variation":r["action_total_variation"],f"{c}_chunk_boundary_discontinuity":r["mean_chunk_boundary_discontinuity"],f"{c}_latency_median_per_replan":r["policy_seconds_median_per_replan"],f"{c}_nonfinite_count":r["nonfinite_action_count"],f"{c}_bound_violation_count":r["normalized_bound_violation_count"]})
        for c in (away_name,toward_name):
            steps=[st for tr in cs[c]["replan_traces"] for st in tr["step_traces"]]
            row[f"{c}_mean_clip_scale"]=float(np.mean([x["clip_scale"] for x in steps]));row[f"{c}_mean_delta_norm"]=float(np.mean([x["negative_delta_norm"] for x in steps]));row[f"{c}_mean_applied_guidance_norm"]=float(np.mean([x["applied_guidance_norm"] for x in steps]))
        pairs.append(row)
    expected_records=200+4*len(superseded_pair_ids)
    if len(specs)!=expected_records or len(records)!=expected_records or len(pairs)!=50 or len(set(ids))!=expected_records:failures.append({"episode_id":"aggregate","reason":f"expected {expected_records} records and 50 analysis pairs"})
    if failures:write_json(artifact/"analysis_integrity_failure.json",{"failures":failures,"episodes":len(records),"pairs":len(pairs)});raise SystemExit(1)
    rates={c:float(np.mean([r[f"{c}_success"] for r in pairs])) for c in conditions}; comps=[compare(pairs,a,b) for a,b in comparisons]; lookup={x["comparison"]:x for x in comps}; primary=lookup[f"{toward_name}_minus_{away_name}"]
    secondary={c:{m:{"mean":float(np.mean([r[f"{c}_{m}"] for r in pairs])),"median":float(np.median([r[f"{c}_{m}"] for r in pairs]))} for m in ("control_steps","action_total_variation","chunk_boundary_discontinuity","latency_median_per_replan","nonfinite_count","bound_violation_count")} for c in conditions}
    mechanism={c:{m:float(np.mean([r[f"{c}_{m}"] for r in pairs])) for m in ("mean_clip_scale","mean_delta_norm","mean_applied_guidance_norm")} for c in (away_name,toward_name)}
    if primary["paired_state_bootstrap_95_ci"][0]>0:status="TOWARD_SIGN_BETTER_THAN_AWAY"
    elif primary["paired_state_bootstrap_95_ci"][1]<0:status="AWAY_SIGN_BETTER_THAN_TOWARD"
    else:status="GUIDANCE_SIGN_INCONCLUSIVE"
    summary={"stage":"mechanism_development_diagnostic_not_confirmation","top_k":top_k,"episodes":200,"raw_records":len(records),"superseded_integrity_records":4*len(superseded_pair_ids),"complete_pairs":50,"missingness_rate":0.0,"success_rates":rates,"comparisons":comps,"primary_comparison":primary,"secondary_metrics":secondary,"mechanism_metrics":mechanism,"decision":status,"integrity_failures":[]}
    write_json(artifact/"analysis_summary.json",summary);write_json(artifact/"decision.json",{"status":status,"primary_comparison":primary})
    with (artifact/"paired_results.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    with (artifact/"missingness.csv").open("w",newline="") as f:csv.DictWriter(f,fieldnames=("episode_id","reason")).writeheader()
    (artifact/"report.md").write_text(f"# Task {protocol['task_id']} CoreAct Top-{top_k} Guidance-Sign Diagnostic\n\nTask: {protocol['task']}\n\n## Integrity\n\n- Episodes: 200/200\n- Raw append-only records: {len(records)}\n- Superseded integrity records: {4*len(superseded_pair_ids)}\n- Complete four-condition pairs: 50/50\n- Missing/integrity failures: 0\n\n## Success\n\n- Vanilla: {rates['vanilla']:.1%}\n- Mask-only: {rates[mask_name]:.1%}\n- Away: {rates[away_name]:.1%}\n- Toward: {rates[toward_name]:.1%}\n\n## Comparisons\n\n```json\n{json.dumps(comps,indent=2,sort_keys=True)}\n```\n\n## Mechanism Metrics\n\n```json\n{json.dumps(mechanism,indent=2,sort_keys=True)}\n```\n\nDecision: `{status}`. This is a development diagnostic, not confirmation.\n")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":main()
