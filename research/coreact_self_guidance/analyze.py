from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ARMS=("A_vanilla","N0_pure_negative","W05_shrink","W15_extrapolate","W20_extrapolate","REF_toward_top8")
COMPARES=(("W15_extrapolate","A_vanilla"),("W20_extrapolate","A_vanilla"),("N0_pure_negative","A_vanilla"),("W05_shrink","A_vanilla"),("W15_extrapolate","REF_toward_top8"))


def sha256(path):
    d=hashlib.sha256();
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): d.update(block)
    return d.hexdigest()


def compare(rows,first,second,seed):
    a=np.asarray([int(r[first]["success"]) for r in rows]); b=np.asarray([int(r[second]["success"]) for r in rows]); delta=a-b; rng=np.random.default_rng(seed); boot=[float(np.mean(delta[rng.integers(0,len(delta),len(delta))])) for _ in range(2000)]; x=int(((a==1)&(b==0)).sum()); y=int(((a==0)&(b==1)).sum()); n=x+y
    return {"comparison":f"{first}_minus_{second}","point_estimate":float(delta.mean()),"paired_bootstrap_95_ci":np.quantile(boot,[.025,.975]).tolist(),"first_only_success":x,"second_only_success":y,"discordant":n,"exact_mcnemar_p":float(binomtest(x,n,.5).pvalue) if n else 1.0}


def contains_zero(row): return row["paired_bootstrap_95_ci"][0] <= 0 <= row["paired_bootstrap_95_ci"][1]


def holm(values):
    order=sorted(values,key=values.get); out={}; running=0.
    for i,k in enumerate(order): running=max(running,min(1.,(len(order)-i)*values[k])); out[k]=running
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument("--artifact",type=Path,required=True); a=p.parse_args(); art=a.artifact.resolve(); specs=[json.loads(x) for x in (art/"episode_manifest.jsonl").read_text().splitlines()]; files=list((art/"episodes").glob("*.json")); expected={s["episode_id"] for s in specs}; actual={f.stem for f in files}
    if len(specs)!=600 or len(files)!=600 or actual!=expected: raise RuntimeError(f"analysis forbidden before 600/600: specs={len(specs)} files={len(files)}")
    grouped=defaultdict(dict); failures=[]
    for f in files:
        r=json.loads(f.read_text()); grouped[r["pair_id"]][r["arm"]]=r
        if r.get("status")!="complete" or not r.get("all_actions_finite"): failures.append({"episode_id":r.get("episode_id"),"reason":"status/finite"})
    pairs=[]
    for pair,arms in sorted(grouped.items()):
        if set(arms)!=set(ARMS): failures.append({"episode_id":pair,"reason":"arm set"}); continue
        ref=arms["A_vanilla"]; keys=("suite","task_id","init_state_id","reset_seed","action_noise_seed","selection_seed","language")
        if any(arms[c][k]!=ref[k] for c in ARMS for k in keys): failures.append({"episode_id":pair,"reason":"identity"}); continue
        if len({arms[c]["initial_sim_state_sha256"] for c in ARMS})!=1 or len({arms[c]["initial_prepared_input_sha256"] for c in ARMS})!=1: failures.append({"episode_id":pair,"reason":"state/preprocessing"}); continue
        m=min(arms[c]["replans"] for c in ARMS); noises=ref["first_noise_sha256_by_replan"][:m]
        if any(arms[c]["first_noise_sha256_by_replan"][:m]!=noises for c in ARMS): failures.append({"episode_id":pair,"reason":"noise"}); continue
        for c in ("N0_pure_negative","W05_shrink","W15_extrapolate","W20_extrapolate"):
            if any(t["masked_token_count"]!=0 or t["changed_indices"] for t in arms[c]["replan_traces"]): failures.append({"episode_id":pair,"reason":f"self arm changed tokens {c}"})
        ref_trace=arms["REF_toward_top8"]["replan_traces"]
        if any(len(t["selected_indices"])!=8 or sorted(t["selected_indices"])!=sorted(t["changed_indices"]) or not t["protected_tokens_untouched"] for t in ref_trace): failures.append({"episode_id":pair,"reason":"REF token integrity"})
        pairs.append({"pair_id":pair,"task_id":ref["task_id"],"init_state_id":ref["init_state_id"],**arms})
    if len(grouped)!=100 or len(pairs)!=100 or failures:
        (art/"analysis_integrity_failure.json").write_text(json.dumps({"failures":failures,"pairs":len(pairs)},indent=2)+"\n"); raise RuntimeError("analysis integrity failure")
    results={}
    flat=[]
    for task in (4,7):
        rows=[r for r in pairs if r["task_id"]==task]; comps={}
        for i,(first,second) in enumerate(COMPARES): comps[f"{first}_minus_{second}"]=compare(rows,first,second,20260809+task*100+i)
        rates={c:float(np.mean([r[c]["success"] for r in rows])) for c in ARMS}; results[str(task)]={"states":len(rows),"success_rates":rates,"comparisons":comps}
        for r in rows:
            out={"pair_id":r["pair_id"],"task_id":task,"init_state_id":r["init_state_id"]}
            for c in ARMS:
                q=r[c]
                for k in ("success","control_steps","action_total_variation","action_total_variation_first_30_steps","chunk_discontinuity","median_replan_latency_seconds","fraction_of_flow_steps_with_guidance_skipped","fraction_of_flow_steps_with_relative_self_active","fraction_of_flow_steps_with_timestep_shift_active"): out[f"{c}_{k}"]=q.get(k,0.0)
            flat.append(out)
    pvals={f"task{task}:{name}":results[str(task)]["comparisons"][name]["exact_mcnemar_p"] for task in (4,7) for name in ("W15_extrapolate_minus_A_vanilla","W20_extrapolate_minus_A_vanilla")}; adjusted=holm(pvals)
    for key,value in adjusted.items(): task,name=key.split(":"); results[task.removeprefix("task")]["comparisons"][name]["holm_adjusted_p"]=value
    def cr(task,name):return results[str(task)]["comparisons"][name]
    n0=[cr(t,"N0_pure_negative_minus_A_vanilla") for t in (4,7)]; w15=[cr(t,"W15_extrapolate_minus_A_vanilla") for t in (4,7)]; w20=[cr(t,"W20_extrapolate_minus_A_vanilla") for t in (4,7)]; w05=[cr(t,"W05_shrink_minus_A_vanilla") for t in (4,7)]
    if any(contains_zero(x) or x["point_estimate"]>=0 for x in n0): decision="NEGATIVE_BRANCH_INVALID"
    elif all(x["point_estimate"]<0 and x["paired_bootstrap_95_ci"][1]<0 for x in n0) and all(x["point_estimate"]>0 and x["paired_bootstrap_95_ci"][0]>0 for x in w15): decision="CONTRAST_MECHANISM_ESTABLISHED"
    elif all(x["point_estimate"]>0 and x["paired_bootstrap_95_ci"][0]>0 for x in w05) and all(contains_zero(x) for x in w15): decision="ENSEMBLE_AGAIN"
    elif all(x["point_estimate"]<0 and x["paired_bootstrap_95_ci"][1]<0 for x in n0) and all(contains_zero(x) for x in w05+w15+w20): decision="VALID_NEGATIVE_BUT_NO_GAIN"
    else: decision="INCONCLUSIVE"
    pooled={c:float(np.mean([r[c]["success"] for r in pairs])) for c in ARMS}
    protocol=__import__("yaml").safe_load((art/"protocol.lock.yaml").read_text())
    calibration=json.loads((art/"phase0_calibration.json").read_text()); selected=calibration["selected"]
    summary={"experiment_name":protocol["experiment_name"],"stage":"mechanism_development_not_confirmation","completion":{"episodes":600,"paired_units":100,"task4_pairs":50,"task7_pairs":50,"missing":0,"duplicates":0},"integrity_pass":True,"locked_delta":selected.get("delta"),"locked_alpha":selected.get("alpha"),"task_results":results,"pooled_success_rates_descriptive_only":pooled,"holm_family":{"raw_p":pvals,"adjusted_p":adjusted},"decision":decision,"confirmation_claim_allowed":False}
    (art/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); (art/"decision.json").write_text(json.dumps({"decision":decision,"integrity_pass":True,"confirmation_claim_allowed":False},indent=2)+"\n")
    with (art/"paired_results.csv").open("x",newline="") as f: w=csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    with (art/"missingness.csv").open("x",newline="") as f: csv.DictWriter(f,fieldnames=("episode_id","reason")).writeheader()
    (art/"report.md").write_text(f"# CoreAct Self-Guidance Negative Branch\n\nThis is mechanism development, not confirmation. E1 was INCONCLUSIVE; no confirmation claim is allowed.\n\nEpisodes: 600/600; paired units: 100/100; missingness: 0.\n\n```json\n{json.dumps(results,indent=2,sort_keys=True)}\n```\n\nPooled rates are descriptive only: {json.dumps(pooled,sort_keys=True)}.\n\nDecision: `{decision}`.\n",encoding="utf-8")
    (art/"reproduction_commands.sh").write_text(f"#!/usr/bin/env bash\nset -euo pipefail\nWORKSPACE={art.parent.parent}\nARTIFACT={art}\nPYTHON=$WORKSPACE/task1/.conda-envs/flow-vla/bin/python\nexport HF_HOME=$WORKSPACE/task1/.hf-cache TRANSFORMERS_CACHE=$WORKSPACE/task1/.hf-cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYTHONPATH=$WORKSPACE:$WORKSPACE/lerobot/src:$WORKSPACE/LIBERO\n$PYTHON -m research.coreact_self_guidance.integrity --workspace $WORKSPACE --artifact $ARTIFACT\nfor shard in 0 1 2 3 4 5; do $PYTHON -m research.coreact_self_guidance.run --workspace $WORKSPACE --artifact $ARTIFACT --shard-index $shard --shard-count 6; done\n$PYTHON -m research.coreact_self_guidance.analyze --artifact $ARTIFACT\n",encoding="utf-8")
    audit={str(path.relative_to(art)):sha256(path) for path in sorted(art.rglob("*")) if path.is_file() and path.name!="sha256_audit.json"}; (art/"sha256_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); (art/"status/analysis.complete").touch(exist_ok=False); print(json.dumps(summary,indent=2,sort_keys=True))


if __name__ == "__main__":main()
