"""Final paired analysis against reused held-out L11-Matched."""
from __future__ import annotations
import json
from scipy.stats import binomtest
from research.semantic_token_cd.target_boost_confidence_protocol import ARMS,ARTIFACT,BASELINE,TASKS

def load(path):return json.loads(path.read_text())
def paired(base,cand):
 rescue=sum((not b) and c for b,c in zip(base,cand));harm=sum(b and (not c) for b,c in zip(base,cand));n=rescue+harm
 return {"rescue":rescue,"harm":harm,"net":rescue-harm,"mcnemar_exact_p":float(binomtest(min(rescue,harm),n,.5).pvalue) if n else 1.0}
def main():
 result={"protocol_id":"TARGET_BOOST_CONFIDENCE_GATE_V1","complete":True,"identity_pass":True,"tasks":{}}
 all_values={"l11_matched":[],**{a:[] for a in ARMS}}
 for task in TASKS:
  base=[];values={a:[] for a in ARMS};diagnostics={a:[] for a in ARMS}
  for seed in range(100,200):
   b=load(BASELINE/"episodes"/task/"l11_matched"/f"episode_{seed:03d}_summary.json");base.append(bool(b["result"]["success"]))
   for arm in ARMS:
    c=load(ARTIFACT/"episodes"/task/arm/f"episode_{seed:03d}_summary.json")
    for key in ("canonical_snapshot_sha256","initial_state_sha256","initial_rgb_sha256"):
     if b[key]!=c[key]:result["identity_pass"]=False
    if not c["technical_pass"]:raise RuntimeError(f"technical fail {task}/{arm}/{seed}")
    values[arm].append(bool(c["success"]));diagnostics[arm].append(c)
  task_result={"n":100,"l11_matched_success":sum(base),"arms":{}}
  all_values["l11_matched"]+=base
  for arm in ARMS:
   all_values[arm]+=values[arm];task_result["arms"][arm]={"success":sum(values[arm]),"vs_matched":paired(base,values[arm]),
    "mean_correct_retention":sum(x["mean_correct_retention"] for x in diagnostics[arm])/100,
    "mean_gate_eligible_fraction":sum(x["mean_gate_eligible_fraction"] for x in diagnostics[arm])/100}
  task_result["gate_vs_ungated"]=paired(values["positive_boost_0p5"],values["confidence_gate_0p4"]);result["tasks"][task]=task_result
 result["overall"]={"n":400,"l11_matched_success":sum(all_values["l11_matched"]),"arms":{}}
 for arm in ARMS:result["overall"]["arms"][arm]={"success":sum(all_values[arm]),"vs_matched":paired(all_values["l11_matched"],all_values[arm])}
 result["overall"]["gate_vs_ungated"]=paired(all_values["positive_boost_0p5"],all_values["confidence_gate_0p4"])
 (ARTIFACT/"FINAL_RESULTS.json").write_text(json.dumps(result,indent=2)+"\n")
 lines=["# Confidence-Gated Target Positive-Boost","",f"Snapshot/hash identity: **{'PASS' if result['identity_pass'] else 'FAIL'}**","",
  "| Task | L11-Matched | Ungated-0.5 | Gated-0.4 |","|---|---:|---:|---:|"]
 for task in TASKS:
  x=result["tasks"][task];lines.append(f"| {task.removeprefix('google_robot_')} | {x['l11_matched_success']}/100 | {x['arms']['positive_boost_0p5']['success']}/100 | {x['arms']['confidence_gate_0p4']['success']}/100 |")
 o=result["overall"];lines.append(f"| **Overall** | **{o['l11_matched_success']}/400** | **{o['arms']['positive_boost_0p5']['success']}/400** | **{o['arms']['confidence_gate_0p4']['success']}/400** |")
 lines += ["","## Overall paired","","| Comparison | Rescue | Harm | Net | exact p |","|---|---:|---:|---:|---:|"]
 for label,p in (("Ungated vs Matched",o["arms"]["positive_boost_0p5"]["vs_matched"]),("Gated vs Matched",o["arms"]["confidence_gate_0p4"]["vs_matched"]),("Gated vs Ungated",o["gate_vs_ungated"])):
  lines.append(f"| {label} | {p['rescue']} | {p['harm']} | {p['net']:+d} | {p['mcnemar_exact_p']:.5g} |")
 (ARTIFACT/"REPORT.md").write_text("\n".join(lines)+"\n");print(json.dumps(result["overall"]))
if __name__=="__main__":main()
