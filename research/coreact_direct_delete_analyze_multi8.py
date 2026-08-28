from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from research.ar_token_closed_loop.common import file_sha256, write_json

CONDS = ("vanilla", "top8_mask_only", "away", "toward")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args(); art=args.artifact.resolve()
    rows=[json.loads(p.read_text()) for p in (art/"episodes").glob("*.json")]
    by={}
    for row in rows: by.setdefault(row["pair_id"], {})[row["condition"]]=row
    if len(rows)!=1000 or len(by)!=250 or any(set(group)!=set(CONDS) for group in by.values()): raise RuntimeError("incomplete paired artifact")
    rates={c:float(np.mean([group[c]["success"] for group in by.values()])) for c in CONDS}
    task_rates={}
    for task in (0,1,2,3,5):
        task_rates[str(task)]={c:float(np.mean([group[c]["success"] for group in by.values() if group[c]["task_id"]==task])) for c in CONDS}
    comparisons={}
    base=np.array([group["vanilla"]["success"] for group in by.values()],int)
    for c in CONDS[1:]:
        treated=np.array([group[c]["success"] for group in by.values()],int); a=int(((base==1)&(treated==0)).sum()); b=int(((base==0)&(treated==1)).sum())
        comparisons[c]={"delta":rates[c]-rates["vanilla"],"vanilla_only":a,"condition_only":b,"discordant":a+b,"mcnemar_p":1.0 if a+b==0 else float(binomtest(min(a,b),n=a+b).pvalue),"mean_replans":float(np.mean([group[c]["replans"] for group in by.values()])),"all_replans_masked":all(group[c]["all_replans_masked"] for group in by.values())}
    summary={"episodes":len(rows),"paired_states":len(by),"success_rates":rates,"task_success_rates":task_rates,"comparisons":comparisons,"decision":"FLOW_DIRECT_DELETE8_MULTI_TASK_COMPLETE","scope":"five-task development replication; direct prefix deletion, not embedding replacement"}
    write_json(art/"summary.json",summary); write_json(art/"decision.json",{"decision":summary["decision"],"integrity_pass":True}); (art/"report.md").write_text("# Flow Direct-Delete8 Multi-Task Every-Replan Replication\n\n"+json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); write_json(art/"sha256_audit.json",{str(p.relative_to(art)):file_sha256(p) for p in sorted(art.rglob("*")) if p.is_file() and p.name!="sha256_audit.json"}); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
