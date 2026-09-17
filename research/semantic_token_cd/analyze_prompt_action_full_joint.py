"""Final paired analysis for PA-Full-Joint-Top2K."""
from __future__ import annotations
import json,math,pickle
from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.prompt_action_full_joint_protocol import ARM,ARTIFACT,CANONICAL,MATCHED_ROOT,SEEDS,TASKS,atomic_json
def pexact(r,h):
    n=r+h
    return 1.0 if not n else min(1.0,2*sum(math.comb(n,i) for i in range(min(r,h)+1))/2**n)
def main():
    result={"complete":True,"new_episodes":400,"tasks":{}}; totals=[0]*5
    for task in TASKS:
        vals=[]
        for seed in SEEDS:
            f=ARTIFACT/"closed_loop/episodes"/task/ARM/f"episode_{seed:03d}_summary.json"; a=json.loads(f.read_text()); b=json.loads((MATCHED_ROOT[task]/f"episode_{seed:03d}_summary.json").read_text())
            with (CANONICAL/"snapshots"/task/f"seed_{seed:03d}.pkl").open("rb") as handle: expected=snapshot_sha(pickle.load(handle))
            if a.get("canonical_snapshot_sha256")!=expected or b.get("canonical_snapshot_sha256")!=expected or a.get("technical_pass") is not True: raise RuntimeError(f"audit failure: {f}")
            vals.append((bool(a["success"]),bool(b["success"])))
        n=len(vals); s=sum(a for a,b in vals); bs=sum(b for a,b in vals); r=sum(a and not b for a,b in vals); h=sum(b and not a for a,b in vals)
        result["tasks"][task]={"n":n,"full_joint_success":s,"l11_success":bs,"rescue":r,"harm":h,"net":r-h,"p_exact":pexact(r,h)}
        for i,z in enumerate((n,s,bs,r,h)): totals[i]+=z
    n,s,bs,r,h=totals; result["overall"]={"n":n,"full_joint_success":s,"l11_success":bs,"rescue":r,"harm":h,"net":r-h,"p_exact":pexact(r,h)}; atomic_json(ARTIFACT/"FINAL_RESULTS.json",result)
    lines=["# PA-Full-Joint-Top2K", "", "| Task | L11-Matched | Full-Joint | Delta | Rescue | Harm |", "|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        x=result["tasks"][task]; lines.append(f"| {task.removeprefix('google_robot_')} | {x['l11_success']}/100 | {x['full_joint_success']}/100 | {x['net']:+d} | {x['rescue']} | {x['harm']} |")
    lines.append(f"| overall | {bs}/400 | {s}/400 | {r-h:+d} | {r} | {h} |"); lines.append(f"\nExact paired p={result['overall']['p_exact']:.4g}.\n"); (ARTIFACT/"FINAL_REPORT.md").write_text("\n".join(lines)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
