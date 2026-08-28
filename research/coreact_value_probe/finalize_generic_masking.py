#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
CONDS=("vanilla","attention_top8","random8","bottom8")
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for x in iter(lambda:f.read(1048576),b""):h.update(x)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();s=json.loads((o/"analysis.json").read_text());i=json.loads((o/"post_integrity_report.json").read_text())
 lines=["# Generic Early Masking Replication","","## Question","","Does masking eight visual tokens for only the first three replans improve closed-loop success regardless of whether tokens are selected by top attention, random selection, or bottom attention?","","## Locked scope","","- New suite: LIBERO-Goal; tasks 0, 1, 2; canonical init states 0-49.","- 150 paired states, 600 episodes, four conditions.","- Development replication only: each task has only 50 canonical states, so this is not a 100-state confirmation.","- No value probe, consistency classifier, sign classifier, or guidance.","","## Integrity","",f"- Episodes: {i['episodes']}/600; complete pairs: {i['pairs']}/150; status: {'PASS' if i['pass'] else 'FAIL'}.","- Frozen deterministic sampler; exactly eight valid visual tokens changed; paired state/reset/noise/preprocessing/solver.","","## Success","","| Condition | Overall | Task 0 | Task 1 | Task 2 | Difference vs vanilla | Cluster-bootstrap 95% CI | Rescued / harmed |","|---|---:|---:|---:|---:|---:|---:|---:|"]
 for c in CONDS:
  r=s["rates"][c];tasks=r["by_task"]
  if c=="vanilla":lines.append(f"| {c} | {r['successes']}/{r['n']} ({100*r['rate']:.1f}%) | {100*tasks['0']:.0f}% | {100*tasks['1']:.0f}% | {100*tasks['2']:.0f}% | - | - | - |")
  else:
   z=s["comparisons"][c];lines.append(f"| {c} | {r['successes']}/{r['n']} ({100*r['rate']:.1f}%) | {100*tasks['0']:.0f}% | {100*tasks['1']:.0f}% | {100*tasks['2']:.0f}% | {100*z['minus_vanilla']:+.1f} pp | [{100*z['cluster_bootstrap_ci95'][0]:+.1f}, {100*z['cluster_bootstrap_ci95'][1]:+.1f}] pp | {z['rescued']} / {z['harmed']} |")
 lines += ["","## Mechanism diagnostics","","| Condition | Mean matched-noise action delta norm | Token Jaccard across replans | Action TV | Chunk discontinuity | Steps |","|---|---:|---:|---:|---:|---:|"]
 for c in CONDS[1:]:
  m=s["mechanism"][c];r=s["rates"][c];lines.append(f"| {c} | {m['mean_mask_clean_action_norm']:.3f} | {m['mean_token_jaccard']:.3f} | {r['mean_action_tv']:.2f} | {r['mean_chunk_discontinuity']:.3f} | {r['mean_steps']:.1f} |")
 lines += ["",f"## Decision","",f"**{s['decision']}**","","Interpretation is limited to this new three-task development suite. The next experiment is duration plus matched-action perturbation only if a generic masking direction replicates; otherwise the task-8 improvement remains task-specific and the next route is a real closed-loop causal-token dataset."]
 (o/"report.md").write_text("\n".join(lines)+"\n");files=sorted(x for x in o.rglob("*") if x.is_file() and x.name!="sha256.audit")
 with (o/"sha256.audit").open("w") as f:
  for x in files:f.write(f"{sha(x)}  {x.relative_to(o)}\n")
if __name__=="__main__":main()
