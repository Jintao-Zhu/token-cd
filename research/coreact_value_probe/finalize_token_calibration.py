#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for x in iter(lambda:f.read(1048576),b""):h.update(x)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();s=json.loads((o/"analysis.json").read_text());d=json.loads((o/"decision.json").read_text())
 lines=["# Signed Value Token Causal Calibration","","## Locked experiment","","- LIBERO-Spatial task 8, init states 0-49, 50 complete paired states.","- Conditions: vanilla, mask strong-positive utility, mask strong-negative utility, mask near-zero utility.","- Attention only proposed top-32 visual tokens. Linear probe utility selected three disjoint 8-token groups.","- Mask-only for the first three replans; all branches then returned to vanilla. No guidance.","","## Integrity","","- 200/200 episodes and 50/50 complete pairs.","- Frozen eval model; deterministic utility/selection; 32 finite candidates; three disjoint 8-token visual groups.","- Paired init state, reset seed, action-noise derivation, preprocessing, and solver were locked before outcomes.","","## Success results","","| Condition | Success | Difference vs vanilla | Paired bootstrap 95% CI | McNemar exact p |","|---|---:|---:|---:|---:|"]
 for c in ("vanilla","mask_strong_positive","mask_strong_negative","mask_near_zero"):
  x=s[c]
  if c=="vanilla":lines.append(f"| {c} | {x['successes']}/50 ({100*x['success_rate']:.0f}%) | - | - | - |")
  else:
   z=x["minus_vanilla"];lines.append(f"| {c} | {x['successes']}/50 ({100*x['success_rate']:.0f}%) | {100*z['point']:+.0f} pp | [{100*z['bootstrap']['ci95'][0]:+.0f}, {100*z['bootstrap']['ci95'][1]:+.0f}] pp | {z['mcnemar_exact_two_sided']:.4f} |")
 overall=s["all_groups_utility_vs_causal_harm_spearman"]
 lines += ["","Strong-negative minus strong-positive success: " + f"{100*s['mask_strong_negative_minus_positive']['point']:+.0f} pp, 95% CI [{100*s['mask_strong_negative_minus_positive']['bootstrap']['ci95'][0]:+.0f}, {100*s['mask_strong_negative_minus_positive']['bootstrap']['ci95'][1]:+.0f}] pp.",f"Across all 150 masked groups, utility versus binary causal harm Spearman rho={overall['rho']:.3f}, p={overall['pvalue']:.3f}.","","## Interpretation","","The signed token-utility hypothesis failed its causal calibration. Tokens predicted to be strong positive anchors did not become harmful to remove; their mask condition had the highest observed success rate. Strong-negative masks were not better than strong-positive masks, and near-zero masks also improved over vanilla. The common improvement across all mask types is more consistent with a generic masking/regularization or trajectory perturbation effect than with correct signed utility.","","The single-state probe result remains valid only for predicting vanilla rollout success. It does not transfer into a reliable counterfactual value difference on masked internal representations. Probe saturation near 1.0 and imperfect calibration make this failure plausible.","",f"**Decision: {d['decision']}**","","Do not build CoreAct guidance from this selector. A next attempt would need a critic trained on actual intervention rollouts (or a trajectory-conditioned critic), with task-held-out causal labels; simply applying the vanilla success probe to masked latents is not enough."]
 (o/"report.md").write_text("\n".join(lines)+"\n")
 files=sorted(x for x in o.rglob("*") if x.is_file() and x.name!="sha256.audit")
 with (o/"sha256.audit").open("w") as f:
  for x in files:f.write(f"{sha(x)}  {x.relative_to(o)}\n")
if __name__=="__main__":main()
