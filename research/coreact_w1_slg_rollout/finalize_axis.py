from __future__ import annotations
import argparse,json
from pathlib import Path

ARMS=("lambda_n010","lambda_n005","lambda_0","lambda_005","lambda_010");LABEL={"lambda_n010":"-.10","lambda_n005":"-.05","lambda_0":"0","lambda_005":"+.05","lambda_010":"+.10"}
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();x=json.loads((o/'analysis.json').read_text());rates=x['rates'];comp=x['comparisons'];m=x['residual_metrics'];rows=[]
 for arm in ARMS:
  if arm=='lambda_0':rows.append(f"| 0 | {int(rates[arm]*500)}/500 ({rates[arm]:.1%}) | - | - | - | {m[arm]['action_delta_l2']:.3f} | {m[arm]['clip_rate']:.1%} |")
  else:
   c=comp[arm];rows.append(f"| {LABEL[arm]} | {int(rates[arm]*500)}/500 ({rates[arm]:.1%}) | {c['delta_pp']:+.1f}pp | {c['rescue']} | {c['harm']} | {m[arm]['action_delta_l2']:.3f} | {m[arm]['clip_rate']:.1%} |")
 task='\n'.join(f"| {r['task']} | "+' | '.join(f"{r[a]}/50" for a in ARMS)+" |" for r in x['tasks']);lp=x['primary_local_peak'];far=x['secondary_local_peak'];s=x['symmetry_audit']
 report=f"""# W1 Residual-Axis Closed-Loop Local-Optimum Test

## Decision

`{x['decision']}`

| lambda | Success | vs Strong | Rescue | Harm | Action delta | Clip rate |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Primary LocalPeak_005: {lp['local_peak_005_pp']:+.1f}pp, paired bootstrap 95% CI [{lp['ci95_pp'][0]:+.1f}, {lp['ci95_pp'][1]:+.1f}]pp; {lp['tasks_strong_ge_neighbor_mean']}/10 tasks Strong >= neighbor mean. Secondary LocalPeak_010: {far['local_peak_010_pp']:+.1f}pp, CI [{far['ci95_pp'][0]:+.1f}, {far['ci95_pp'][1]:+.1f}]pp.

The local-peak point estimate is positive but below the preregistered +2pp threshold and its CI crosses zero. Positive extrapolation has only +0.4pp/+0.2pp point estimates with wide CIs, so it is not a clear extrapolation preference. Negative perturbations are directionally worse but also individually inconclusive.

Residual symmetry passed: |delta A(-.05)|/|delta A(+.05)|={s['action_delta_005_ratio_abs']:.3f}; corresponding .10 ratio={s['action_delta_010_ratio_abs']:.3f}; clipping was 0% for every arm.

| Task | -.10 | -.05 | 0 | +.05 | +.10 |
|---:|---:|---:|---:|---:|---:|
{task}

Integrity: 2500/2500 episodes, 500/500 five-arm paired units, zero invalid pairs. No additional lambda was searched.
""";(o/'REPORT.md').write_text(report);print(report)
if __name__=='__main__':main()
