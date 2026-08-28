#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,random,statistics
from collections import defaultdict
from pathlib import Path
SELECTORS=('attention','random')
def mean(x):return sum(x)/len(x)
def ci(values,fn,seed,n=10000):
 rng=random.Random(seed);boot=sorted(fn([values[rng.randrange(len(values))] for _ in values]) for _ in range(n));return [boot[249],boot[9749]]
def pairwise_jaccard(groups):
 pairs=[(0,1),(0,2),(1,2)];return mean([len(set(groups[a])&set(groups[b]))/len(set(groups[a])|set(groups[b])) for a,b in pairs])
def write_csv(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();art=a.artifact.resolve();manifest=[json.loads(x) for x in (art/'unit_manifest.jsonl').read_text().splitlines()];files=list((art/'units').glob('*.json'));by_id={x['unit_id']:x for x in manifest};bad=[];units={}
 for path in files:
  x=json.loads(path.read_text());uid=x['unit_id']
  if uid in units:bad.append(f'duplicate:{uid}')
  units[uid]=x
  if uid not in by_id or not x['finite'] or x['eligible_visual_count']!=128 or len(x['per_tau'])!=10 or [z['tau'] for z in x['per_tau']]!=[round(1-.1*i,1) for i in range(10)]:bad.append(f'unit:{uid}')
  for selector in SELECTORS:
   if len(x['selected_indices'][selector])!=8 or sorted(x['selected_indices'][selector])!=sorted(x['changed_indices'][selector]):bad.append(f'selection:{uid}:{selector}')
 missing=sorted(set(by_id)-set(units));extra=sorted(set(units)-set(by_id));groups=defaultdict(list)
 for x in units.values():groups[x['state_id']].append(x)
 for state,xs in groups.items():
  if len(xs)!=3 or {x['noise_ordinal'] for x in xs}!={0,1,2}:bad.append(f'noise:{state}')
  for field in ('sim_state_sha256','camera1_sha256','camera2_sha256','preprocessing_sha256','prefix_sha256','random_selection_seed'):
   if len({json.dumps(x[field],sort_keys=True) for x in xs})!=1:bad.append(f'identity:{state}:{field}')
 integrity={'pass':len(files)==1500 and len(groups)==500 and not missing and not extra and not bad,'units':len(files),'states':len(groups),'tasks':len({x['task_id'] for x in units.values()}),'missing':missing,'extra':extra,'bad':bad}
 if not integrity['pass']:(art/'summary.json').write_text(json.dumps({'decision':'FAILED_INTEGRITY','integrity':integrity},indent=2)+'\n');raise RuntimeError('integrity failed')
 state_rows=[]
 for state,xs in sorted(groups.items()):
  xs=sorted(xs,key=lambda x:x['noise_ordinal']);base={'state_id':state,'task_id':xs[0]['task_id'],'demo_ordinal':xs[0]['demo_ordinal'],'phase':xs[0]['phase'],'target_progress':xs[0]['target_progress'],'frame':xs[0]['resolved_frame']}
  for selector in SELECTORS:
   all_steps=[z[selector] for x in xs for z in x['per_tau']];unit_delta=[mean([z[selector]['delta_toward_mse'] for z in x['per_tau']]) for x in xs];delta=mean([z['delta_toward_mse'] for z in all_steps]);away=mean([z['delta_away_mse'] for z in all_steps]);cos=mean([z['cos_toward_expert'] for z in all_steps]);relative=mean([z['delta_toward_mse']/max(z['clean_mse'],1e-12) for z in all_steps]);base[f'{selector}_mean_cos']=cos;base[f'{selector}_mean_delta_mse']=delta;base[f'{selector}_mean_relative_delta']=relative;base[f'{selector}_toward_positive']=int(delta<0);base[f'{selector}_cos_positive']=int(cos>0);base[f'{selector}_noise_sign_stable']=int(all(v<0 for v in unit_delta) or all(v>0 for v in unit_delta));base[f'{selector}_noise_positive_count']=sum(v<0 for v in unit_delta);base[f'{selector}_best_direction']=min((delta,'toward'),(away,'away'),(0.,'off'))[1]
  base['attention_selection_jaccard_across_noise']=pairwise_jaccard([x['selected_indices']['attention'] for x in xs]);state_rows.append(base)
 write_csv(art/'state_results.csv',state_rows);task_rows=[]
 for task in range(10):
  rs=[x for x in state_rows if x['task_id']==task];row={'task_id':task,'states':len(rs)}
  for selector in SELECTORS:
   positives=[x[f'{selector}_toward_positive'] for x in rs];row[f'{selector}_toward_positive_fraction']=mean(positives);row[f'{selector}_toward_positive_ci_low'],row[f'{selector}_toward_positive_ci_high']=ci(positives,mean,20260815+task*10+(selector=='random'));row[f'{selector}_mean_cos']=mean([x[f'{selector}_mean_cos'] for x in rs]);row[f'{selector}_mean_delta_mse']=mean([x[f'{selector}_mean_delta_mse'] for x in rs]);row[f'{selector}_mean_relative_delta']=mean([x[f'{selector}_mean_relative_delta'] for x in rs]);row[f'{selector}_noise_sign_stable_fraction']=mean([x[f'{selector}_noise_sign_stable'] for x in rs]);row[f'{selector}_toward_best_fraction']=mean([x[f'{selector}_best_direction']=='toward' for x in rs]);row[f'{selector}_away_best_fraction']=mean([x[f'{selector}_best_direction']=='away' for x in rs]);row[f'{selector}_off_best_fraction']=mean([x[f'{selector}_best_direction']=='off' for x in rs])
  row['attention_minus_random_positive_fraction']=row['attention_toward_positive_fraction']-row['random_toward_positive_fraction'];task_rows.append(row)
 write_csv(art/'task_results.csv',task_rows);phase_rows=[]
 for phase in ('early','middle','late'):
  rs=[x for x in state_rows if x['phase']==phase];phase_rows.append({'phase':phase,'states':len(rs),**{f'{s}_toward_positive_fraction':mean([x[f'{s}_toward_positive'] for x in rs]) for s in SELECTORS},**{f'{s}_mean_delta_mse':mean([x[f'{s}_mean_delta_mse'] for x in rs]) for s in SELECTORS}})
 write_csv(art/'phase_results.csv',phase_rows);tau_rows=[]
 for tau in [round(1-.1*i,1) for i in range(10)]:
  vals={s:[] for s in SELECTORS}
  for x in units.values():
   z=next(y for y in x['per_tau'] if y['tau']==tau)
   for s in SELECTORS:vals[s].append(z[s])
  tau_rows.append({'tau':tau,**{f'{s}_cos_positive_fraction':mean([z['toward_positive_cos'] for z in vals[s]]) for s in SELECTORS},**{f'{s}_toward_improves_fraction':mean([z['toward_improves_mse'] for z in vals[s]]) for s in SELECTORS},**{f'{s}_mean_delta_mse':mean([z['delta_toward_mse'] for z in vals[s]]) for s in SELECTORS}})
 write_csv(art/'tau_results.csv',tau_rows);paired=[x['attention_mean_delta_mse']-x['random_mean_delta_mse'] for x in state_rows];positive_diff=[x['attention_toward_positive']-x['random_toward_positive'] for x in state_rows];global_result={'attention_toward_positive_fraction':mean([x['attention_toward_positive'] for x in state_rows]),'random_toward_positive_fraction':mean([x['random_toward_positive'] for x in state_rows]),'attention_minus_random_positive_fraction':mean(positive_diff),'attention_minus_random_positive_fraction_ci':ci(positive_diff,mean,202608151),'attention_minus_random_delta_mse':mean(paired),'attention_minus_random_delta_mse_ci':ci(paired,mean,202608152),'attention_noise_sign_stable_fraction':mean([x['attention_noise_sign_stable'] for x in state_rows]),'random_noise_sign_stable_fraction':mean([x['random_noise_sign_stable'] for x in state_rows]),'attention_selection_jaccard_across_noise':mean([x['attention_selection_jaccard_across_noise'] for x in state_rows])}
 spread=max(x['attention_toward_positive_fraction'] for x in task_rows)-min(x['attention_toward_positive_fraction'] for x in task_rows);decision='EXPERT_ALIGNED_DIRECTION_HETEROGENEITY_OBSERVED' if spread>=.2 else 'EXPERT_ALIGNED_DIRECTION_HETEROGENEITY_NOT_ESTABLISHED';summary={'integrity':integrity,'global':global_result,'tasks':task_rows,'phases':phase_rows,'decision':decision,'limits':['offline teacher-forced expert-orbit audit; not closed-loop success evidence','development data; no sign predictor trained']};(art/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(art/'decision.json').write_text(json.dumps({'decision':decision,'integrity':'PASS','capture_complete':True},indent=2)+'\n')
 report=['# Expert-Aligned Guidance Direction Audit','',f"Integrity PASS: 1500/1500 units, 500/500 expert states, 10/10 tasks.",'',f"Attention Toward-positive: {global_result['attention_toward_positive_fraction']:.1%}; Random: {global_result['random_toward_positive_fraction']:.1%}.",f"Attention noise-sign stability: {global_result['attention_noise_sign_stable_fraction']:.1%}; selection Jaccard across noise: {global_result['attention_selection_jaccard_across_noise']:.3f}.",'','## Tasks','| Task | Attention positive | Random positive | Attention mean cosine | Attention relative delta | Noise sign stable | Toward/Away/Off best |','|---:|---:|---:|---:|---:|---:|---:|']
 for x in task_rows:report.append(f"| {x['task_id']} | {x['attention_toward_positive_fraction']:.1%} | {x['random_toward_positive_fraction']:.1%} | {x['attention_mean_cos']:.3f} | {x['attention_mean_relative_delta']:+.3f} | {x['attention_noise_sign_stable_fraction']:.1%} | {x['attention_toward_best_fraction']:.0%}/{x['attention_away_best_fraction']:.0%}/{x['attention_off_best_fraction']:.0%} |")
 report+=['','## Decision',decision,'','This audit measures teacher-forced velocity alignment around official expert demonstrations. It does not establish online success effects and does not justify training a sign predictor without further causal validation.'];(art/'report.md').write_text('\n'.join(report)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
